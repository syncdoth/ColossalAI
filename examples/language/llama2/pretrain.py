import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import argparse
import os
import resource
from contextlib import nullcontext
from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from attn import SUPPORT_XFORMERS, replace_xformers
from data_utils import load_json, prepare_dataloader, save_json
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
# from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
from accelerate import Accelerator

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from model_utils import format_numel_str, get_model_numel

# retnet
from retnet.retnet.modeling_retnet import RetNetForCausalLM
from retnet.retnet.configuration_retnet import RetNetConfig

# nucleus
from nucleus import data_loader
from nucleus.utils import get_lr_scheduler, init_dataloader_decoder_utils

LLAMA_CONFIGS = {
    "7b": LlamaConfig(max_position_embeddings=4096),
    "13b": LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        max_position_embeddings=4096,
    ),
    "70b": LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        max_position_embeddings=4096,
        num_key_value_heads=8,
    ),
}

MODEL_CLASS = {
    "llama": LlamaForCausalLM,
    "retnet": RetNetForCausalLM,
}


def tokenize_batch_for_pretrain(collate_fn, batch):
    input_ids, labels, attention_mask = collate_fn(batch)
    data = {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
        "labels": labels.cuda(),
    }
    return data


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def save(
    booster: Booster,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    batch_size: int,
    coordinator: DistCoordinator,
    save_dir: str,
):
    save_dir = os.path.join(save_dir, f"epoch{epoch}-step{step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
    booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
    booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    running_states = {
        "epoch": epoch,
        "step": step,
        "sample_start_index": step * batch_size,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))


def load(
    booster: Booster, model: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler, load_dir: str
) -> Tuple[int, int, int]:
    booster.load_model(model, os.path.join(load_dir, "model"))
    booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
    booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    return running_states["epoch"], running_states["step"], running_states["sample_start_index"]


def _criterion(outputs, inputs):
    return outputs.loss


def get_config(model_name, config_name, **kwargs):
    if model_name == "llama":
        return LLAMA_CONFIGS[config_name]
    elif model_name == "retnet":
        return RetNetConfig.from_pretrained(config_name, **kwargs)
    else:
        raise ValueError(f"Unknown model {model_name}")


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="7b", help="Model configuration")
    parser.add_argument("-m", "--model_name", type=str, default="llama", choices=["llama", "retnet"],
                        help="which model to pretrain")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="which tokenizer to use")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "hybrid_parallel"],
        default="gemini",
        help="Choose which plugin to use",
    )
    parser.add_argument("--max_iters", type=int, default=1e6, help="Max number of iterations")
    parser.add_argument("-e", "--num_epochs", type=int, default=0, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2048, help="Local batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("-w", "--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("-s", "--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--block_size", type=int, default=2048, help="Max sequence length")
    parser.add_argument("-x", "--mixed_precision", default="fp16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("-i", "--save_interval", type=int, default=1000, help="Save interval")
    parser.add_argument("-o", "--save_dir", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("-f", "--load", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--wandb_dir", type=str, default="/home/opsai", help="wandb directory")
    parser.add_argument("--run_name", type=str, default="retnet-3D", help="wandb run name")
    parser.add_argument("-a", "--flash_attention", action="store_true", help="Use Flash Attention")
    # shardformer related
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=2, help="Pipeline parallel size")
    parser.add_argument("--num_pp_mbs", type=int, default=2, help="Number of micro batches in pipeline parallel")
    parser.add_argument("--micro_batch_size", type=int, default=1024, help="Micro batch size (related to grad accum)")
    parser.add_argument("--zero_stage", type=int, default=0, choices=[0, 1, 2], help="zero stage (0, 1, 2)")
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    # data
    parser.add_argument("--datasets", type=str, default="wikipedia", help="comma-separated list of dataset names")
    parser.add_argument("--dataset_weights", type=str, default=None, help="comma-separated list of dataset weights")
    args = parser.parse_args()

    # ==============================
    # Parse comma-separated Arguments
    # ==============================
    args.datasets = args.datasets.split(",")
    if args.dataset_weights is not None:
        args.dataset_weights = tuple(float(x) for x in args.dataset_weights.split(","))
        assert (
            len(args.datasets) == len(args.dataset_weights)
        ), "expected equal number of dataset names and weights"

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    # ==============================
    # compute batch size correctly
    # ==============================
    grad_accum_step = args.batch_size // args.micro_batch_size
    assert grad_accum_step == 1, "grad_accum_step must be 1 for now"

    if args.pp > 1:
        per_device_batch_size = args.micro_batch_size // (coordinator.world_size * args.num_pp_mbs)
        dataloader_batch_size = per_device_batch_size * args.num_pp_mbs
    else:
        per_device_batch_size = args.micro_batch_size // coordinator.world_size
        dataloader_batch_size = per_device_batch_size

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "gemini":
        plugin = GeminiPlugin(precision=args.mixed_precision, initial_scale=2**16, max_norm=args.grad_clip)
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision, placement_policy="auto", initial_scale=2**16, max_norm=args.grad_clip
        )
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, max_norm=args.grad_clip
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, cpu_offload=True, max_norm=args.grad_clip
        )
    elif args.plugin == "hybrid_parallel":
        # modify the param accordingly, default configuration is for llama2-7b
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            num_microbatches=args.num_pp_mbs,
            enable_jit_fused=False,
            zero_stage=args.zero_stage,
            precision=args.mixed_precision,
            initial_scale=1,  # TODO or 2**8
            cpu_offload=args.offload,
            enable_fused_normalization=True,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_stage)

    # ==============================
    # Initialize Tensorboard
    # ==============================
    if print_flag:
        # os.makedirs(args.tensorboard_dir, exist_ok=True)
        # writer = SummaryWriter(args.tensorboard_dir)
        wandb_args = {
            "project": "RetNet Chronicles",  # "iqmt1xcu"
            "resume": False,  # "allow"
            "dir": args.wandb_dir,
            "config": args,
            "mode": "disabled",
            "name": args.run_name + f"-{coordinator.rank}",
            "group": args.run_name,
        }
        wandb.init(**wandb_args)

    # ==============================
    # Initialize Tokenizer
    # ==============================
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # follows fast chat: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py#L257
    # tokenizer.pad_token = tokenizer.unk_token
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if not tokenizer.bos_token:
        tokenizer.add_special_tokens({"pad_token": "<s>"})

    # ==============================
    # Initialize Model, Optimizer
    # ==============================
    tokenizer_kwargs = {
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    config = get_config(args.model_name, args.config, **tokenizer_kwargs)
    model_class = MODEL_CLASS[args.model_name]
    # use lazy init when using GeminiPlugin
    init_ctx = (
        LazyInitContext(default_device=get_current_device()) if isinstance(plugin, GeminiPlugin) else nullcontext()
    )

    with init_ctx:
        model = model_class(config, tensor_parallel=args.tp > 1)

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    if args.flash_attention and not 'retnet' in args.model_name:
        assert SUPPORT_XFORMERS, "Use flash attention while xfomers is not installed"
        replace_xformers(model)

    model_numel = get_model_numel(model)
    coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")

    optimizer = HybridAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, eps=1e-15)  # !!

    # ==============================
    # Initialize Data, DataLoader
    # ==============================
    dataloader, _ = data_loader.create_dataloader_decoder(
        batch_size=dataloader_batch_size, block_size=args.block_size,
        tokenizer=tokenizer, datasets=args.datasets, dataset_weights=args.dataset_weights,
        meta_collate_fn=tokenize_batch_for_pretrain)

    # TODO: use accelerator just for data prepare...
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_step,
        split_batches=False,
    )
    dataloader = accelerator.prepare_data_loader(dataloader)

    # tokenizer, collate_fn = init_dataloader_decoder_utils(
    #     padding="longest", max_length=args.block_size, tokenizer=tokenizer,
    # )

    # train_ds = data_loader.CombinedDataset_decoder(
    #     block_size=args.block_size, tokenizer=tokenizer, datasets=args.datasets, dataset_weights=args.dataset_weights,
    #     use_sampler=True
    # )

    # dataloader = prepare_dataloader(
    #     train_ds,
    #     batch_size=dataloader_batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     collate_fn=partial(tokenize_batch_for_pretrain, collate_fn),
    # )
    if args.max_iters > 0:
        total_step = args.max_iters
        coordinator.print_on_master("if max_iters set, ignore num_epochs")
        args.num_epochs = args.max_iters // len(dataloader) + 1
    else:
        total_step = args.num_epochs * len(dataloader)
    # ==============================
    # Initialize LR Scheduler
    # ==============================
    lr_scheduler = get_lr_scheduler(optimizer, optim_name='adamw', warmup_steps=args.warmup_steps,
                                    base_lr=args.lr, min_lr=0.1 * args.lr, total_steps=total_step)
    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    # ==============================
    # during pretrain, seqlen is fixed. So we can get relpos first.
    # ==============================
    pre_retention_rel_pos = model.model.retnet_rel_pos(
        args.block_size,
        forward_impl='parallel',
        get_decay_scale=False,
    )
    retention_rel_pos_device = []
    for rel_pos_group in pre_retention_rel_pos:
        group = []
        for rel_pos in rel_pos_group:
            if rel_pos is not None:
                rel_pos = rel_pos.cuda()
                rel_pos = rel_pos.to(default_dtype)
            group.append(rel_pos)
        retention_rel_pos_device.append(tuple(group))
    pre_retention_rel_pos = tuple(retention_rel_pos_device)
    # ==============================

    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model, optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
    )
    torch.set_default_dtype(torch.float)

    coordinator.print_on_master(f"Booster init max CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )

    # load checkpoint if specified
    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0
    if args.load is not None:
        coordinator.print_on_master("Loading checkpoint")
        start_epoch, start_step, sampler_start_idx = load(booster, model, optimizer, lr_scheduler, args.load)
        coordinator.print_on_master(f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}")

    num_steps_per_epoch = len(dataloader)

    # if resume training, set the sampler start index to the correct value
    # dataloader.sampler.set_start_index(sampler_start_idx)
    for epoch in range(start_epoch, args.num_epochs):
        # dataloader.sampler.set_epoch(epoch)
        step_nums = num_steps_per_epoch - start_step
        dataloader_iter = iter(dataloader)

        with tqdm(
            range(step_nums),
            desc=f"Epoch {epoch}",
            disable=not print_flag,
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step in pbar:
                if use_pipeline:
                    outputs = booster.execute_pipeline(
                        dataloader_iter, model, _criterion, optimizer, return_loss=True, return_outputs=True,
                        pre_retention_rel_pos=pre_retention_rel_pos,
                    )
                    loss = outputs["loss"]
                else:
                    batch = next(dataloader_iter)
                    outputs = model(**batch, retention_rel_pos=pre_retention_rel_pos)
                    loss = outputs[0]
                    booster.backward(loss, optimizer)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if not use_pipeline:
                    all_reduce_mean(loss)
                if print_flag:
                    pbar.set_postfix({"loss": loss.item()})
                    # writer.add_scalar("loss", loss.item(), epoch * num_steps_per_epoch + step)
                    wandb.log({
                        "train_loss_per_batch": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }, step=epoch * num_steps_per_epoch + step)

                if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                    coordinator.print_on_master(f"Saving checkpoint")
                    save(
                        booster,
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        args.batch_size,
                        coordinator,
                        os.path.join(args.save_dir, args.run_name),
                    )
                    coordinator.print_on_master(f"Saved checkpoint at epoch {epoch} step {step + 1}")

                if epoch * num_steps_per_epoch + step >= args.max_iters:
                    break
        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        # dataloader.sampler.set_start_index(0)
        start_step = 0

    coordinator.print_on_master(f"Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    main()
