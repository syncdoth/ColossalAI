import warnings
warnings.filterwarnings("ignore")  # ignore warnings

import argparse
import resource
from contextlib import nullcontext

import torch
from attn import SUPPORT_FLASH, replace_xformers
from data_utils import RandomDataset
from model_utils import format_numel_str, get_model_numel
from performance_evaluator import PerformanceEvaluator
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, TorchFSDPPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext, LazyTensor
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


from retnet.retnet.modeling_retnet import RetNetForCausalLM
from retnet.retnet.configuration_retnet import RetNetConfig

# ==============================
# Constants
# ==============================

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
    parser.add_argument("--model_name", type=str, default='retnet')
    parser.add_argument("-c", "--config", type=str, default="7b", help="Model configuration")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["gemini", "gemini_auto", "fsdp", "fsdp_cpu", "3d", "3d_cpu", "zero", "zero_cpu"],
        default="gemini",
        help="Choose which plugin to use",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-s", "--num_steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("-i", "--ignore_steps", type=int, default=2, help="Number of steps to ignore")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument(
        "-w", "--warmup_ratio", type=float, default=0.8, help="warm up ratio of non-model data. Only for gemini-auto"
    )
    parser.add_argument("-m", "--memory_limit", type=int, help="Gemini memory limit in mb")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use xformers")
    parser.add_argument("--shard_param_frac", type=float, default=1.0, help="Shard param fraction. Only for gemini")
    parser.add_argument("--offload_optim_frac", type=float, default=0.0, help="Offload optim fraction. Only for gemini")
    parser.add_argument("--offload_param_frac", type=float, default=0.0, help="Offload param fraction. Only for gemini")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument("--zero", type=int, default=0)
    args = parser.parse_args()

    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    def empty_init():
        pass

    # ==============================
    # Initialize Booster
    # ==============================
    use_empty_init = True
    if args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision="bf16",
            shard_param_frac=args.shard_param_frac,
            offload_optim_frac=args.offload_optim_frac,
            offload_param_frac=args.offload_param_frac,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(placement_policy="auto", precision="bf16", warmup_non_model_data_ratio=args.warmup_ratio)
    elif args.plugin == "fsdp":
        if use_empty_init:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
                ),
                param_init_fn=empty_init(),
            )
        else:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
                )
            )
    elif args.plugin == "fsdp_cpu":
        if use_empty_init:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
                ),
                cpu_offload=CPUOffload(offload_params=True),
                param_init_fn=empty_init(),
            )
        else:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
                ),
                cpu_offload=CPUOffload(offload_params=True),
            )
    elif args.plugin == "zero":
        plugin = LowLevelZeroPlugin(
            stage=args.zero, precision="bf16", initial_scale=2**16, cpu_offload=False,
        )
    elif args.plugin == "zero_cpu":
        plugin = LowLevelZeroPlugin(
            stage=args.zero, precision="bf16", initial_scale=2**16, cpu_offload=True,
        )
    elif args.plugin == "3d":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            zero_stage=args.zero,
            enable_fused_normalization=True,
            num_microbatches=args.mbs,
            precision="bf16",
        )
    elif args.plugin == "3d_cpu":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            zero_stage=args.zero,
            cpu_offload=True,
            enable_fused_normalization=True,
            num_microbatches=args.mbs,
            initial_scale=2**8,
            precision="bf16",
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Dataset and Dataloader
    # ==============================
    dp_size = plugin.dp_size if isinstance(plugin, HybridParallelPlugin) else coordinator.world_size

    config = get_config(args.model_name, args.config, use_cache=False, vocab_size=32000)
    model_class = MODEL_CLASS[args.model_name]
    if args.model_name == 'retnet':
        model_kwargs = dict(tensor_parallel=args.tp > 1)
    else:
        model_kwargs = {}

    dataset = RandomDataset(
        num_samples=args.batch_size * args.num_steps * dp_size, max_length=args.max_length, vocab_size=config.vocab_size,
        max_steps=args.batch_size * args.num_steps * dp_size,
    )
    dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    init_ctx = (
        LazyInitContext(default_device=get_current_device())
        if isinstance(plugin, (GeminiPlugin, HybridParallelPlugin))
        else nullcontext()
    )

    with init_ctx:
        model = model_class(config, **model_kwargs)

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    if args.xformers and not 'retnet' in args.model_name:
        assert SUPPORT_FLASH, "Use flash attention while xfomers is not installed"
        replace_xformers(model)

    model_numel = get_model_numel(model)
    coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")
    performance_evaluator = PerformanceEvaluator(
        model_numel, args.grad_checkpoint, args.ignore_steps, dp_world_size=dp_size
    )

    optimizer = HybridAdam(model.parameters())
    torch.set_default_dtype(torch.bfloat16)

    if args.model_name == 'retnet':
        pre_retention_rel_pos = model.model.retnet_rel_pos(
            args.max_length,
            forward_impl='parallel',
            get_decay_scale=False,
        )
        retention_rel_pos_device = []
        for rel_pos_group in pre_retention_rel_pos:
            group = []
            for rel_pos in rel_pos_group:
                if rel_pos is not None:
                    rel_pos = rel_pos.cuda()
                    if isinstance(rel_pos, LazyTensor):
                        rel_pos.materialize()
                    rel_pos = rel_pos.to(torch.bfloat16)
                group.append(rel_pos)
            retention_rel_pos_device.append(tuple(group))
        pre_retention_rel_pos = tuple(retention_rel_pos_device)
        if isinstance(plugin, HybridParallelPlugin) and args.pp > 1:
            forward_kwargs = dict(pre_retention_rel_pos=pre_retention_rel_pos)
        else:
            forward_kwargs = dict(retention_rel_pos=pre_retention_rel_pos)
    else:
        forward_kwargs = {}


    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)
    torch.set_default_dtype(torch.float)
    coordinator.print_on_master(f"Booster init max CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )

    if isinstance(plugin, HybridParallelPlugin) and args.pp > 1:
        data_iter = iter(dataloader)
        for step in tqdm(range(len(dataloader)), desc="Step", disable=not coordinator.is_master()):
            performance_evaluator.on_step_start(step)
            booster.execute_pipeline(
                data_iter, model, criterion=lambda outputs, inputs: outputs[0], optimizer=optimizer, return_loss=False,
                **forward_kwargs,
            )
            optimizer.step()
            optimizer.zero_grad()
            performance_evaluator.on_step_end(input_ids=torch.empty(args.batch_size, args.max_length))
    else:
        for step, batch in enumerate(tqdm(dataloader, desc="Step", disable=not coordinator.is_master())):
            performance_evaluator.on_step_start(step)
            outputs = model(**batch, **forward_kwargs)
            loss = outputs[0]
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
            performance_evaluator.on_step_end(**batch)

    performance_evaluator.on_fit_end()
    coordinator.print_on_master(f"Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    main()
