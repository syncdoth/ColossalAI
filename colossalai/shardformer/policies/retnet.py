import warnings
from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from colossalai.shardformer.layer import FusedRMSNorm, FusedLayerNorm, Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D

from ..modeling.retnet import RetNetPipelineForwards
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["RetNetPolicy", "RetNetForCausalLMPolicy", "RetNetForSequenceClassificationPolicy"]


class RetNetPolicy(Policy):

    def config_sanity_check(self):
        pass

    def preprocess(self):
        if self.shard_config.enable_tensor_parallelism:
            # Resize embedding
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size

            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)

        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from retnet.retnet.modeling_retnet import MultiScaleRetention, RetNetDecoderLayer, RetNetModel, FeedForwardNetwork

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "RetNet dosen't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        if self.shard_config.enable_tensor_parallelism:
            # TODO: check more over here
            decoder_attribute_replacement = {
                "retention.value_dim":
                    self.model.config.decoder_value_embed_dim // self.shard_config.tensor_parallel_size,
                "retention.num_heads":
                    self.model.config.decoder_retention_heads //
                    self.shard_config.tensor_parallel_size,
                # for subln
                "ffn.ffn_layernorm.normalized_shape": self.model.config.decoder_ffn_embed_dim // self.shard_config.tensor_parallel_size,
                "ffn.ffn_layernorm.elementwise_affine": False,  # TODO: this should be true.
            }

            sub_module_replacement = [
                SubModuleReplacementDescription(
                    suffix="retention.q_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="retention.k_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="retention.v_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="retention.g_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="retention.out_proj",
                    target_module=Linear1D_Row,
                ),
            ]

            if self.model.config.use_glu:
                sub_module_replacement.append(
                    SubModuleReplacementDescription(
                        suffix="ffn.gate",
                        target_module=Linear1D_Col,
                    ),
                )
            sub_module_replacement.extend([
                SubModuleReplacementDescription(
                    suffix="ffn.fc1",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="ffn.fc2",
                    target_module=Linear1D_Row,
                ),
            ])

            policy[RetNetDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=sub_module_replacement,
            )

            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="embed_tokens",
                        target_module=VocabParallelEmbedding1D,
                    ),
                    SubModuleReplacementDescription(
                        suffix="retnet_rel_pos.proj",
                        target_module=Linear1D_Col,
                    ),
                ],
                policy=policy,
                target_key=RetNetModel,
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="group_norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key=MultiScaleRetention,
            )

            if self.model.config.subln and not self.model.config.use_glu:
                self.append_or_create_submodule_replacement(
                    description=SubModuleReplacementDescription(
                        suffix="ffn_layernorm",
                        target_module=FusedLayerNorm,
                    ),
                    policy=policy,
                    target_key=FeedForwardNetwork,
                )

            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="retention_layer_norm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="final_layer_norm",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=RetNetDecoderLayer,
            )

            if self.model.config.layernorm_embedding:
                self.append_or_create_submodule_replacement(
                    description=SubModuleReplacementDescription(
                        suffix="layernorm_embedding",
                        target_module=FusedRMSNorm,
                    ),
                    policy=policy,
                    target_key=RetNetModel,
                )
            if self.model.config.decoder_normalize_before:
                self.append_or_create_submodule_replacement(
                    description=SubModuleReplacementDescription(
                        suffix="layer_norm",
                        target_module=FusedRMSNorm,
                    ),
                    policy=policy,
                    target_key=RetNetModel,
                )

        if self.shard_config.enable_flash_attention:
            warnings.warn("RetNet doesn't have attention! Can't apply flash attention.")

        return policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable,
                             policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "RetNetModel":
                module = self.model
            else:
                module = self.model.model

            layers_per_stage = Policy.distribute_layers(len(module.layers),
                                                        stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {
                "forward":
                    partial(new_forward, stage_manager=stage_manager, stage_index=stage_index)
            }
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=model_cls)

        return

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "RetNetModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.layers), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.retnet_rel_pos)
            held_layers.append(module.embed_tokens)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            if module.layer_norm is not None:
                held_layers.append(module.layer_norm)

        return held_layers


class RetNetModelPolicy(RetNetPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        from retnet.retnet.modeling_retnet import RetNetModel

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(model_cls=RetNetModel,
                                      new_forward=RetNetPipelineForwards.retnet_model_forward,
                                      policy=policy)
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in retnet model"""
        return []


class RetNetForCausalLMPolicy(RetNetPolicy):

    def module_policy(self):
        from retnet.retnet.modeling_retnet import RetNetForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                RetNetForCausalLM:
                    ModulePolicyDescription(sub_module_replacement=[
                        SubModuleReplacementDescription(suffix="lm_head",
                                                        target_module=Linear1D_Col,
                                                        kwargs=dict(gather_output=True))
                    ])
            }
            policy.update(new_item)

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=RetNetForCausalLM,
                new_forward=RetNetPipelineForwards.retnet_for_causal_lm_forward,
                policy=policy)

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        retnet_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if (id(retnet_model.embed_tokens.weight) == id(self.model.lm_head.weight) and
                    self.pipeline_stage_manager.num_stages > 1):
                # tie weights
                return [{
                    0: retnet_model.embed_tokens.weight,
                    self.pipeline_stage_manager.num_stages - 1: self.model.lm_head.weight,
                }]
        return []


class RetNetForSequenceClassificationPolicy(RetNetPolicy):
    def module_policy(self):
        from retnet.retnet.modeling_retnet import RetNetForSequenceClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for sequence classification
            new_item = {
                RetNetForSequenceClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="score", target_module=Linear1D_Col, kwargs=dict(gather_output=True)
                        )
                    ]
                )
            }
            policy.update(new_item)
        # to be confirmed
        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=RetNetForSequenceClassification,
                new_forward=RetNetPipelineForwards.retnet_for_sequence_classification_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in llama for sequence classification model"""
        return []
