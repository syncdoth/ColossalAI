from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
from retnet import RetNetCausalLMOutputWithPast, RetNetModel, RetNetOutputWithPast, RetNetForCausalLM
from torch.nn import CrossEntropyLoss
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager


class RetNetPipelineForwards:
    """
    This class serves as a micro library for forward function substitution of RetNet models
    under pipeline setting.
    """

    @staticmethod
    def retnet_model_forward(
        self: RetNetModel,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        retention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        forward_impl: Optional[str] = "parallel",
        recurrent_chunk_size: Optional[int] = None,
        past_key_values: Optional[List[Dict[str, torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ):
        del position_ids  # RetNet does not use position_ids
        logger = logging.get_logger(__name__)

        output_retentions = output_attentions if output_attentions is not None else self.config.output_retentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if stage_manager.is_first_stage():
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if inputs_embeds is None:
                inputs_embeds = self.forward_embedding(input_ids, inputs_embeds, past_key_values)
        else:
            input_shape = hidden_states.shape[:-1]
            batch_size, seq_length = input_shape
            device = hidden_states.device

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask

        hidden_states = inputs_embeds

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_retentions:
            logger.warning_once(
                "output_attentions(retentions)=True is not supported for pipeline models at the moment."
            )
            output_retentions = False
        if output_hidden_states:
            logger.warning_once(
                "output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once(
                "use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        # if past_key_values is not None:
        #     past_key_values_length = past_key_values[0][0].shape[2]
        #     seq_length_with_past = seq_length_with_past + past_key_values_length

        # # embed positions, for the first stage, hidden_states is the input embeddings,
        # # for the other stages, hidden_states is the output of the previous stage
        # if attention_mask is None:
        #     attention_mask = torch.ones((batch_size, seq_length_with_past),
        #                                 dtype=torch.bool,
        #                                 device=hidden_states.device)
        # attention_mask = self._prepare_decoder_attention_mask(attention_mask,
        #                                                       (batch_size, seq_length),
        #                                                       hidden_states, past_key_values_length)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # handling chunking here
        if recurrent_chunk_size is None:
            recurrent_chunk_size = self.recurrent_chunk_size
        need_pad_for_chunkwise = (forward_impl == 'chunkwise' and
                                  seq_length % recurrent_chunk_size != 0)
        if need_pad_for_chunkwise:
            padding_len = recurrent_chunk_size - seq_length % recurrent_chunk_size
            slen = seq_length + padding_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding_len))
        else:
            slen = seq_length
        # relative position
        retention_rel_pos = self.retnet_rel_pos(slen,
                                                forward_impl=forward_impl,
                                                recurrent_chunk_size=recurrent_chunk_size,
                                                retention_mask=retention_mask)

        # start running through the decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        # layers * [bsz, num_head, qk_dim, decoder_embed_dim]
        next_decoder_cache = () if use_cache else None

        start_idx, end_idx = stage_index[0], stage_index[1]
        for idx, decoder_layer in enumerate(self.layers[start_idx:end_idx], start=start_idx):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_retentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    retention_rel_pos,
                    retention_mask,
                    forward_impl,
                    past_key_value,
                )
            else:
                layer_outputs = decoder_layer(hidden_states,
                                              retention_rel_pos,
                                              retention_mask=retention_mask,
                                              forward_impl=forward_impl,
                                              past_key_value=past_key_value,
                                              output_retentions=output_retentions)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
            if output_retentions:
                all_retentions += (layer_outputs[2],)

        next_cache = next_decoder_cache if use_cache else None

        if need_pad_for_chunkwise:
            hidden_states = hidden_states[:, :seq_length, :]

        if stage_manager.is_last_stage() and self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if stage_manager.is_last_stage():
            if not return_dict:
                return tuple(
                    v for v in [hidden_states, next_cache, all_hidden_states, all_retentions]
                    if v is not None)
            return RetNetOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                retentions=all_retentions,
            )
        # always return dict for imediate stage
        return {"hidden_states": hidden_states}

    @staticmethod
    def retnet_for_causal_lm_forward(
        self: RetNetForCausalLM,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        retention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        forward_impl: Optional[str] = "parallel",
        recurrent_chunk_size: Optional[int] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from retnet import RetNetForCausalLM

        >>> model = RetNetForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        del position_ids  # RetNet does not use position_ids
        logger = logging.get_logger(__name__)
        output_retentions = output_attentions if output_attentions is not None else self.config.output_retentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        forward_impl = forward_impl if forward_impl is not None else self.config.forward_impl
        recurrent_chunk_size = recurrent_chunk_size if recurrent_chunk_size is not None else self.config.recurrent_chunk_size

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_retentions:
            logger.warning_once(
                "output_attentions (retentions)=True is not supported for pipeline models at the moment."
            )
            output_retentions = False
        if output_hidden_states:
            logger.warning_once(
                "output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = RetNetPipelineForwards.retnet_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            retention_mask=retention_mask,
            forward_impl=forward_impl,
            recurrent_chunk_size=recurrent_chunk_size,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
        )
        past_key_values = None

        if stage_manager.is_last_stage():
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return RetNetCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                retentions=outputs.retentions,
            )
        else:
            hidden_states = outputs.get("hidden_states")
            return {"hidden_states": hidden_states}
