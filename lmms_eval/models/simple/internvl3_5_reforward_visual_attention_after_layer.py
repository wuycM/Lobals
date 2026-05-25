import json
from pathlib import Path
from types import MethodType
from typing import List

import torch
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3 import DEFAULT_GEN_KWARGS, InternVL3, load_image, load_video


def _compute_visual_attention_ratios(
    attn_weights: torch.Tensor, visual_token_indices_by_sample: list[list[int]]
) -> list[float]:
    ratios = []
    for sample_idx, visual_token_indices in enumerate(visual_token_indices_by_sample):
        if not visual_token_indices:
            ratios.append(0.0)
            continue

        index_tensor = torch.tensor(visual_token_indices, device=attn_weights.device, dtype=torch.long)
        index_tensor = index_tensor[index_tensor < attn_weights.shape[-1]]
        if index_tensor.numel() == 0:
            ratios.append(0.0)
            continue

        per_head_visual_mass = attn_weights[sample_idx, :, -1, index_tensor].sum(dim=-1)
        ratios.append(float(per_head_visual_mass.mean().item()))
    return ratios


def _embed_input_ids(language_model, input_ids: torch.Tensor) -> torch.Tensor:
    embed_tokens = getattr(language_model, "embed_tokens", None)
    if embed_tokens is None and hasattr(language_model, "model"):
        embed_tokens = getattr(language_model.model, "embed_tokens", None)
    if embed_tokens is None and hasattr(language_model, "get_input_embeddings"):
        embed_tokens = language_model.get_input_embeddings()
    if embed_tokens is None:
        raise AttributeError(
            f"{type(language_model).__name__} has no accessible input embedding layer via "
            "`embed_tokens`, `model.embed_tokens`, or `get_input_embeddings()`."
        )
    return embed_tokens(input_ids)


class InternVLAfterLayerWindowReForwardPatcher:
    def __init__(
        self,
        model,
        after_layer: int,
        candidate_start: int,
        candidate_end: int,
        top_k: int,
    ):
        self.model = model
        self.language_model = model.language_model
        self.decoder_model = self.language_model.model if hasattr(self.language_model, "model") else self.language_model
        self.after_layer_idx = after_layer - 1
        self.candidate_start_idx = candidate_start - 1
        self.candidate_end_idx = candidate_end - 1
        self.window_size = top_k

        self.original_model_forward = model.forward
        self.original_model_generate = model.generate
        self.original_forward = self.decoder_model.forward

        self.visual_token_indices_by_sample: list[list[int]] = []
        self.cached_visual_token_indices_by_sample: list[list[int]] = []
        self.last_selected_layers = []
        self.last_selected_window_means = []
        self.selected_window_history_by_sample = []
        self._selection_history_step_idx = 0

    def _reset_selected_window_history(self, batch_size: int):
        self.selected_window_history_by_sample = [[] for _ in range(batch_size)]
        self._selection_history_step_idx = 0

    def _record_selected_windows(
        self,
        selected_layer_indices_by_sample: list[list[int]],
        selected_window_means: list[float],
        position_ids: torch.Tensor,
    ):
        batch_size = len(selected_layer_indices_by_sample)
        if len(self.selected_window_history_by_sample) != batch_size:
            self._reset_selected_window_history(batch_size)

        if position_ids.ndim == 1:
            per_sample_positions = [int(position_ids[-1].item())]
        else:
            per_sample_positions = [int(sample_positions[-1].item()) for sample_positions in position_ids]

        for sample_idx, selected_layer_indices in enumerate(selected_layer_indices_by_sample):
            layers = [layer_idx + 1 for layer_idx in selected_layer_indices]
            window = [layers[0], layers[-1]] if layers else []
            #print(window)
            cache_position_value = (
                per_sample_positions[sample_idx] if sample_idx < len(per_sample_positions) else per_sample_positions[-1]
            )
            self.selected_window_history_by_sample[sample_idx].append(
                {
                    "step": self._selection_history_step_idx,
                    "cache_position": cache_position_value,
                    "window": window,
                    "layers": layers,
                    "window_mean": float(selected_window_means[sample_idx]),
                }
            )

        self._selection_history_step_idx += 1

    def _capture_visual_token_indices(
        self,
        input_ids: torch.Tensor | None,
        batch_size: int,
        is_decode_step: bool,
    ):
        if input_ids is not None and self.model.img_context_token_id is not None:
            special_image_mask = input_ids == self.model.img_context_token_id
            if special_image_mask.any():
                visual_token_indices_by_sample = [
                    torch.nonzero(sample_mask, as_tuple=False).flatten().tolist() for sample_mask in special_image_mask
                ]
                self.visual_token_indices_by_sample = [indices.copy() for indices in visual_token_indices_by_sample]
                self.cached_visual_token_indices_by_sample = [indices.copy() for indices in visual_token_indices_by_sample]
                return

        if input_ids is None and len(self.cached_visual_token_indices_by_sample) == batch_size:
            self.visual_token_indices_by_sample = [indices.copy() for indices in self.cached_visual_token_indices_by_sample]
            return

        if is_decode_step and len(self.cached_visual_token_indices_by_sample) == batch_size:
            self.visual_token_indices_by_sample = [indices.copy() for indices in self.cached_visual_token_indices_by_sample]
            return

        self.visual_token_indices_by_sample = [[] for _ in range(batch_size)]
        if not is_decode_step:
            self.cached_visual_token_indices_by_sample = []

    def _is_decode_step(self, inputs_embeds: torch.Tensor, past_key_values) -> bool:
        if past_key_values is None:
            return False
        return past_key_values.get_seq_length() > 0 and inputs_embeds.shape[1] == 1

    def _should_split_prefill_for_first_token(self, inputs_embeds: torch.Tensor, past_key_values, use_cache) -> bool:
        if not use_cache or inputs_embeds.shape[1] <= 1:
            return False
        if past_key_values is None:
            return True
        return past_key_values.get_seq_length() == 0

    def _clone_cache(self, past_key_values, sample_idx: int | None = None) -> DynamicCache:
        temp_cache = DynamicCache(config=self.decoder_model.config)
        if past_key_values is None:
            return temp_cache

        for layer_idx, layer_cache in enumerate(past_key_values.layers):
            if layer_cache is None or not getattr(layer_cache, "is_initialized", False):
                continue
            if layer_cache.get_seq_length() == 0:
                continue

            keys = layer_cache.keys if sample_idx is None else layer_cache.keys[sample_idx : sample_idx + 1]
            values = layer_cache.values if sample_idx is None else layer_cache.values[sample_idx : sample_idx + 1]
            temp_cache.update(keys.clone(), values.clone(), layer_idx)
        return temp_cache

    def _build_history_only_cache(
        self,
        past_key_values,
        layer_idx: int,
        sample_idx: int,
        cache_layer_idx: int | None = None,
    ) -> DynamicCache:
        temp_cache = DynamicCache(config=self.decoder_model.config)
        if past_key_values is None or len(past_key_values.layers) <= layer_idx:
            return temp_cache

        layer_cache = past_key_values.layers[layer_idx]
        if layer_cache is None or not getattr(layer_cache, "is_initialized", False):
            return temp_cache
        if layer_cache.get_seq_length() <= 1:
            return temp_cache

        history_keys = layer_cache.keys[sample_idx : sample_idx + 1, :, :-1, :].clone()
        history_values = layer_cache.values[sample_idx : sample_idx + 1, :, :-1, :].clone()
        if history_keys.shape[-2] == 0:
            return temp_cache
        temp_cache.update(
            history_keys,
            history_values,
            layer_idx if cache_layer_idx is None else cache_layer_idx,
        )
        return temp_cache

    def _slice_sample_attention_mask(self, attention_mask: torch.Tensor | None, sample_idx: int):
        if attention_mask is None:
            return None
        if attention_mask.ndim in (2, 4):
            return attention_mask[sample_idx : sample_idx + 1]
        raise ValueError("InternVL reforward only supports 2D or 4D attention masks.")

    def _slice_sequence_tensor(self, tensor, end: int):
        if tensor is None:
            return None
        return tensor[..., :end]

    def _forward_layer_with_attention(
        self,
        layer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values,
        cache_position: torch.Tensor | None,
        position_embeddings,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, attn_weights = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights

    def _forward_layer_with_frozen_memory(
        self,
        layer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
        frozen_past_key_values,
        position_embeddings,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        attn_module = layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)

        query_states = attn_module.q_norm(attn_module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        probe_key_states = attn_module.k_norm(attn_module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, probe_key_states, cos, sin)

        if frozen_past_key_values is None or len(frozen_past_key_values.layers) <= layer_idx:
            raise ValueError(f"Missing frozen KV cache for layer {layer_idx}.")
        frozen_layer_cache = frozen_past_key_values.layers[layer_idx]
        if frozen_layer_cache is None or not getattr(frozen_layer_cache, "is_initialized", False):
            raise ValueError(f"Frozen KV cache for layer {layer_idx} is not initialized.")

        attn_output, _ = eager_attention_forward(
            attn_module,
            query_states,
            frozen_layer_cache.keys,
            frozen_layer_cache.values,
            attention_mask,
            dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
            scaling=attn_module.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_module.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _select_top_k_layers(self, candidate_ratios: list[tuple[int, float]]) -> list[int]:
        if not candidate_ratios:
            return []

        window_size = min(self.window_size, len(candidate_ratios))
        window_sum = sum(ratio for _layer_idx, ratio in candidate_ratios[:window_size])
        best_sum = window_sum
        best_start = 0

        for start in range(1, len(candidate_ratios) - window_size + 1):
            window_sum += candidate_ratios[start + window_size - 1][1] - candidate_ratios[start - 1][1]
            if window_sum > best_sum:
                best_sum = window_sum
                best_start = start

        return [layer_idx for layer_idx, _ratio in candidate_ratios[best_start : best_start + window_size]]

    def _compute_window_mean(
        self,
        candidate_ratios: list[tuple[int, float]],
        selected_layer_indices: list[int],
    ) -> float:
        if not selected_layer_indices:
            return 0.0
        ratio_by_layer = dict(candidate_ratios)
        return sum(ratio_by_layer[layer_idx] for layer_idx in selected_layer_indices) / len(selected_layer_indices)

    def _run_probe_decode(
        self,
        language_model_self,
        inputs_embeds: torch.Tensor,
        input_attention_mask,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        kwargs: dict,
    ):
        probe_cache = self._clone_cache(past_key_values)
        attention_mask = create_causal_mask(
            config=language_model_self.config,
            input_embeds=inputs_embeds,
            attention_mask=input_attention_mask,
            cache_position=cache_position,
            past_key_values=probe_cache,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = language_model_self.rotary_emb(hidden_states, position_ids=position_ids)
        batch_size = hidden_states.shape[0]
        candidate_ratios_by_sample = [[] for _ in range(batch_size)]
        after_layer_hidden_states = None

        for layer_idx, decoder_layer in enumerate(language_model_self.layers):
            hidden_states, attn_weights = self._forward_layer_with_attention(
                decoder_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=probe_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if self.candidate_start_idx <= layer_idx <= self.candidate_end_idx:
                candidate_layer_ratios = _compute_visual_attention_ratios(attn_weights, self.visual_token_indices_by_sample)
                for sample_idx, ratio in enumerate(candidate_layer_ratios):
                    candidate_ratios_by_sample[sample_idx].append((layer_idx, ratio))

            if layer_idx == self.after_layer_idx:
                after_layer_hidden_states = hidden_states.clone()

            del attn_weights

        hidden_states = language_model_self.norm(hidden_states)
        if after_layer_hidden_states is None:
            raise ValueError(f"Failed to capture hidden states for after_layer index {self.after_layer_idx}.")
        return candidate_ratios_by_sample, after_layer_hidden_states, hidden_states, probe_cache

    def _run_after_layer_window_decode(
        self,
        language_model_self,
        input_attention_mask,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        frozen_past_key_values,
        sample_idx: int,
        after_layer_hidden_states: torch.Tensor,
        selected_layer_indices: list[int],
        kwargs: dict,
    ) -> torch.Tensor:
        full_attention_mask = self._slice_sample_attention_mask(input_attention_mask, sample_idx)
        hidden_states = after_layer_hidden_states.clone()
        position_embeddings = language_model_self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer_idx in selected_layer_indices:
            decoder_layer = language_model_self.layers[layer_idx]
            history_only_cache = self._build_history_only_cache(
                frozen_past_key_values,
                layer_idx=layer_idx,
                sample_idx=sample_idx,
                cache_layer_idx=0,
            )
            layer_attention_mask = create_causal_mask(
                config=language_model_self.config,
                input_embeds=hidden_states,
                attention_mask=full_attention_mask,
                cache_position=cache_position,
                past_key_values=history_only_cache,
                position_ids=position_ids,
            )
            decoder_layer_original_idx = decoder_layer.self_attn.layer_idx
            decoder_layer.self_attn.layer_idx = 0
            try:
                hidden_states, _ = self._forward_layer_with_attention(
                    decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    past_key_values=history_only_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            finally:
                decoder_layer.self_attn.layer_idx = decoder_layer_original_idx

        for layer_idx in range(self.after_layer_idx + 1, len(language_model_self.layers)):
            decoder_layer = language_model_self.layers[layer_idx]
            history_only_cache = self._build_history_only_cache(
                frozen_past_key_values,
                layer_idx=layer_idx,
                sample_idx=sample_idx,
                cache_layer_idx=0,
            )
            layer_attention_mask = create_causal_mask(
                config=language_model_self.config,
                input_embeds=hidden_states,
                attention_mask=full_attention_mask,
                cache_position=cache_position,
                past_key_values=history_only_cache,
                position_ids=position_ids,
            )
            original_layer_idx = decoder_layer.self_attn.layer_idx
            decoder_layer.self_attn.layer_idx = 0
            try:
                hidden_states, _ = self._forward_layer_with_attention(
                    decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    past_key_values=history_only_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            finally:
                decoder_layer.self_attn.layer_idx = original_layer_idx

        hidden_states = language_model_self.norm(hidden_states)
        return hidden_states

    def _run_reforward_decode(
        self,
        language_model_self,
        inputs_embeds: torch.Tensor,
        input_attention_mask,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        kwargs: dict,
    ):
        (
            candidate_ratios_by_sample,
            after_layer_hidden_states,
            probe_final_hidden_states,
            probe_cache,
        ) = self._run_probe_decode(
            language_model_self=language_model_self,
            inputs_embeds=inputs_embeds,
            input_attention_mask=input_attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            kwargs=kwargs,
        )

        batch_size = inputs_embeds.shape[0]
        selected_layer_indices_by_sample = []
        selected_window_means = []
        for sample_idx, candidate_ratios in enumerate(candidate_ratios_by_sample):
            if not self.visual_token_indices_by_sample[sample_idx]:
                selected_layer_indices_by_sample.append([])
                selected_window_means.append(0.0)
                continue
            selected_layer_indices = self._select_top_k_layers(candidate_ratios)
            selected_layer_indices_by_sample.append(selected_layer_indices)
            selected_window_means.append(self._compute_window_mean(candidate_ratios, selected_layer_indices))

        if batch_size == 1:
            self.last_selected_layers = [layer_idx + 1 for layer_idx in selected_layer_indices_by_sample[0]]
            self.last_selected_window_means = [] if not self.last_selected_layers else selected_window_means[:1]
        else:
            self.last_selected_layers = [
                [selected_layer_idx + 1 for selected_layer_idx in selected_layer_indices]
                for selected_layer_indices in selected_layer_indices_by_sample
            ]
            self.last_selected_window_means = selected_window_means

        self._record_selected_windows(selected_layer_indices_by_sample, selected_window_means, position_ids)

        updated_hidden_states = []
        for sample_idx, selected_layer_indices in enumerate(selected_layer_indices_by_sample):
            if not selected_layer_indices:
                updated_hidden_states.append(probe_final_hidden_states[sample_idx : sample_idx + 1])
                continue

            sample_hidden_states = self._run_after_layer_window_decode(
                language_model_self=language_model_self,
                input_attention_mask=input_attention_mask,
                position_ids=position_ids[sample_idx : sample_idx + 1],
                cache_position=cache_position,
                frozen_past_key_values=self._clone_cache(probe_cache, sample_idx=sample_idx),
                sample_idx=sample_idx,
                after_layer_hidden_states=after_layer_hidden_states[sample_idx : sample_idx + 1],
                selected_layer_indices=selected_layer_indices,
                kwargs=kwargs,
            )
            updated_hidden_states.append(sample_hidden_states)

        hidden_states = torch.cat(updated_hidden_states, dim=0)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=probe_cache)

    def apply(self):
        def patched_model_forward(
            model_self,
            pixel_values,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            image_flags=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            batch_size = 0 if input_ids is None else input_ids.shape[0]
            self._capture_visual_token_indices(input_ids, batch_size=batch_size, is_decode_step=False)
            return self.original_model_forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        def patched_model_generate(
            model_self,
            pixel_values=None,
            input_ids=None,
            attention_mask=None,
            visual_features=None,
            generation_config=None,
            output_hidden_states=None,
            **generate_kwargs,
        ):
            batch_size = 0 if input_ids is None else input_ids.shape[0]
            self._capture_visual_token_indices(input_ids, batch_size=batch_size, is_decode_step=False)
            return self.original_model_generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                **generate_kwargs,
            )

        def patched_forward(
            decoder_model_self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            **kwargs,
        ):
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = _embed_input_ids(decoder_model_self, input_ids)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=decoder_model_self.config)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

            is_decode_step = self._is_decode_step(inputs_embeds, past_key_values)
            self._capture_visual_token_indices(input_ids, batch_size=inputs_embeds.shape[0], is_decode_step=is_decode_step)

            if self._should_split_prefill_for_first_token(inputs_embeds, past_key_values, use_cache):
                self._reset_selected_window_history(inputs_embeds.shape[0])
                if not any(self.visual_token_indices_by_sample):
                    self.last_selected_layers = []
                    self.last_selected_window_means = []
                    return self.original_forward(
                        input_ids=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )

                prefix_seq_len = inputs_embeds.shape[1] - 1
                prefix_outputs = self.original_forward(
                    input_ids=None,
                    attention_mask=self._slice_sequence_tensor(attention_mask, prefix_seq_len),
                    position_ids=self._slice_sequence_tensor(position_ids, prefix_seq_len),
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds[:, :-1, :],
                    use_cache=use_cache,
                    cache_position=cache_position[:-1],
                    **kwargs,
                )

                last_outputs = self._run_reforward_decode(
                    language_model_self=decoder_model_self,
                    inputs_embeds=inputs_embeds[:, -1:, :],
                    input_attention_mask=attention_mask,
                    position_ids=position_ids[:, -1:],
                    cache_position=cache_position[-1:],
                    past_key_values=past_key_values,
                    kwargs=kwargs,
                )
                hidden_states = torch.cat([prefix_outputs.last_hidden_state, last_outputs.last_hidden_state], dim=1)
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=last_outputs.past_key_values,
                )

            if not is_decode_step:
                self._reset_selected_window_history(inputs_embeds.shape[0])
                self.last_selected_layers = []
                self.last_selected_window_means = []
                return self.original_forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            if not any(self.visual_token_indices_by_sample):
                self.last_selected_layers = []
                self.last_selected_window_means = []
                return self.original_forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            return self._run_reforward_decode(
                language_model_self=decoder_model_self,
                inputs_embeds=inputs_embeds,
                input_attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                kwargs=kwargs,
            )

        self.model.forward = MethodType(patched_model_forward, self.model)
        self.model.generate = MethodType(patched_model_generate, self.model)
        self.decoder_model.forward = MethodType(patched_forward, self.decoder_model)


@register_model("internvl3_5_reforward_visual_attention_after_layer")
class InternVL3_5_ReForwardVisualAttentionAfterLayer(InternVL3):
    @property
    def selected_window_history_by_sample(self):
        return self._reforward_visual_attention_patcher.selected_window_history_by_sample

    @property
    def last_selected_layers(self):
        return self._reforward_visual_attention_patcher.last_selected_layers

    @property
    def last_selected_window_means(self):
        return self._reforward_visual_attention_patcher.last_selected_window_means

    def _resolve_window_history_save_path(self) -> Path | None:
        if not self.save_window_history_path:
            return None

        path = Path(self.save_window_history_path)
        if path.exists() and path.is_dir():
            path = path / "window_history.json"

        if self.world_size > 1:
            if path.suffix:
                path = path.with_name(f"{path.stem}.rank{self.rank}{path.suffix}")
            else:
                path = path.with_name(f"{path.name}.rank{self.rank}")

        return path

    def _dump_window_history(self, window_history_records: list[dict]):
        save_path = self._resolve_window_history_save_path()
        if save_path is None:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(window_history_records, f, ensure_ascii=False, indent=2)

    def _build_window_history_record(
        self,
        request: Instance,
        sample_history: list[dict],
    ) -> dict:
        return {
            "request_idx": int(request.idx),
            "task": str(request.task_name),
            "doc_id": str(request.doc_id),
            "split": request.metadata.get("split"),
            "window_history": [
                {
                    "step": int(step_record["step"]),
                    "cache_position": int(step_record["cache_position"]),
                    "window": list(step_record["window"]),
                    "layers": list(step_record["layers"]),
                    "window_mean": float(step_record["window_mean"]),
                }
                for step_record in sample_history
            ],
        }

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        window_history_records: list[dict] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k in gen_kwargs:
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)
            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if self.modality == "image":
                if visuals:
                    image_num = len(visuals)
                    dynamic_max_num = max(1, min(self.max_num, self.total_max_num // image_num))

                    processed_visuals = [load_image(visual, max_num=dynamic_max_num).to(torch.bfloat16).to(self._device) for visual in visuals]
                    pixel_values = torch.cat(processed_visuals, dim=0)
                    num_patches_list = [v.size(0) for v in processed_visuals]
                    existing_tags = contexts.count("<image>")

                    if existing_tags == 0:
                        image_tokens = " ".join(["<image>"] * len(processed_visuals))
                        contexts = image_tokens + "\n" + contexts
                    elif existing_tags != len(processed_visuals):
                        eval_logger.warning(
                            f"[InternVL3.5-ReForward-AfterLayer] Token mismatch: text has {existing_tags} tags but {len(processed_visuals)} images provided."
                        )
                        image_tokens = " ".join(["<image>"] * len(processed_visuals))
                        contexts = image_tokens + "\n" + contexts
                else:
                    pixel_values = None
                    num_patches_list = None

                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    contexts,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]
                dynamic_max_num = max(1, min(self.max_num, self.total_max_num // self.num_frame))
                pixel_values, num_patches_list = load_video(video_path, num_segments=self.num_frame, max_num=dynamic_max_num)
                pixel_values = pixel_values.to(torch.bfloat16).to(self._device)
                video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
                question = video_prefix + contexts

                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

            sample_history = self.selected_window_history_by_sample
            if len(sample_history) != 1:
                raise ValueError(
                    "internvl3_5_reforward_visual_attention_after_layer expects batch size 1 and exactly one history record, "
                    f"got {len(sample_history)}."
                )
            window_history_records.append(self._build_window_history_record(request, sample_history[0]))

            res.append(response)
            pbar.update(1)

        self._dump_window_history(window_history_records)
        pbar.close()
        return res

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B",
        after_layer: int = 35,
        candidate_start: int = 8,
        candidate_end: int = 35,
        top_k: int = 9,
        window_size: int | None = None,
        **kwargs,
    ):
        self.save_window_history_path = kwargs.pop("save_window_history_path", None)
        kwargs["pretrained"] = pretrained
        kwargs["use_flash_attn"] = False
        super().__init__(**kwargs)

        self.after_layer = int(after_layer)
        self.candidate_start = int(candidate_start)
        self.candidate_end = int(candidate_end)
        self.window_size = int(top_k if window_size is None else window_size)
        self.top_k = self.window_size

        total_layers = len(self.model.language_model.model.layers) if hasattr(self.model.language_model, "model") else len(self.model.language_model.layers)
        if self.after_layer < 1 or self.after_layer > total_layers:
            raise ValueError(f"after_layer must satisfy 1 <= after_layer <= {total_layers}, got {self.after_layer}.")
        if self.candidate_start < 1 or self.candidate_end > self.after_layer or self.candidate_start > self.candidate_end:
            raise ValueError(
                f"candidate layers must satisfy 1 <= start <= end <= after_layer={self.after_layer}, "
                f"got [{self.candidate_start}, {self.candidate_end}]."
            )
        candidate_span = self.candidate_end - self.candidate_start + 1
        if self.top_k < 1 or self.top_k > candidate_span:
            raise ValueError(f"top_k must satisfy 1 <= top_k <= candidate span {candidate_span}, got {self.top_k}.")

        patcher = InternVLAfterLayerWindowReForwardPatcher(
            self.model,
            after_layer=self.after_layer,
            candidate_start=self.candidate_start,
            candidate_end=self.candidate_end,
            top_k=self.window_size,
        )
        patcher.apply()
        self._reforward_visual_attention_patcher = patcher
