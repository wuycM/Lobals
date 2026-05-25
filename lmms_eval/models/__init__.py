from __future__ import annotations

from lmms_eval.models.registry_v2 import ModelManifest, ModelRegistryV2


_MANIFESTS = [
    ModelManifest(
        model_id="internvl3",
        simple_class_path="lmms_eval.models.simple.internvl3.InternVL3",
    ),
    ModelManifest(
        model_id="internvl3_5_reforward_visual_attention_after_layer",
        simple_class_path=(
            "lmms_eval.models.simple."
            "internvl3_5_reforward_visual_attention_after_layer."
            "InternVL3_5_ReForwardVisualAttentionAfterLayer"
        ),
        aliases=("internvl3_5_reforward_visual_attention_after_layer",),
    ),
    ModelManifest(
        model_id="qwen3_vl",
        simple_class_path="lmms_eval.models.simple.qwen3_vl.Qwen3_VL",
    ),
    ModelManifest(
        model_id="qwen3_vl_reforward_visual_attention_after_layer",
        simple_class_path=(
            "lmms_eval.models.simple."
            "qwen3_vl_reforward_visual_attention_after_layer."
            "Qwen3_VL_ReForwardVisualAttentionAfterLayer"
        ),
        aliases=(
            "qwen3_vl_reforward_visual_attention_after_layer",
            "qwen3vl_reforward_visual_attention_after_layer",
        ),
    ),
]

AVAILABLE_SIMPLE_MODELS = {
    "internvl3": "InternVL3",
    "internvl3_5_reforward_visual_attention_after_layer": "InternVL3_5_ReForwardVisualAttentionAfterLayer",
    "qwen3_vl": "Qwen3_VL",
    "qwen3_vl_reforward_visual_attention_after_layer": "Qwen3_VL_ReForwardVisualAttentionAfterLayer",
}
AVAILABLE_CHAT_TEMPLATE_MODELS = {}
MODEL_ALIASES = {
    "internvl3_5_reforward_visual_attention_after_layer": ("internvl3_5_reforward_visual_attention_after_layer",),
    "qwen3_vl_reforward_visual_attention_after_layer": (
        "qwen3_vl_reforward_visual_attention_after_layer",
        "qwen3vl_reforward_visual_attention_after_layer",
    ),
}


def _initialize_model_registry() -> ModelRegistryV2:
    registry = ModelRegistryV2()
    for manifest in _MANIFESTS:
        registry.register_manifest(manifest)
    return registry


MODEL_REGISTRY_V2 = _initialize_model_registry()
AVAILABLE_MODELS = {
    manifest.model_id: (manifest.chat_class_path or manifest.simple_class_path).rsplit(".", 1)[-1]
    for manifest in _MANIFESTS
}


def list_available_models(include_aliases: bool = False) -> list[str]:
    if include_aliases:
        return MODEL_REGISTRY_V2.list_model_names()
    return MODEL_REGISTRY_V2.list_canonical_model_ids()


def get_model_manifest(model_name: str) -> ModelManifest:
    return MODEL_REGISTRY_V2.get_manifest(model_name)


def get_model(model_name: str, force_simple: bool = False) -> type:
    return MODEL_REGISTRY_V2.get_model_class(model_name, force_simple=force_simple)
