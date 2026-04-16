import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


MODEL_ATTRS = {
    "GlmMoeDsaQModel": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "GlmMoeDsaForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Qwen3-Coder-30B-A3B-Instruct": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "NonUniformQwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Llama4ForCausalLM": {
        "moe_block": "feed_forward",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "MixtralForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w3",
        "up_proj": "w1",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "DeepseekV2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoEForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "moe_k",
    },
    "gpt-oss-20b": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Glm4MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
}


def get_transformer_layers_root(model):
    """
    Return the module that owns `.layers` for both native HF models and wrappers
    such as GPTQModel QModel.

    Examples:
    - native GLM-5: model.model.layers
    - GPTQModel AWQ wrapper: model.model.model.layers
    """
    cur = model
    seen = set()

    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))

        if hasattr(cur, "layers"):
            return cur

        if hasattr(cur, "model"):
            cur = cur.model
            continue

        break

    raise AttributeError(
        f"Could not find transformer layers root for {model.__class__.__name__}"
    )


def get_transformer_layers_root(model):
    """
    Find the module that owns `.layers` for both native HF models and wrappers
    such as GPTQModel QModel.

    Examples:
    - native GLM-5: model.model.layers
    - GPTQModel AWQ wrapper: model.model.model.layers
    """
    cur = model
    seen = set()

    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))

        if hasattr(cur, "layers"):
            return cur

        if hasattr(cur, "model"):
            cur = cur.model
            continue

        break

    raise AttributeError(
        f"Could not find transformer layers root for {model.__class__.__name__}"
    )


def get_moe(model, layer):
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    if model_attrs is None:
        raise KeyError(
            f"MODEL_ATTRS does not contain {model.__class__.__name__}. "
            f"Known: {list(MODEL_ATTRS.keys())}"
        )

    moe_attr_name = model_attrs["moe_block"]
    layers_root = get_transformer_layers_root(model)
    return getattr(layers_root.layers[layer], moe_attr_name)


def _set_tensor_by_name(module: nn.Module, name: str, tensor: torch.Tensor, is_buffer: bool):
    parts = name.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    leaf = parts[-1]
    if is_buffer:
        parent._buffers[leaf] = tensor
    else:
        parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)


def _prune_modulelists_recursively(module: nn.Module, keep_list: list[int], old_num_experts: int) -> bool:
    """
    Recursively prune any ModuleList whose length equals the expert count.
    This is the key path for AWQ/GPTQModel layouts where experts are exploded
    into per-expert modules instead of packed tensors.
    """
    changed = False

    for name, child in list(module.named_children()):
        if isinstance(child, nn.ModuleList) and len(child) == old_num_experts:
            new_child = nn.ModuleList([child[i] for i in keep_list])
            setattr(module, name, new_child)
            changed = True
        else:
            changed = _prune_modulelists_recursively(child, keep_list, old_num_experts) or changed

    return changed


def _prune_packed_expert_tensors(module: nn.Module, keep: torch.Tensor, old_num_experts: int) -> bool:
    """
    Prune packed expert params/buffers whose first dimension is expert index.
    This is the native/FP8 path.
    """
    changed = False

    for name, param in list(module.named_parameters(recurse=True)):
        if param is None:
            continue
        if param.ndim > 0 and param.shape[0] == old_num_experts:
            new_param = param.index_select(0, keep).contiguous()
            _set_tensor_by_name(module, name, new_param, is_buffer=False)
            changed = True

    for name, buf in list(module.named_buffers(recurse=True)):
        if buf is None:
            continue
        if buf.ndim > 0 and buf.shape[0] == old_num_experts:
            new_buf = buf.index_select(0, keep).contiguous()
            _set_tensor_by_name(module, name, new_buf, is_buffer=True)
            changed = True

    return changed


def prune_glm5_moe_inplace(moe, retained_expert_indices):
    """
    In-place pruning for GLM-5 across both:
    - native / FP8 packed expert tensors
    - AWQ/GPTQModel exploded per-expert ModuleList layouts
    """
    keep_list = [int(i) for i in retained_expert_indices]
    keep = torch.as_tensor(
        keep_list,
        dtype=torch.long,
        device=moe.gate.weight.device,
    )

    old_num_experts = int(moe.n_routed_experts)
    new_num_experts = int(keep.numel())

    changed = False

    # 1) Native / FP8 packed tensors under moe.experts
    changed = _prune_packed_expert_tensors(moe.experts, keep, old_num_experts) or changed

    # 2) AWQ / GPTQModel exploded expert modules under moe.experts
    changed = _prune_modulelists_recursively(moe.experts, keep_list, old_num_experts) or changed

    if not changed:
        child_names = [name for name, _ in moe.experts.named_children()]
        param_names = [name for name, _ in moe.experts.named_parameters(recurse=True)]
        raise RuntimeError(
            "prune_glm5_moe_inplace could not find a recognized GLM-5 expert layout. "
            f"experts_cls={moe.experts.__class__.__name__}, "
            f"child_names={child_names[:30]}, "
            f"sample_params={param_names[:30]}"
        )

    # Router is still expert-axis even under AWQ.
    moe.gate.weight = nn.Parameter(
        moe.gate.weight.index_select(0, keep).contiguous(),
        requires_grad=False,
    )
    if hasattr(moe.gate, "e_score_correction_bias") and moe.gate.e_score_correction_bias is not None:
        moe.gate.e_score_correction_bias = (
            moe.gate.e_score_correction_bias.index_select(0, keep).contiguous()
        )

    # Metadata
    moe.n_routed_experts = new_num_experts
    moe.gate.n_routed_experts = new_num_experts
    if hasattr(moe.experts, "num_experts"):
        moe.experts.num_experts = new_num_experts

    new_top_k = min(int(moe.top_k), new_num_experts)
    moe.top_k = new_top_k
    moe.gate.top_k = new_top_k

    if hasattr(moe, "config"):
        moe.config.n_routed_experts = new_num_experts
        moe.config.num_experts_per_tok = new_top_k


def assert_merge(model, merged_moe, cluster_label):
    model_attr = MODEL_ATTRS.get(model.__class__.__name__)
    assert hasattr(merged_moe, "experts"), (
        "The merged module must have an 'experts' attribute."
    )

    gate_proj = model_attr["gate_proj"]
    down_proj = model_attr["down_proj"]

    if model_attr["fused"]:
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert torch.allclose(
                    getattr(merged_moe.experts, gate_proj)[dom_expert],
                    getattr(merged_moe.experts, gate_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
                assert torch.allclose(
                    getattr(merged_moe.experts, down_proj)[dom_expert],
                    getattr(merged_moe.experts, down_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
    else:
        up_proj = model_attr["up_proj"]
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert (
                    getattr(merged_moe.experts[dom_expert], up_proj).weight
                    == getattr(merged_moe.experts[expert], up_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], down_proj).weight
                    == getattr(merged_moe.experts[expert], down_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], gate_proj).weight
                    == getattr(merged_moe.experts[expert], gate_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."


def patched_model_map(model: str):
    patched = False
    model_name = model

    if model == "deepseek-ai/DeepSeek-V2-Lite-Chat":
        patched = True
        model_name = "artifacts/models/DeepSeek-V2-Lite-Chat"

    # until hf version lands
    if model == "baidu/ERNIE-4.5-21B-A3B-PT":
        patched = True
        model_name = "artifacts/models/ERNIE-4.5-21B-A3B-PT"

    if model == "Qwen/NonUniformQwen3-30B-A3B":
        patched = True
        model_name = "artifacts/models/NonUniformQwen3-30B-A3B"

    if model == "zai-org/GLM-4.5-Air":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air"

    if model == "zai-org/GLM-4.5-Air-FP8":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air-FP8"

    if model == "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":
        patched = True
        model_name = "artifacts/models/Qwen3-Coder-480B-A35B-Instruct-FP8"

    if patched:
        logger.info(f"Using patched model for {model} from: {model_name}")
    return model_name


def assert_tied_weights(model, clusters_labels):
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    for layer_idx in clusters_labels:
        clusters = clusters_labels[layer_idx]
        moe = get_moe(model, layer_idx)
        experts = getattr(moe, model_attrs["experts"])
        for cluster_idx in torch.unique(clusters):
            experts_in_cluster = torch.where(clusters == cluster_idx)[0].tolist()
            dom_expert = experts[experts_in_cluster[0]]
            for attr in ["up_proj", "down_proj", "gate_proj"]:
                for expert_idx in experts_in_cluster:
                    if expert_idx == dom_expert:
                        continue
                    expert = experts[expert_idx]
                    proj = getattr(expert, attr)
                    weight = proj.weight
                    dom_proj = getattr(dom_expert, attr)
                    dom_weight = dom_proj.weight
                    if not torch.allclose(weight, dom_weight):
                        print(
                            f"Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and attr {attr} are not tied!"
                        )
                        print(f"Max diff: {torch.abs(weight - dom_weight).max()}")
                    # check adapters
                    for lora_adapter in ["lora_A", "lora_B"]:
                        if hasattr(proj, lora_adapter):
                            lora_weight = getattr(proj, lora_adapter).default.weight
                            dom_lora_weight = getattr(
                                dom_proj, lora_adapter
                            ).default.weight
                            if not torch.allclose(lora_weight, dom_lora_weight):
                                print(
                                    f"LoRA Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and adapter {lora_adapter} are not tied!"
                                )
                                print(
                                    f"Max diff: {torch.abs(lora_weight - dom_lora_weight).max()}"
                                )

def get_super_expert_indices(observer_data, include_last_layers: bool = False):
    logger.info("Identifying super experts to preserve...")
    quantile = 99.5
    times = 10
    all_max_activations = [layer['max_activations'] for layer in observer_data.values()]
    num_layers = len(all_max_activations)
    all_max_activations = torch.cat(all_max_activations).flatten()
    percentile_threshold = torch.quantile(all_max_activations, quantile / 100.0).item()
    abs_threshold = all_max_activations.max().item() / times
    final_threshold = max(percentile_threshold, abs_threshold)
    # reshape back into per layer data
    all_max_activations = all_max_activations.reshape(num_layers, -1)
    super_experts_mask = all_max_activations > final_threshold
    if not include_last_layers:
        # only consider first 75% of layers for super experts
        logger.info(
            "Only considering first 75% of layers for super expert "
            "identification since perserve_outliers is False"
        )
        num_layers = int(num_layers * 0.75)
        super_experts_mask[num_layers:, :] = False
    super_expert_idx = torch.argwhere(super_experts_mask)
    logger.info(f"Identified {super_experts_mask.sum().item()} super experts with threshold: {final_threshold:.4f}")
    return super_expert_idx

def register_llama_with_vllm():
    from vllm.model_executor.models import ModelRegistry
    print("Registering Llama4ForCausalLM with vLLM")
    ModelRegistry.register_model("Llama4ForCausalLM", "vllm.model_executor.models.llama4:Llama4ForCausalLM")