import json
import pathlib
import torch
import torch.nn as nn
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# EDIT THESE PATHS
# -----------------------------
SOURCE_MODEL = "/models/GLM-5-AWQ"
RETAINED_JSON = "./retained_experts.json"
OUTPUT_DIR = "/root/reap/artifacts/GLM-5-AWQ-pruned-20"
# -----------------------------


def get_transformer_layers_root(model):
    """
    Find the module that owns `.layers` for both native HF models and wrappers
    such as GPTQModel QModel.
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
    layers_root = get_transformer_layers_root(model)
    return getattr(layers_root.layers[layer], "mlp")


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
    changed = False

    for name, child in list(module.named_children()):
        if isinstance(child, nn.ModuleList) and len(child) == old_num_experts:
            setattr(module, name, nn.ModuleList([child[i] for i in keep_list]))
            changed = True
        else:
            changed = _prune_modulelists_recursively(child, keep_list, old_num_experts) or changed

    return changed


def _prune_numeric_children_container(module: nn.Module, keep_list: list[int], old_num_experts: int) -> bool:
    """
    Handle containers that store experts as numeric child names:
      module._modules = {'0': expert0, '1': expert1, ..., '255': expert255, 'act_fn': ...}

    This is the layout you hit with AWQ/GPTQModel.
    """
    changed = False

    child_names = list(module._modules.keys())
    numeric_names = [name for name in child_names if name.isdigit()]

    # Direct expert container case.
    if len(numeric_names) == old_num_experts:
        new_modules = OrderedDict()

        # preserve all non-expert children first (e.g. act_fn)
        for name, child in module._modules.items():
            if not name.isdigit():
                new_modules[name] = child

        # reindex retained experts contiguously: 0..new_num_experts-1
        for new_idx, old_idx in enumerate(keep_list):
            new_modules[str(new_idx)] = module._modules[str(old_idx)]

        module._modules = new_modules
        changed = True

    # Recurse into children too.
    for _, child in list(module.named_children()):
        changed = _prune_numeric_children_container(child, keep_list, old_num_experts) or changed

    return changed


def _prune_packed_expert_tensors(module: nn.Module, keep: torch.Tensor, old_num_experts: int) -> bool:
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
    In-place pruning for GLM-5 across:
    - native / FP8 packed expert tensors
    - exploded nn.ModuleList expert layouts
    - exploded numeric-child expert layouts used by AWQ/GPTQModel
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

    # 1) native packed layout
    changed = _prune_packed_expert_tensors(moe.experts, keep, old_num_experts) or changed

    # 2) ModuleList layout
    changed = _prune_modulelists_recursively(moe.experts, keep_list, old_num_experts) or changed

    # 3) numeric-child layout (AWQ/GPTQModel)
    changed = _prune_numeric_children_container(moe.experts, keep_list, old_num_experts) or changed

    if not changed:
        child_names = [name for name, _ in moe.experts.named_children()]
        param_names = [name for name, _ in moe.experts.named_parameters(recurse=True)]
        raise RuntimeError(
            "Could not find a recognized expert layout. "
            f"experts_cls={moe.experts.__class__.__name__}, "
            f"child_names={child_names[:50]}, "
            f"sample_params={param_names[:50]}"
        )

    # Router is still expert-axis.
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


def sanitize_generation_config(model):
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return
    if getattr(gen_cfg, "do_sample", False) is False:
        if hasattr(gen_cfg, "top_p"):
            gen_cfg.top_p = 1.0
        if hasattr(gen_cfg, "temperature"):
            gen_cfg.temperature = 1.0


def main():
    with open(RETAINED_JSON, "r") as f:
        retained_by_layer = {int(k): [int(x) for x in v] for k, v in json.load(f).items()}

    print(f"Loading model from {SOURCE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_MODEL,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = GPTQModel.load(
        SOURCE_MODEL,
        device_map="auto",
        backend=BACKEND.AWQ_GEMM_TRITON,
    )

    # Basic consistency check from the first pruned layer.
    first_layer = min(retained_by_layer.keys())
    first_moe = get_moe(model, first_layer)
    current_num_experts = int(first_moe.n_routed_experts)
    expected_keep = len(retained_by_layer[first_layer])
    print(f"Current experts/layer: {current_num_experts}")
    print(f"Retained experts from file: {expected_keep}")

    for layer in sorted(retained_by_layer.keys()):
        keep = retained_by_layer[layer]
        moe = get_moe(model, layer)

        if max(keep) >= int(moe.n_routed_experts):
            raise RuntimeError(
                f"Layer {layer}: keep-list contains expert id {max(keep)} "
                f"but model only has {int(moe.n_routed_experts)} experts."
            )

        prune_glm5_moe_inplace(moe, keep)
        print(f"Pruned layer {layer}: kept {len(keep)} experts")

    # keep top-k / config consistent globally too
    final_kept = len(retained_by_layer[first_layer])
    if hasattr(model, "config"):
        if hasattr(model.config, "n_routed_experts"):
            model.config.n_routed_experts = final_kept
        if hasattr(model.config, "num_experts_per_tok"):
            model.config.num_experts_per_tok = min(
                int(model.config.num_experts_per_tok), final_kept
            )

    sanitize_generation_config(model)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving pruned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Copy the keep-lists into the output for traceability.
    with open(output_dir / "retained_experts.json", "w") as f:
        json.dump({str(k): v for k, v in retained_by_layer.items()}, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
