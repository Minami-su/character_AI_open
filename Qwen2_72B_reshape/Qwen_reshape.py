import logging
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2MLP
from qwen2.tokenization_qwen2 import Qwen2Tokenizer
from qwen2.configuration_qwen2 import Qwen2Config
import copy

logger = logging.getLogger(__name__)

def get_llm(filename: Optional[str | Path] = None, device: torch.device | str = "cpu"):
    """
    Initialize a new Language Model (LLM).

    Args:
        filename (Optional[str | Path]): Path to the pretrained model.
        device (torch.device | str): Device to load the model on.

    Returns:
        tuple: The LLM model and tokenizer.
    """
    config = Qwen2Config.from_pretrained(filename)
    config.output_hidden_states = True  # Ensure hidden states are returned
    model = Qwen2ForCausalLM.from_pretrained(filename, config=config,low_cpu_mem_usage=True,
    load_in_8bit=False,
    torch_dtype=torch.bfloat16)#.to(device)
    print("load finish")
    tokenizer = Qwen2Tokenizer.from_pretrained(filename)
    print("load finish")
    return model, tokenizer

# def load_weights(model, filename: str | Path) -> None:
#     """Load weights from a checkpoint file."""
#     state_dict = load_file(filename)
#     missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
#     if missing_keys:
#         logger.warning(f"Missing keys when loading state_dict: {missing_keys}")
#     if unexpected_keys:
#         logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
#     print(f"Loaded weights from: {filename} with strict=False")

def modify_intermediate_size(model: Qwen2ForCausalLM, add_dim: int = 128):
    """
    Modify the intermediate_size of MLP layers by adding `add_dim` zero dimensions.

    Args:
        model (Qwen2ForCausalLM): The model to modify.
        add_dim (int): The number of dimensions to add.
    """
    for name, module in model.named_modules():
        if isinstance(module, Qwen2MLP):
            print(f"Modifying MLP layer: {name}")
            
            # 修改 gate_proj (hidden_size -> intermediate_size + add_dim)
            gate_proj = module.gate_proj
            old_weight_gate = gate_proj.weight.data  # (intermediate_size, hidden_size)
            new_weight_gate = torch.cat([
                old_weight_gate,
                torch.zeros((add_dim, gate_proj.in_features), dtype=old_weight_gate.dtype, device=old_weight_gate.device)
            ], dim=0)  # 增加输出特征维度
            gate_proj.out_features += add_dim
            gate_proj.weight = nn.Parameter(new_weight_gate)
            print(f"Updated gate_proj: out_features={gate_proj.out_features}")

            # 修改 up_proj (hidden_size -> intermediate_size + add_dim)
            up_proj = module.up_proj
            old_weight_up = up_proj.weight.data  # (intermediate_size, hidden_size)
            new_weight_up = torch.cat([
                old_weight_up,
                torch.zeros((add_dim, up_proj.in_features), dtype=old_weight_up.dtype, device=old_weight_up.device)
            ], dim=0)  # 增加输出特征维度
            up_proj.out_features += add_dim
            up_proj.weight = nn.Parameter(new_weight_up)
            print(f"Updated up_proj: out_features={up_proj.out_features}")

            # 修改 down_proj (intermediate_size + add_dim -> hidden_size)
            down_proj = module.down_proj
            old_weight_down = down_proj.weight.data  # (hidden_size, intermediate_size)
            new_weight_down = torch.cat([
                old_weight_down,
                torch.zeros((down_proj.out_features, add_dim), dtype=old_weight_down.dtype, device=old_weight_down.device)
            ], dim=1)  # 增加输入特征维度
            down_proj.in_features += add_dim
            down_proj.weight = nn.Parameter(new_weight_down)
            print(f"Updated down_proj: in_features={down_proj.in_features}")

            # 更新模型的 intermediate_size
    model.config.intermediate_size += add_dim
    print(f"Updated intermediate_size to {model.config.intermediate_size}")

def save_modified_weights(model: Qwen2ForCausalLM,tokenizer: Qwen2Tokenizer, save_path: str | Path):
    """
    Save the modified model weights using safetensors.

    Args:
        model (Qwen2ForCausalLM): The modified model.
        save_path (str | Path): Path to save the modified weights.
    """
    #state_dict = model.state_dict()
    print(f"Saving... modified weights to: {save_path}")
    #save_file(state_dict, save_path)
    model.save_pretrained(save_path, max_shard_size="4GB")
    tokenizer.save_pretrained(save_path)
    print(f"Saved modified weights to: {save_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Modify Qwen2 MLP intermediate_size by adding zero dimensions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the modified weights.")
    parser.add_argument("--add_dim", type=int, default=128, help="Number of zero dimensions to add.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load the model on.")
    
    args = parser.parse_args()

    # 初始化模型和 tokenizer
    model, tokenizer = get_llm(args.model_path, args.device)
    #load_weights(model, args.model_path)
    # 修改模型的 intermediate_size
    modify_intermediate_size(model, add_dim=args.add_dim)
    print(model)
    # 保存修改后的权重
    save_modified_weights(model,tokenizer, args.save_path)

if __name__ == "__main__":
    main()
