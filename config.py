"""
EdgeRLHF项目配置中心
统一管理所有超参数、路径和实验配置
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# 项目路径配置
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# 模型配置
# ============================================================================

@dataclass
class ModelConfig:
    """基础模型配置"""
    base_model_name: str = "distilgpt2"
    model_max_length: int = 512
    pad_token: str = "<|endoftext|>"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"

@dataclass
class SFTConfig:
    """监督微调配置"""
    # 训练参数
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    
    # 数据配置
    max_length: int = 512
    train_on_inputs: bool = False

@dataclass
class RewardModelConfig:
    """奖励模型配置"""
    # 训练参数
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 2
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    
    # 量化配置
    quantization_precisions: List[str] = field(default_factory=lambda: ["bf16", "int8", "int4"])
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # 数据配置
    max_length: int = 512

@dataclass
class PPOConfig:
    """PPO对齐配置"""
    # 基础PPO参数
    learning_rate: float = 1.41e-5
    batch_size: int = 32
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.5
    
    # PPO特定参数
    kl_penalty: str = "kl"
    adap_kl_ctrl: bool = True
    init_kl_coef: float = 0.1
    target_kl: float = 6.0
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    
    # 生成参数
    forward_batch_size: int = 8
    response_length: int = 64
    temperature: float = 1.0
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95

# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验基本信息
    experiment_name: str = "edgerlhf_experiment"
    seed: int = 42
    device: str = "cuda"
    
    # 数据配置
    dataset_name: str = "Anthropic/hh-rlhf"
    train_split: str = "train"
    test_split: str = "test"
    max_train_samples: Optional[int] = 10000
    max_eval_samples: Optional[int] = 1000
    
    # 评估配置
    eval_batch_size: int = 8
    eval_accumulation_steps: int = 1
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 保存配置
    output_dir: str = str(MODELS_DIR)
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    save_strategy: str = "steps"
    evaluation_strategy: str = "steps"
    
    # 日志配置
    logging_dir: str = str(LOGS_DIR)
    report_to: List[str] = field(default_factory=list)  # ["wandb", "tensorboard"]
    logging_first_step: bool = True
    
    # 内存优化
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True

# ============================================================================
# 硬件配置
# ============================================================================

@dataclass
class HardwareConfig:
    """硬件配置 - 针对RTX 4060 8GB优化"""
    gpu_name: str = "RTX 4060"
    vram_gb: int = 8
    
    # 内存优化设置
    max_memory_allocation: float = 0.85  # 85%的VRAM
    cpu_offload: bool = True
    pin_memory: bool = True
    
    # 批处理大小建议
    sft_recommended_batch_size: int = 4
    rm_recommended_batch_size: int = 8
    ppo_recommended_batch_size: int = 32
    
    # 精度建议
    recommended_dtype: str = "bfloat16"
    use_gradient_checkpointing: bool = True

# ============================================================================
# 全局配置实例
# ============================================================================

# 创建配置实例
MODEL_CONFIG = ModelConfig()
SFT_CONFIG = SFTConfig()
REWARD_MODEL_CONFIG = RewardModelConfig()
PPO_CONFIG = PPOConfig()
EXPERIMENT_CONFIG = ExperimentConfig()
HARDWARE_CONFIG = HardwareConfig()

# ============================================================================
# 配置验证和实用函数
# ============================================================================

def get_model_path(model_type: str, precision: str = None) -> Path:
    """获取模型保存路径"""
    if model_type == "sft":
        return MODELS_DIR / "sft"
    elif model_type == "rm":
        if precision:
            return MODELS_DIR / "rm" / precision
        return MODELS_DIR / "rm"
    elif model_type == "ppo":
        if precision:
            return MODELS_DIR / f"ppo_policy_{precision}"
        return MODELS_DIR / "ppo_policy"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_data_path(data_type: str) -> Path:
    """获取数据文件路径"""
    if data_type == "train":
        return DATA_DIR / "train_prefs.jsonl"
    elif data_type == "test":
        return DATA_DIR / "test_prefs.jsonl"
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def get_results_path(experiment_name: str) -> Path:
    """获取实验结果路径"""
    return RESULTS_DIR / experiment_name

def validate_hardware_config():
    """验证硬件配置是否合理"""
    import torch
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory < HARDWARE_CONFIG.vram_gb * 0.9:
            print(f"⚠️ Warning: Detected VRAM ({total_memory:.1f}GB) is less than configured ({HARDWARE_CONFIG.vram_gb}GB)")
    else:
        print("⚠️ Warning: CUDA not available")

def print_config_summary():
    """打印配置摘要"""
    print("🔧 EdgeRLHF Configuration Summary")
    print("=" * 50)
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print(f"🤖 Base Model: {MODEL_CONFIG.base_model_name}")
    print(f"🎯 Hardware Target: {HARDWARE_CONFIG.gpu_name} ({HARDWARE_CONFIG.vram_gb}GB)")
    print(f"📊 Batch Sizes - SFT: {SFT_CONFIG.batch_size}, RM: {REWARD_MODEL_CONFIG.batch_size}, PPO: {PPO_CONFIG.batch_size}")
    print(f"⚡ Precision: {MODEL_CONFIG.torch_dtype}")
    print("=" * 50)

if __name__ == "__main__":
    print_config_summary()
    validate_hardware_config() 