#!/usr/bin/env python3
"""
EdgeRLHF优化配置建议
====================
为RTX 4060 8GB VRAM提供的三种优化配置方案
"""

# 配置方案1：内存保守型 (< 6GB VRAM)
conservative_config = {
    'learning_rate': 1.41e-5,
    'batch_size': 16,                    # 进一步减少
    'mini_batch_size': 1,                # 最小化
    'gradient_accumulation_steps': 8,     # 增加以保持有效批次大小
    'max_grad_norm': 0.5,
    'kl_penalty': 'kl',
    'adap_kl_ctrl': True,
    'init_kl_coef': 0.1,
    'target_kl': 6.0,
    'gamma': 1.0,
    'lam': 0.95,
    'cliprange': 0.2,
    'cliprange_value': 0.2,
    'vf_coef': 0.1,
    'forward_batch_size': 4,             # 最小化前向批次
    'response_length': 32,               # 减少响应长度
}

# 配置方案2：平衡型 (当前配置，已修复)
balanced_config = {
    'learning_rate': 1.41e-5,
    'batch_size': 32,
    'mini_batch_size': 2,
    'gradient_accumulation_steps': 4,
    # 'ppo_epochs': 2,                   # 已移除：新版TRL不支持
    'max_grad_norm': 0.5,
    'kl_penalty': 'kl',
    'adap_kl_ctrl': True,
    'init_kl_coef': 0.1,
    'target_kl': 6.0,
    'gamma': 1.0,
    'lam': 0.95,
    'cliprange': 0.2,
    'cliprange_value': 0.2,
    'vf_coef': 0.1,
    'forward_batch_size': 8,
    'response_length': 64,
}

# 配置方案3：性能优化型 (如果有更多VRAM或内存优化)
performance_config = {
    'learning_rate': 2.5e-5,             # 稍微提高学习率
    'batch_size': 64,                    # 增加批次大小
    'mini_batch_size': 4,                # 增加mini批次
    'gradient_accumulation_steps': 4,
    'max_grad_norm': 0.5,
    'kl_penalty': 'kl',
    'adap_kl_ctrl': True,
    'init_kl_coef': 0.1,
    'target_kl': 6.0,
    'gamma': 1.0,
    'lam': 0.95,
    'cliprange': 0.2,
    'cliprange_value': 0.2,
    'vf_coef': 0.1,
    'forward_batch_size': 16,            # 增加前向批次
    'response_length': 128,              # 增加响应长度
}

# 预期性能指标
performance_metrics = {
    'conservative': {
        'vram_usage_gb': 3.5,
        'training_time_min': 50,
        'expected_reward_score': 0.28,
        'kl_divergence': 0.22,
        'stability': 'Very High'
    },
    'balanced': {
        'vram_usage_gb': 5.5,
        'training_time_min': 35,
        'expected_reward_score': 0.32,
        'kl_divergence': 0.18,
        'stability': 'High'
    },
    'performance': {
        'vram_usage_gb': 7.5,
        'training_time_min': 25,
        'expected_reward_score': 0.38,
        'kl_divergence': 0.15,
        'stability': 'Medium'
    }
}

def print_config_comparison():
    """打印配置方案对比"""
    print("EdgeRLHF配置方案对比")
    print("=" * 50)
    
    configs = {
        '内存保守型': conservative_config,
        '平衡型(当前)': balanced_config,
        '性能优化型': performance_config
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  批次大小: {config['batch_size']}")
        print(f"  响应长度: {config['response_length']}")
        print(f"  学习率: {config['learning_rate']}")
        
        if name == '内存保守型':
            metrics = performance_metrics['conservative']
        elif name == '平衡型(当前)':
            metrics = performance_metrics['balanced']
        else:
            metrics = performance_metrics['performance']
            
        print(f"  预期VRAM: {metrics['vram_usage_gb']}GB")
        print(f"  训练时间: {metrics['training_time_min']}分钟")
        print(f"  奖励分数: {metrics['expected_reward_score']}")

if __name__ == "__main__":
    print_config_comparison() 