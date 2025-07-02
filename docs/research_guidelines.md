# EdgeRLHF Research Guidelines

## 🎯 研究目标
EdgeRLHF项目致力于在消费级GPU上实现高质量的RLHF训练，降低AI对齐研究的门槛。

## 📊 实验规范

### 内存管理原则
- **VRAM限制**: 严格控制在8GB以内 (RTX 4060)
- **批次大小**: 优先使用gradient accumulation而非大批次
- **模型量化**: 系统性比较BF16/INT8/INT4的效果
- **检查点**: 定期保存，避免训练中断造成损失

### 可复现性要求
```python
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 固定CUDA算法
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 评估指标
1. **奖励模型质量**
   - 准确率 (Accuracy)
   - AUC-ROC
   - 校准误差 (Calibration Error)

2. **PPO训练效果**
   - 平均奖励分数
   - KL散度 (与SFT模型)
   - 响应质量人工评估

3. **资源效率**
   - VRAM峰值使用量
   - 训练时间
   - 模型大小

## 🛡️ 安全考虑

### 数据安全
- 使用公开数据集 (Anthropic/hh-rlhf)
- 避免训练数据泄露
- 定期检查模型输出的安全性

### 训练稳定性
- 监控梯度爆炸/消失
- 设置合理的学习率调度
- 使用梯度裁剪防止训练崩溃

## 🔬 实验记录

### 必须记录的信息
- 硬件配置 (GPU型号、VRAM、驱动版本)
- 软件环境 (Python、PyTorch、transformers版本)
- 超参数设置
- 训练曲线和最终指标
- 异常情况和解决方案

### 文档格式
```json
{
  "experiment_id": "exp_001",
  "date": "2024-01-01",
  "model_config": {
    "base_model": "distilgpt2",
    "precision": "bf16",
    "lora_rank": 16
  },
  "training_config": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 3
  },
  "results": {
    "final_reward": 0.35,
    "kl_divergence": 0.15,
    "training_time": "35min"
  }
}
```

## 💡 优化策略

### 内存优化
1. **QLoRA**: 4bit量化 + LoRA微调
2. **Gradient Checkpointing**: 时间换空间
3. **DeepSpeed ZeRO**: 分布式优化器状态
4. **模型并行**: 必要时分割大模型

### 训练加速
1. **混合精度**: BF16训练
2. **编译优化**: torch.compile()
3. **数据加载**: 多进程DataLoader
4. **缓存策略**: 预处理数据缓存

## 🚨 常见问题解决

### CUDA OOM (显存不足)
```python
# 降低批次大小
batch_size = batch_size // 2

# 增加梯度累积
gradient_accumulation_steps *= 2

# 启用梯度检查点
model.gradient_checkpointing_enable()
```

### 训练不稳定
```python
# 降低学习率
learning_rate *= 0.5

# 增加预热步数
warmup_steps = total_steps * 0.1

# 使用梯度裁剪
max_grad_norm = 0.5
```

### 奖励模型过拟合
```python
# 增加dropout
dropout = 0.1

# 早停策略
early_stopping_patience = 3

# 数据增强
data_augmentation = True
```

## 📈 性能基准

### 目标指标
- **奖励分数**: > 0.3 (相比SFT baseline 0.12)
- **KL散度**: < 0.2 (保持与原模型的相似性)
- **训练时间**: < 1小时 (完整PPO训练)
- **VRAM使用**: < 8GB峰值

### 对比基准
| 配置 | 奖励分数 | KL散度 | 训练时间 | VRAM |
|------|----------|--------|----------|------|
| BF16 | 0.35 | 0.15 | 35min | 4.7GB |
| INT8 | 0.31 | 0.18 | 40min | 3.2GB |
| INT4 | 0.28 | 0.22 | 45min | 2.1GB |

## 🎓 学习资源

### 推荐阅读
1. InstructGPT论文 (OpenAI, 2022)
2. Constitutional AI (Anthropic, 2022)  
3. Training language models to follow instructions with human feedback
4. Deep reinforcement learning from human preferences

### 相关项目
- [trl](https://github.com/huggingface/trl): HuggingFace RLHF库
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): 内存优化
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes): 量化训练 