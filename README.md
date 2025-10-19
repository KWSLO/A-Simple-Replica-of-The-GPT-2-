# A-Simple-Replica-of-The-GPT-2-
# 模型 & 训练 — 简要介绍

## Model & Training — Brief Introduction

**中文（Chinese）**
这是一个教学用的 GPT-2 风格自回归 Transformer 实现（小型复刻）。核心模块包括：词嵌入（token embedding）、可学习位置嵌入（position embedding）、多头自注意力（一次性 qkv 投影、分头并行）、前馈网络（FFN）和若干 Pre-LayerNorm Transformer Block。模型最后通过 `lm_head` 投影到词表得到 logits，并采用权重绑定（weight tying）共享 embedding 与输出权重。训练脚本使用 AdamW 优化器，线性 warmup → 线性 decay 学习率调度，支持混合精度（AMP）、梯度累积和 checkpoint（模型/优化器/调度器/scaler/step 保存与恢复）。数据处理把文本编码为连续 token 流，按 `block_size + 1` 切片得到 `(x, y)` 对：`x = chunk[:-1]`（输入），`y = chunk[1:]`（next-token 目标），便于实现自回归训练。

**要点**：

* 教学/实验导向：代码清晰、模块化，适合理解 GPT 内部机制；
* 训练特性：AdamW + warmup, AMP, grad accumulation, checkpoint；
* 数据切片：拼接 token 流并用 `block_size+1` 切片以获得 next-token 监督；
* 生成：含简单采样接口（temperature/top_k），当前为教学版，不含 KV-cache。

**参考教程**
https://www.bilibili.com/video/BV1qWwke5E3K/?share_source=copy_web&vd_source=c0c9b89f1b7736c273bebef554a571c5

---

**English**
This repository contains a small, educational GPT-2 style autoregressive Transformer implementation. Key components: token embeddings, learned positional embeddings, multi-head self-attention (single qkv projection → split heads → parallel attention), feed-forward network (FFN), and stacked Pre-LayerNorm Transformer blocks. The model ends with an `lm_head` projecting to vocabulary logits; weight tying is used to share the embedding and output weights. The training script uses AdamW optimizer with linear warmup → linear decay learning rate scheduling, supports mixed precision (AMP), gradient accumulation, and checkpointing (saves/restores model, optimizer, scheduler, scaler, step). Data preprocessing concatenates texts to a token stream and slices with windows of `block_size + 1` so that `x = chunk[:-1]` (input) and `y = chunk[1:]` (next-token target), which directly implements autoregressive training.

**Highlights**:

* Educational & modular: easy to read and extend;
* Training features: AdamW, warmup schedule, AMP, accumulation, checkpointing;
* Data slicing: `block_size + 1` windows for direct next-token supervision;
* Generation: simple sampling (temperature/top_k). This is a teaching implementation and does not include KV-cache.

**Refer to the tutoria**
https://www.bilibili.com/video/BV1qWwke5E3K/?share_source=copy_web&vd_source=c0c9b89f1b7736c273bebef554a571c5

---



