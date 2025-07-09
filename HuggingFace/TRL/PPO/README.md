# 🤖 Hugging Face `.generate()` 输出结构与手动接合的必要性

## 一、背景简介

使用 Hugging Face Transformers 中的 `model.generate()` 生成文本时，常见的返回结构：

```python
output = model.generate(
    input_ids=queries,
    attention_mask=attention_mask,
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=True,
)
```

返回的 `output` 包括：

* `output.sequences`: 包含 `[prompt + response]` 的张量，形状为 `[batch_size, total_seq_len]`
* `output.scores`: 每一步生成 token 的 logits，列表形式，长度为 `gen_len`

---

## 二、为什么不直接使用 `output.sequences`

### 1. `.generate()` 可能插入特殊 token

* 包括 `[PAD]`, `<|endoftext|>` 等特殊标记
* 模型可能对输入做 left-padding 或 truncation，导致 prompt 位置偏移

### 2. `output.scores` 与 `sequences` 长度不一致

* `output.scores` 仅记录 **生成部分** 的 logits
* 若不手动分离 `response`，难以准确对齐 logits 与 token

### 3. 难以分开处理 prompt / response

* 多数训练方法（SFT、PPO、DPO、RM）需要分别处理 prompt 与 response
* 直接使用 `output.sequences` 会导致边界不清晰

---

## 三、推荐的手动接合流程

### ✅ 步骤 1：调用 `.generate()`

```python
output = model.generate(
    input_ids=queries,
    attention_mask=(queries != tokenizer.pad_token_id),
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=True,
)
```

### ✅ 步骤 2：手动提取 response

```python
context_len = queries.shape[1]
response = output.sequences[:, context_len:]
full_sequence = torch.cat((queries, response), dim=1)
```

### ✅ 步骤 3：提取 logits 并对齐 token

```python
logits = torch.stack(output.scores, dim=1)  # [batch, gen_len, vocab_size]
logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
logprobs_for_tokens = logprobs.gather(-1, response.unsqueeze(-1)).squeeze(-1)
```

### ✅ 步骤 4：可选解码（便于调试）

```python
decoded_prompt = tokenizer.batch_decode(queries, skip_special_tokens=True)
decoded_response = tokenizer.batch_decode(response, skip_special_tokens=True)
decoded_full = tokenizer.batch_decode(full_sequence, skip_special_tokens=True)
```

---

## 四、推荐使用情形总结

| 场景                  | 是否推荐手动接合                   | 理由 |
| ------------------- | -------------------------- | -- |
| 推理阶段，无需对齐 logits    | ❌ 可直接使用 `output.sequences` |    |
| RLHF / PPO / DPO 训练 | ✅✅ 强烈推荐手动分割与拼接             |    |
| 需要打分、评价响应质量         | ✅ 推荐拆分后独立处理                |    |

---

# 🚀 高效查找张量中第一个 True 的索引

## 一、应用背景

在训练中经常需要：

* 找到序列中第一个 padding 位置
* 判断 response 有效 token 的范围
* 避免使用 Python 循环，提高效率

---

## 二、推荐函数：`first_true_indices`

```python
def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values
```

---

## 三、工作原理详解

1. `~bools`: 将 True 变为 False，False 变为 True
2. `~bools * row_len`: 非 True 位置赋值一个较大的数字（无效）
3. `+ torch.arange(...)`: 加上当前索引
4. `min(...)`: 找出每行最小值，即第一个 True 的索引

---

## 四、函数优势

| 优势       | 说明                            |
| -------- | ----------------------------- |
| 🚀 高性能   | 完全矢量化，适用于 GPU 运行              |
| 🧠 优雅简洁  | 无需 Python 循环，代码短小易懂           |
| ⚡ 自动边界处理 | 若没有 True，返回行长度（表示无效）          |
| ✅ 支持梯度   | 可嵌入 differentiable pipeline 中 |

---

## 五、不推荐写法

```python
# 存在性能问题与安全隐患：
for row in bools:
    index = (row == True).nonzero()[0]  # 若没有 True 会抛异常
```

---

## 六、实际示例：用于 PPO 中获取响应长度

```python
response_mask = (response == pad_token_id)
valid_response_lengths = first_true_indices(response_mask) - 1
```

该逻辑用于判断生成响应中最后一个有效 token 的位置。

---

## 七、适用场景总结

| 目标                             | 推荐方法                      |
| ------------------------------ | ------------------------- |
| 获取响应 token 有效长度                | 使用 `first_true_indices()` |
| 替代 for 循环搜索 True               | 使用矢量化实现                   |
| PPO / DPO / RLHF 中 response 截断 | 配合 `pad_token_id` 判断      |


# 🎯 PPO 中为何分别使用 `query_response` 和 `postprocessed_query_response`

在 PPO（Proximal Policy Optimization）训练语言模型中，对同一个模型输出（response），我们会构造两个不同版本的输入：

## 📦 变量区别

| 变量名                            | 含义                      | 是否截断                   | 用途                               |
| ------------------------------ | ----------------------- | ---------------------- | -------------------------------- |
| `query_response`               | 原始的 `query + response`  | 否                      | 用于 value model（估计每个 token 的状态价值） |
| `postprocessed_query_response` | 截断后的 `query + response` | 是（截断于 `stop_token_id`） | 用于 reward model（对最终 response 打分） |

---

## 🧠 为什么用两个不同版本？

### ✅ Value Model 使用 `query_response`

* value model 需要估计每个 token 的 "未来回报"（V）
* 所以必须看到 **完整的序列**（包括 stop token 后的 token）
* 用于计算 PPO 的 advantage：

```math
A_t = R_t - V(s_t)
```

---

### ✅ Reward Model 使用 `postprocessed_query_response`

* reward model 用于给整个 response 打一个总体分数（R）
* response 中 stop token 之后的 token 往往是模型“乱生成”的内容，如废话、重复等
* 所以必须 **截断至 stop\_token\_id**，确保 reward 不受无效 token 干扰

---

## 🔄 调用示例

```python
# full value estimation from entire response
full_value, _, _ = get_reward(value_model, query_response, pad_token_id, context_length)
value = full_value[:, context_length-1:-1].squeeze(-1)  # for PPO advantage

# final scalar reward from cleaned response
_, score, _ = get_reward(reward_model, postprocessed_query_response, pad_token_id, context_length)
```

---

## ✅ 总结

> 使用两个版本，是为了确保：
>
> * value model 的输入 **信息完整**，用于学习价值估计
> * reward model 的输入 **语义干净**，避免评分受到干扰

这种输入分离，是 RLHF/PPO 实践中的重要工程优化策略。
