# ModernBERT 医学实体关系抽取项目

本项目使用 ModernBERT 的多语言改良版本来训练一个医学文本实体关系抽取的模型。

## ModernBERT 模型特点

ModernBERT 是对传统 BERT 模型的现代化改进，引入了多项先进的架构改进：

### 1. 核心架构参数（固定值）
- `hidden_size`: 1024
- `intermediate_size`: 2624
- `num_hidden_layers`: 28
- `num_attention_heads`: 16
- `max_position_embeddings`: 8192
- `vocab_size`: 50368

### 2. 现代化改进
1. **Rotary Positional Embeddings (RoPE)**
   - 支持最大 8192 tokens 的序列长度
   - 使用不同的 RoPE 参数:
     - Global attention: theta = 160000.0
     - Local attention: theta = 10000.0

2. **Alternating Attention 机制**
   - 大部分注意力层使用 128 tokens 的滑动窗口
   - 每隔 3 层使用一次全局注意力 (`global_attn_every_n_layers`: 3)
   - Local attention window size: 128

3. **Unpadding 技术**
   - 优化填充 tokens 的处理
   - 提高混合长度序列的处理效率

4. **GeGLU 激活函数**
   - 替代原始 MLP 层
   - 提升模型性能

### 3. 特殊 Token IDs
- `bos_token_id`: 50281 (同 cls_token_id)
- `eos_token_id`: 50282 (同 sep_token_id)
- `pad_token_id`: 50283

### 4. 其他特性
- 不使用 attention bias (`attention_bias`: false)
- 不使用 classifier bias (`classifier_bias`: false)
- 不使用 mlp bias (`mlp_bias`: false)
- 不使用 norm bias (`norm_bias`: false)
- Layer normalization epsilon: 1e-5
- 不需要`token_type_ids`