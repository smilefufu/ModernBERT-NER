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

## 模型权重说明

### 新增的模型组件

为了实现医学文本的实体关系提取任务，我们在原始 ModernBERT 模型的基础上添加了以下组件：

1. 实体范围预测头：
   ```
   - span_classifier.weight
   - span_classifier.bias
   ```

2. 实体类型分类头：
   ```
   - entity_type_classifier.weight
   - entity_type_classifier.bias
   ```

3. SPO关系模式分类头：
   ```
   - spo_pattern_classifier.weight
   - spo_pattern_classifier.bias
   ```

这些组件在加载预训练权重时会显示为"缺少的键"，这是正常的，因为它们是我们新增的任务特定组件，会在训练过程中从随机初始化开始学习。

### 未使用的预训练组件

原始 ModernBERT 模型中有一些用于预训练任务的组件，在我们的任务中不需要使用：

1. MLM（掩码语言模型）预测头：
   ```
   - head.dense.weight
   - head.norm.weight
   ```

2. MLM 解码器：
   ```
   - decoder.bias
   ```

这些组件在加载模型时会显示为"未使用的键"，这是正常的，因为我们只使用 ModernBERT 的基础编码器部分，而不需要这些预训练任务相关的组件。

## 注意事项

1. **模型加载**
   - 模型的固定参数会从预训练模型的 `config.json` 中自动读取
   - 不要在配置文件中覆盖模型的固定参数

2. **序列长度**
   - 虽然模型支持最大 8192 tokens，但建议根据实际需求和硬件限制设置 `max_seq_length`
   - 使用较长序列时需要相应减小 batch size

3. **性能优化**
   - 训练时可以禁用 cache (`use_cache: false`)
   - 可以使用梯度累积来模拟更大的 batch size
   - 可以使用 `freeze_backbone` 选择是否冻结预训练模型参数
