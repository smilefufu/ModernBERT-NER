# ModernBERT-RE 项目架构设计

## 1. 项目概述

本项目使用 ModernBERT 实现医疗文本的实体关系抽取任务。项目采用端到端的 SPO 三元组抽取方案，直接从文本中提取主实体(Subject)、关系类型(Predicate)和客实体(Object)。

## 2. 核心模块

### 2.1 模型层 (modern_model_re.py)

#### 职责
- 基于 ModernBERT 的模型结构定义
- 端到端的 SPO 三元组抽取
- 损失计算和预测输出

#### 关键接口
```python
class ModernBertForRelationExtraction:
    def forward(self, 
                input_ids,            # 输入token的ID序列
                attention_mask,       # 注意力掩码
                spo_list=None,        # 训练时的SPO标注
                **kwargs
    ) -> Dict[str, torch.Tensor]:
        """模型前向传播"""
        return {
            'spo_predictions': List[Dict],  # 预测的SPO三元组列表
            'loss': tensor                  # 总损失（训练时）
        }
```

### 2.2 数据层 (data/cmeie.py)

#### 职责
- 数据加载和预处理
- SPO三元组标注处理
- 数据格式转换和标准化

#### 数据格式规范

##### 2.2.1 Dataset 返回格式
每个样本包含以下字段：
```python
{
    # 需要移动到计算设备的 tensor 数据
    'input_ids': torch.LongTensor,      # 输入token的ID序列
    'attention_mask': torch.LongTensor,  # 注意力掩码
    'labels': torch.LongTensor,         # NER标签序列
    'relation_labels': torch.LongTensor, # 关系标签序列
    
    # 不需要移动到计算设备的数据
    'spo_list': List[Dict]              # SPO三元组列表，用于评估
}
```

##### 2.2.2 Batch 数据格式
DataLoader 的 collate_fn 函数将确保每个 batch 中的数据格式如下：
```python
{
    # 需要移动到计算设备的 tensor 数据（batch_size 为第一维）
    'input_ids': torch.LongTensor,      # shape: [batch_size, seq_len]
    'attention_mask': torch.LongTensor,  # shape: [batch_size, seq_len]
    'labels': torch.LongTensor,         # shape: [batch_size, seq_len]
    'relation_labels': torch.LongTensor, # shape: [batch_size, seq_len]
    
    # 不需要移动到计算设备的数据
    'spo_list': List[List[Dict]]        # 长度为 batch_size 的列表，每个元素是该样本的 SPO 列表
}
```

#### 关键接口
```python
class CMeIEDataset(Dataset):
    def __getitem__(self, idx) -> Dict[str, Any]:
        """返回标准化的训练数据"""
        return {
            'input_ids': List[int],       # 文本的token ID
            'attention_mask': List[int],   # 注意力掩码
            'spo_list': List[Dict]        # SPO三元组列表，每个三元组包含：
                                         # - spo_pattern: int, 关系模式ID
                                         # - subject: Dict, 包含：
                                         #   - text: str, 主体实体文本
                                         #   - type: int, 主体实体类型ID
                                         #   - start: int, 主体实体起始位置
                                         #   - end: int, 主体实体结束位置
                                         # - object: Dict, 包含：
                                         #   - @value: str, 客体实体文本
                                         #   - type: Dict, 包含：
                                         #     - @value: int, 客体实体类型ID
                                         #   - start: int, 客体实体起始位置
                                         #   - end: int, 客体实体结束位置
                                         # - Combined: bool, 是否为组合关系
        }
```

### 2.3 数据格式说明

#### SPO三元组
每个 SPO 三元组由以下部分组成：
- Subject（主体实体）：包含文本、类型、位置信息
- Predicate（关系类型）：表示主体和客体之间的关系，如"临床表现"、"药物治疗"等
- Object（客体实体）：包含文本、类型、位置信息

#### SPO 关系模式
- 每个 SPO 关系模式由 (subject_type, predicate, object_type) 三元组唯一确定
- 例如：("疾病", "临床表现", "症状") 表示一个从疾病到症状的临床表现关系
- 关系模式的数字ID是由数据集类在初始化时根据三元组字典序排序后自动分配的
- 同一个关系模式可能出现在多个不同的文本实例中，但每个实例中的具体实体是不同的

#### 实体类型
- 每个实体（主体或客体）都有一个类型，如"疾病"、"症状"等
- 实体类型也会被映射为数字ID，用于模型训练
- 实体类型的数字ID是由数据集类在初始化时根据类型名称排序后自动分配的

### 2.4 ModernBERT 配置

ModernBERT 的配置参数定义在 `ModernBertConfig` 中，主要包括：

#### 基础配置
- `vocab_size`: 词表大小，默认50368
- `hidden_size`: 隐藏层维度，默认768
- `num_hidden_layers`: Transformer层数，默认22
- `num_attention_heads`: 注意力头数，默认12
- `intermediate_size`: MLP中间层维度，默认1152

#### 注意力机制
- `attention_dropout`: 注意力dropout率，默认0.0
- `global_attn_every_n_layers`: 全局注意力层间隔，默认3
- `local_attention`: 局部注意力窗口大小，默认128
- `global_rope_theta`: 全局RoPE基数，默认160000.0
- `local_rope_theta`: 局部RoPE基数，默认10000.0

### 2.5 模型架构特点

1. **端到端的三元组抽取**
   - 直接从文本中提取完整的SPO三元组
   - 避免实体识别和关系分类的分步处理
   - 减少误差累积

2. **混合注意力机制**
   - 交替使用全局注意力和局部注意力
   - 局部注意力使用滑动窗口机制
   - 支持Flash Attention 2优化

3. **位置编码**
   - 使用RoPE (Rotary Position Embedding)
   - 支持全局和局部两种尺度的位置编码

4. **规范化和激活**
   - 使用LayerNorm进行归一化
   - 支持多种激活函数（默认GELU）
   - 可配置的bias和dropout

5. **优化设计**
   - 支持梯度检查点（gradient checkpointing）
   - 可选的稀疏预测模式
   - 编译优化支持

## 模型架构更新说明

### 2025-02-03 更新

#### 1. 配置管理
- 移除了自定义的 `ModernBertConfig` 类，直接使用预训练模型的配置文件
- 使用 `getattr` 获取配置参数，设置合理的默认值：
  - `num_labels=5`（默认支持2个实体类型：B-Type1, I-Type1, B-Type2, I-Type2, O）
  - `num_relations=53`（支持CMeIE数据集的53种关系类型）

#### 2. 模型组件优化
- **Dropout 和归一化**：
  - 添加 `classifier_dropout` 用于分类器的输入
  - 使用 `LayerNorm` 配合 `norm_eps` 和 `norm_bias` 参数
  - 这些参数直接继承自预训练模型的配置，保持一致性

- **实体表示计算**：
  - 移除了实体类型嵌入层，改为直接使用实体的上下文表示
  - 优化注意力机制的实现，使用更简洁的方式计算实体表示
  - 使用 `entity_attention` 和 `entity_norm` 组件处理实体表示

- **关系分类**：
  - 简化了关系分类的特征表示，从 `hidden_size * 4` 减少到 `hidden_size * 2`
  - 移除了实体类型特征的显式编码，改为依赖模型学习隐式特征
  - 添加 `relation_dropout` 和 `relation_norm` 增强特征的鲁棒性

#### 3. 损失计算
- 使用统一的 `CrossEntropyLoss`，移除了自定义权重
- 优化了有效样本的处理方式：
  - NER任务：使用 attention_mask 过滤无效位置
  - 关系分类：使用 -100 标记过滤无效关系

#### 4. 代码优化
- 移除了冗余的配置和权重相关代码
- 简化了模型的前向传播逻辑
- 提高了代码的可读性和维护性

#### 5. 更新原因
1. **配置简化**：直接使用预训练模型的配置，避免配置不一致导致的问题
2. **性能考虑**：
   - 减少了特征维度，降低了计算开销
   - 使用更高效的注意力机制计算实体表示
3. **训练稳定性**：
   - 添加 dropout 和层归一化增强模型鲁棒性
   - 简化损失计算逻辑，使训练更稳定
4. **代码质量**：
   - 提高代码可读性和可维护性
   - 减少重复代码，遵循 DRY 原则

## 3. 训练流程

### 3.1 数据预处理
1. 文本分词和编码
2. SPO三元组标注处理
3. 数据增强和标准化

### 3.2 模型训练
1. 批处理数据准备
2. 前向传播计算损失
3. 反向传播更新参数
4. 验证和评估

### 3.3 模型评估
1. SPO三元组抽取评估
2. 综合性能评估

## 4. 部署和服务

### 4.1 模型导出
- 支持ONNX格式
- 支持TorchScript
- 权重量化选项

### 4.2 推理服务
- RESTful API接口
- 批处理接口
- 性能监控
