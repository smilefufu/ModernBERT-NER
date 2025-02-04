# ModernBERT 医学实体关系抽取模型数据格式规范

## 1. 概述

本文档定义了 ModernBERT 医学实体关系抽取模型的数据格式规范，包括输入数据格式、SPO三元组标注规范以及数据转换要求。本规范基于 CMeIE 数据集格式，旨在统一不同来源数据的格式，便于模型训练和评估。

## 2. 数据格式规范

### 2.1 基本数据格式

数据以 JSONL 格式存储，每行一个样本。每个样本包含以下字段：

```json
{
    "text": "原始文本内容",
    "spo_list": [
        {
            "Combined": false,        // 是否为组合关系
            "predicate": "关系类型",  // 如：临床表现、并发症等
            "subject": "主实体文本",
            "subject_type": "主实体类型", // 如：疾病、症状等
            "object": {
                "@value": "客实体文本"
            },
            "object_type": {
                "@value": "客实体类型"  // 如：症状、疾病、检查等
            },
            "subject_start_idx": 0,  // 主实体在原文中的起始位置
            "object_start_idx": 10   // 客实体在原文中的起始位置
        }
    ]
}
```

### 2.2 数据字段说明

1. **text**：原始文本内容
   - 可能包含特殊标记，如 "@"、"###" 等
   - 文本长度不限，但建议控制在模型最大输入长度内

2. **spo_list**：SPO三元组列表
   - **Combined**：布尔值，表示是否为组合关系
   - **predicate**：关系类型，如"临床表现"、"并发症"、"药物治疗"等
   - **subject**：主实体文本
   - **subject_type**：主实体类型，如"疾病"、"症状"等
   - **object**：客实体信息，使用 {"@value": "文本"} 格式
   - **object_type**：客实体类型，使用 {"@value": "类型"} 格式
   - **subject_start_idx**：主实体在原文中的起始位置
   - **object_start_idx**：客实体在原文中的起始位置

### 2.3 示例说明

1. **单关系示例**
```json
{
    "text": "类风湿关节炎@尺侧偏斜是由于MCP关节炎症造成的。",
    "spo_list": [
        {
            "Combined": false,
            "predicate": "临床表现",
            "subject": "MCP关节炎症",
            "subject_type": "疾病",
            "object": {"@value": "尺侧偏斜"},
            "object_type": {"@value": "症状"},
            "subject_start_idx": 14,
            "object_start_idx": 7
        }
    ]
}
```

2. **多关系示例**
```json
{
    "text": "胰腺癌@首次治疗4个月后,上腹部超声检查显示胰腺肿块与肝转移。",
    "spo_list": [
        {
            "Combined": false,
            "predicate": "影像学检查",
            "subject": "胰腺癌",
            "subject_type": "疾病",
            "object": {"@value": "上腹部超声检查"},
            "object_type": {"@value": "检查"},
            "subject_start_idx": 0,
            "object_start_idx": 13
        },
        {
            "Combined": false,
            "predicate": "临床表现",
            "subject": "胰腺癌",
            "subject_type": "疾病",
            "object": {"@value": "胰腺肿块"},
            "object_type": {"@value": "症状"},
            "subject_start_idx": 0,
            "object_start_idx": 22
        }
    ]
}
```

3. **组合关系示例**
```json
{
    "text": "成人资料约1/3 AHF患者并发真菌感染，主要为白假丝酵母。",
    "spo_list": [
        {
            "Combined": true,
            "predicate": "并发症",
            "subject": "AHF",
            "subject_type": "疾病",
            "object": {"@value": "真菌感染"},
            "object_type": {"@value": "疾病"},
            "subject_start_idx": 9,
            "object_start_idx": 16
        },
        {
            "Combined": true,
            "predicate": "病因",
            "subject": "真菌感染",
            "subject_type": "疾病",
            "object": {"@value": "白假丝酵母"},
            "object_type": {"@value": "社会学"},
            "subject_start_idx": 16,
            "object_start_idx": 24
        }
    ]
}
```

## 3. 数据加载接口规范

### 3.1 数据集类接口

所有数据集类都应继承自 `torch.utils.data.Dataset`，并实现以下接口：

```python
class BaseDataset(Dataset):
    def __init__(self, data_file, tokenizer, schema_file, max_length):
        """
        初始化数据集
        Args:
            data_file: str, 数据文件路径
            tokenizer: transformers.PreTrainedTokenizer, 分词器
            schema_file: str, schema文件路径
            max_length: int, 最大序列长度
        """
        self.schema = self.load_schema(schema_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_file)

    def __getitem__(self, idx):
        """返回单个样本"""
        return {
            'input_ids': List[int],       # 文本的token ID
            'attention_mask': List[int],   # 注意力掩码
            'spo_list': List[Dict]        # SPO三元组列表，保持原始格式
        }
```

## 4. 数据质量要求

### 4.1 基本要求
1. 文本完整性：原始文本必须完整、无乱码
2. SPO完整性：三元组中的所有字段都必须存在且有效
3. 位置准确性：实体位置必须与原文精确对应
4. 关系有效性：关系类型必须在预定义的schema中存在

### 4.2 数据验证
在数据加载时应进行以下验证：
1. 文本有效性：检查文本是否为空、是否包含无效字符
2. SPO有效性：
   - 验证实体文本与位置信息的一致性
   - 验证关系类型的有效性
   - 验证实体类型的有效性
3. Schema一致性：确保所有关系和实体类型都在schema中定义
