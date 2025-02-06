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