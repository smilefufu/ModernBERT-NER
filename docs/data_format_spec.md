# ModernBERT 医学实体关系抽取模型数据格式规范

## 1. 概述

本文档定义了 ModernBERT 医学实体关系抽取模型的数据格式规范，包括输入数据格式、数据加载接口以及数据转换要求。本规范旨在统一不同来源数据的格式，便于模型训练和评估。

## 2. 数据格式规范

### 2.1 基本数据格式

数据应以 JSON 或 JSONL 格式存储，每个样本包含以下字段：

```json
{
    "text": "原始文本内容",
    "entities": [
        {
            "entity": "实体文本",
            "start_idx": 0,  // 实体在原文中的起始位置。
            "type": "实体类型"  // 必须是预定义的实体类型之一
        }
    ],
    "spo_list": [
        {
            "subject": "主实体文本",
            "predicate": "关系类型",
            "object": {
                "@value": "客实体文本"
            },
            "subject_type": "主实体类型",
            "object_type": "客实体类型",
            "subject_start_idx": 0,  // 主实体在原文中的起始位置
            "object_start_idx": 10,  // 客实体在原文中的起始位置
            "Combined": false  // 是否为组合关系
        }
    ]
}
```

### 2.2 Schema 格式

关系模式（Schema）文件应为 JSONL 格式，每行包含一个关系定义：

```json
{
    "subject_type": "实体类型",
    "predicate": "关系类型",
    "object_type": "实体类型"
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
            data_file: str 或 List, 数据文件路径或数据列表
            tokenizer: transformers.PreTrainedTokenizer, 分词器
            schema_file: str, schema文件路径
            max_length: int, 最大序列长度
        """
        # 加载schema
        self.schema = self.load_schema(schema_file)
        self.relation2id = {item: idx for idx, item in enumerate(self.schema)}
        
        # 初始化分词器
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        if isinstance(data_file, str):
            self.data = self.load_data(data_file)
        else:
            self.data = data_file

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        计算过程：
        1. input_ids 和 attention_mask:
           - 使用 tokenizer 对文本进行编码
           - 返回 list 格式，在 collate_fn 中再转换为 tensor
           
        2. labels (NER标签):
           - 使用 BIO 标注方案
           - 对于每个 token，标注其对应的标签ID
           - 返回与 input_ids 等长的标签序列
           
        3. relations:
           - 使用邻接矩阵表示实体间的关系
           - 矩阵大小取决于实体数量
           - 每个单元格存储关系类型ID
           
        4. entity_spans:
           - 记录实体在 token 序列中的起止位置
           - 需要考虑 tokenizer 分词后的位置偏移
        
        Returns:
            dict: {
                'input_ids': List[int],       # 输入token的ID
                'attention_mask': List[int],   # 注意力掩码
                'labels': List[int],          # NER标签序列
                'relations': List[List[int]],  # 关系邻接矩阵
                'entity_spans': List[tuple],   # 实体span位置列表
            }
        """
        pass

    @staticmethod
    def validate_sample(sample):
        """
        验证样本格式是否合法
        Args:
            sample: dict, 单个样本数据
        Returns:
            bool: 样本是否合法
        """
        pass
```

### 3.2 数据转换接口

为支持不同格式的数据源，提供以下转换接口：

```python
class BaseDataConverter:
    """数据格式转换基类"""
    
    def convert_sample(self, source_sample):
        """
        转换单个样本到标准格式
        Args:
            source_sample: 源数据样本（格式由子类定义）
        Returns:
            dict: 符合标准格式的样本
        """
        raise NotImplementedError
        
    def convert_file(self, source_file, target_file):
        """
        转换整个文件
        Args:
            source_file: str, 源数据文件路径
            target_file: str, 目标文件路径
        """
        # 读取源文件
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
            
        # 转换数据
        converted_data = []
        for sample in source_data:
            converted_sample = self.convert_sample(sample)
            if converted_sample:
                converted_data.append(converted_sample)
                
        # 保存转换后的数据
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
```

### 3.3 数据验证器

提供统一的数据验证工具：

```python
class DataValidator:
    """数据验证器"""
    
    def __init__(self, schema_file):
        """
        初始化验证器
        Args:
            schema_file: str, schema文件路径
        """
        self.schema = self.load_schema(schema_file)
        self.relation_types = {item['predicate'] for item in self.schema}
        self.entity_types = {item['subject_type'] for item in self.schema} | \
                           {item['object_type'] for item in self.schema}
    
    def validate_sample(self, sample):
        """
        验证单个样本
        Args:
            sample: dict, 样本数据
        Returns:
            bool: 是否合法
        """
        try:
            # 1. 验证基本字段
            if not all(k in sample for k in ['text', 'entities', 'spo_list']):
                return False
                
            # 2. 验证文本
            if not isinstance(sample['text'], str) or not sample['text'].strip():
                return False
                
            # 3. 验证实体
            entity_set = set()
            for entity in sample['entities']:
                # 验证实体字段
                if not all(k in entity for k in ['entity', 'start_idx', 'end_idx', 'type']):
                    return False
                # 验证实体类型
                if entity['type'] not in self.entity_types:
                    return False
                # 验证位置索引
                if not (0 <= entity['start_idx'] < entity['end_idx'] <= len(sample['text'])):
                    return False
                # 验证实体文本
                if entity['entity'] != sample['text'][entity['start_idx']:entity['end_idx']]:
                    return False
                entity_set.add(entity['entity'])
                
            # 4. 验证关系
            for spo in sample['spo_list']:
                # 验证关系字段
                if not all(k in spo for k in ['subject', 'predicate', 'object', 'subject_type', 'object_type', 'subject_start_idx', 'object_start_idx']):
                    return False
                # 验证关系类型
                if spo['predicate'] not in self.relation_types:
                    return False
                # 验证实体存在
                if spo['subject'] not in entity_set or spo['object']['@value'] not in entity_set:
                    return False
                    
            return True
            
        except Exception:
            return False
    
    def validate_file(self, data_file):
        """
        验证整个文件
        Args:
            data_file: str, 数据文件路径
        Returns:
            bool: 是否合法
        """
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                return False
                
            return all(self.validate_sample(sample) for sample in data)
            
        except Exception:
            return False
```

## 4. 数据质量要求

### 4.1 基本要求

- 文本长度：建议不超过 1024 个字符
- 实体标注：实体边界准确，类型正确
- 关系标注：关系类型符合 schema 定义

### 4.2 数据验证规则

1. 文本完整性：
   - 不包含 HTML 标签
   - 不包含特殊控制字符
   - 文本不为空

2. 实体有效性：
   - start_idx 和 end_idx 在文本范围内
   - 实体文本与原文对应位置文本一致
   - 实体类型在预定义类型列表中

3. 关系有效性：
   - subject 和 object 必须在 entities 中存在
   - predicate 必须在 schema 中定义
   - subject_type 和 object_type 与 schema 一致

## 5. 使用示例

### 5.1 加载数据

```python
from torch.utils.data import DataLoader

# 初始化数据集
dataset = CMeIEDataset(
    data_file="path/to/data.json",
    tokenizer=tokenizer,
    schema_file="path/to/schema.jsonl",
    max_length=1024
)

# 创建数据加载器
data_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
```

### 5.2 转换其他格式数据

```python
# 转换其他格式数据到标准格式
converter = CustomDataConverter()
converter.convert(
    source_file="path/to/source.json",
    target_file="path/to/target.json",
    source_format="custom"
)
```

## 6. 注意事项

1. 数据隐私：
   - 确保数据已经过脱敏处理
   - 不包含敏感个人信息

2. 性能考虑：
   - 建议使用 JSONL 格式以提高加载效率
   - 大规模数据集考虑使用数据流式加载

3. 扩展性：
   - 新增实体类型需要更新配置文件
   - 新增关系类型需要更新 schema 文件

## 7. 错误处理

数据加载过程中的错误应该被合理处理：

1. 文件格式错误：抛出 `ValueError`
2. 数据格式错误：记录日志并跳过错误样本
3. Schema 不匹配：抛出 `ValueError`

## 8. 版本控制

- 数据格式版本：1.0.0
- 最后更新时间：2025-01-31
- 作者：Cascade AI Assistant
