"""
CMeIE 数据集加载器
"""
import json
import logging
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def load_json_or_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSON/JSONL 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            return json.load(f)
    except Exception as e:
        logger.error(f"加载文件 {file_path} 失败: {e}")
        raise

class CMeIEDataset(Dataset):
    """CMeIE 数据集"""
    
    def __init__(self, data_file: str, tokenizer, schema_file: str, max_length: int):
        """
        初始化数据集
        Args:
            data_file: 数据文件路径
            tokenizer: transformers tokenizer
            schema_file: schema文件路径
            max_length: 最大序列长度
        """
        # 加载schema
        self.schema = self.load_schema(schema_file)
        self.relation2id = {item: idx for idx, item in enumerate(self.schema)}
        
        # 初始化分词器
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        raw_data = load_json_or_jsonl(data_file)
        self.data = [sample for sample in raw_data if self.validate_sample(sample)]
        logger.info(f"数据集初始化 - 加载了 {len(self.data)}/{len(raw_data)} 个有效样本")

    def load_schema(self, filename: str) -> List[str]:
        """加载schema文件，返回所有predicate列表"""
        schema_data = load_json_or_jsonl(filename)
        return sorted(set(item['predicate'] for item in schema_data))

    def validate_sample(self, sample: Dict) -> bool:
        """验证样本格式"""
        try:
            if not all(k in sample for k in ['text', 'spo_list']):
                return False
            if not isinstance(sample['text'], str) or not sample['text'].strip():
                return False
            if not isinstance(sample['spo_list'], list):
                return False
            return True
        except Exception:
            return False

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        sample = self.data[idx]
        text = sample['text']
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        
        # 获取token的位置映射
        offset_mapping = encoding.pop('offset_mapping')
        
        # 初始化NER标签序列（使用BIO标注方案）
        labels = [0] * len(encoding['input_ids'])  # 0表示O标签
        
        # 记录实体span
        entity_spans = []
        
        # 处理实体
        for entity in sample.get('entities', []):
            start_idx = entity['start_idx']
            # 通过实体文本长度计算结束位置
            end_idx = start_idx + len(entity['entity'])
            
            # 找到实体的token范围
            token_start = None
            token_end = None
            for i, (start, end) in enumerate(offset_mapping):
                if start <= start_idx < end:
                    token_start = i
                if start < end_idx <= end:
                    token_end = i
                    break
            
            if token_start is not None and token_end is not None:
                # 标记实体的token
                labels[token_start] = 1  # B
                for i in range(token_start + 1, token_end + 1):
                    labels[i] = 2  # I
                
                entity_spans.append((token_start, token_end))
        
        # 初始化关系矩阵
        max_relations = 64  # 每个样本最多处理的关系数
        relations = torch.full((max_relations,), -1, dtype=torch.long)
        spans = torch.zeros((max_relations, 4), dtype=torch.long)
        
        # 处理实体关系
        relation_count = 0
        for spo in sample.get('spo_list', []):
            if relation_count >= max_relations:
                break
                
            subject_text = spo['subject']
            object_text = spo['object'].get('@value', '')
            predicate = spo['predicate']
            
            # 使用标注的起始位置
            subject_start = spo['subject_start_idx']
            object_start = spo['object_start_idx']
            
            # 计算实体结束位置
            subject_end = subject_start + len(subject_text)
            object_end = object_start + len(object_text)
            
            # 验证位置的正确性
            if not (text[subject_start:subject_end] == subject_text and 
                   text[object_start:object_end] == object_text):
                logger.warning(f"实体位置与文本不匹配: {subject_text}, {object_text}")
                continue
            
            # 找到对应的token范围
            subject_token_start = None
            subject_token_end = None
            object_token_start = None
            object_token_end = None
            
            for i, (start, end) in enumerate(offset_mapping):
                if start <= subject_start < end:
                    subject_token_start = i
                if start < subject_end <= end:
                    subject_token_end = i
                if start <= object_start < end:
                    object_token_start = i
                if start < object_end <= end:
                    object_token_end = i
            
            if all(x is not None for x in [subject_token_start, subject_token_end, 
                                         object_token_start, object_token_end]):
                spans[relation_count] = torch.tensor([
                    subject_token_start, subject_token_end,
                    object_token_start, object_token_end
                ])
                relations[relation_count] = self.relation2id[predicate]
                relation_count += 1
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels,
            'relations': relations,
            'entity_spans': spans
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    处理batch数据
    Args:
        batch: 样本列表
    Returns:
        batch字典
    """
    # 获取batch中最大的实体数量
    max_entities = max(len(item['entity_spans']) for item in batch)
    
    # 填充关系矩阵
    for item in batch:
        num_entities = len(item['entity_spans'])
        if num_entities < max_entities:
            # 填充关系矩阵
            for row in item['relations']:
                row.extend([0] * (max_entities - num_entities))
            for _ in range(max_entities - num_entities):
                item['relations'].append([0] * max_entities)
            # 填充实体spans
            item['entity_spans'].extend([(0, 0)] * (max_entities - num_entities))
    
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch]),
        'attention_mask': torch.tensor([item['attention_mask'] for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch]),
        'relations': torch.tensor([item['relations'] for item in batch]),
        'entity_spans': torch.tensor([item['entity_spans'] for item in batch])
    }
