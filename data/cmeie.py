"""
CMeIE 数据集加载器
"""
import json
import logging
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple, Set

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

    def extract_entities_from_spo(self, text: str, spo_list: List[Dict]) -> List[Tuple[str, int, int]]:
        """从 spo_list 中提取实体信息
        Args:
            text: 原始文本
            spo_list: SPO列表
        Returns:
            实体列表，每个实体是一个元组 (实体文本, 起始位置, 结束位置)
        """
        entities: Set[Tuple[str, int, int]] = set()
        
        for spo in spo_list:
            # 处理主实体
            subject_text = spo['subject']
            subject_start = spo['subject_start_idx']
            subject_end = subject_start + len(subject_text)
            
            # 处理客实体
            object_text = spo['object'].get('@value', '')
            object_start = spo['object_start_idx']
            object_end = object_start + len(object_text)
            
            # 验证位置的正确性
            if text[subject_start:subject_end] == subject_text:
                entities.add((subject_text, subject_start, subject_end))
            else:
                logger.warning(f"主实体位置与文本不匹配: {subject_text}")
                
            if text[object_start:object_end] == object_text:
                entities.add((object_text, object_start, object_end))
            else:
                logger.warning(f"客实体位置与文本不匹配: {object_text}")
        
        # 按照起始位置排序，确保标注的一致性
        return sorted(entities, key=lambda x: (x[1], x[2]))

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
            return_offsets_mapping=True,
            add_special_tokens=True,  # 确保添加特殊token
            return_token_type_ids=True,  # 返回token类型ID
            return_attention_mask=True  # 返回注意力掩码
        )
        
        # 输出分词调试信息（仅对前100个样本）
        if idx < 100:
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            logger.debug(f"\n样本 {idx} 分词结果:")
            logger.debug(f"原文: {text}")
            logger.debug(f"分词: {' '.join(tokens)}")
            logger.debug(f"位置映射: {encoding['offset_mapping']}")
        
        # 获取token的位置映射
        offset_mapping = encoding.pop('offset_mapping')
        
        # 初始化NER标签序列（使用BIO标注方案）
        labels = [0] * len(encoding['input_ids'])  # 0表示O标签
        
        # 从 spo_list 中提取实体信息
        entities = self.extract_entities_from_spo(text, sample['spo_list'])
        
        # 标注实体
        for entity_text, start_idx, end_idx in entities:
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
        
        # 初始化关系矩阵
        max_relations = 64  # 每个样本最多处理的关系数
        relations = torch.full((max_relations,), -1, dtype=torch.long)
        spans = torch.zeros((max_relations, 4), dtype=torch.long)
        
        # 处理实体关系
        relation_count = 0
        for spo in sample['spo_list']:
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
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    relations = torch.stack([item['relations'] for item in batch])
    entity_spans = torch.stack([item['entity_spans'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'relations': relations,
        'entity_spans': entity_spans
    }
