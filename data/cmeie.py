"""
CMeIE 数据集加载器
"""
import torch
import json
import logging
from torch.utils.data import Dataset
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_json_or_jsonl(filename: str) -> List[Dict]:
    """加载JSON或JSONL格式的文件
    
    Args:
        filename: 文件路径
        
    Returns:
        数据列表
    """
    with open(filename, 'r', encoding='utf-8') as f:
        # 尝试作为JSON文件加载
        try:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            return data
        except json.JSONDecodeError:
            # 如果JSON加载失败，尝试作为JSONL加载
            f.seek(0)
            data = []
            for line in f:
                if line.strip():  # 跳过空行
                    item = json.loads(line.strip())
                    if not isinstance(item, list):
                        data.append(item)
                    else:
                        data.extend(item)
            return data

class CMeIEDataset(Dataset):
    """CMeIE 数据集"""
    
    def __init__(self, data_file: str, tokenizer, schema_file: str, max_length: int):
        """初始化数据集
        
        Args:
            data_file: 训练数据文件路径
            tokenizer: 分词器
            schema_file: schema文件路径
            max_length: 最大序列长度
        """
        self.data = load_json_or_jsonl(data_file)
        self.schema = self.load_schema(schema_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 收集所有实体类型
        self.entity_types = set()
        for schema_item in self.schema:
            self.entity_types.add(schema_item['subject_type'])
            object_type = schema_item['object_type']
            if isinstance(object_type, list):
                self.entity_types.update(object_type)
            else:
                self.entity_types.add(object_type)
        
        # 构建实体类型映射
        self.entity_type2id = {t: i for i, t in enumerate(sorted(self.entity_types))}
        self.id2entity_type = {i: t for t, i in self.entity_type2id.items()}
        
        # 构建 SPO 关系模式映射
        # 每个关系模式由 (subject_type, predicate, object_type) 三元组唯一确定
        self.spo_patterns = []
        for schema_item in self.schema:
            subject_type = schema_item['subject_type']
            predicate = schema_item['predicate']
            object_type = schema_item['object_type']
            if isinstance(object_type, list):
                # 如果 object_type 是列表，为每个类型创建一个模式
                for obj_type in object_type:
                    self.spo_patterns.append((subject_type, predicate, obj_type))
            else:
                self.spo_patterns.append((subject_type, predicate, object_type))
        
        # 对关系模式排序以确保映射的一致性
        self.spo_patterns.sort()
        self.spo2id = {pattern: idx for idx, pattern in enumerate(self.spo_patterns)}
        self.id2spo = {idx: pattern for pattern, idx in self.spo2id.items()}
        
        logger.info(f"SPO关系模式映射:")
        for pattern, idx in self.spo2id.items():
            subject_type, predicate, object_type = pattern
            logger.info(f"  [{idx}] {subject_type} - {predicate} -> {object_type}")

    def load_schema(self, schema_file: str) -> List[Dict]:
        """加载schema文件
        
        Args:
            schema_file: schema文件路径
            
        Returns:
            schema列表
        """
        return load_json_or_jsonl(schema_file)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本
        
        Returns:
            处理后的样本，包含：
            - input_ids: torch.LongTensor, 输入token的ID序列
            - attention_mask: torch.LongTensor, 注意力掩码
            - labels: torch.LongTensor, NER标签序列
            - relation_labels: torch.LongTensor, 关系标签序列
            - spo_list: List[Dict], SPO三元组列表，用于评估
        """
        item = self.data[idx]
        text = item['text']
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 去掉第一维的batch维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 初始化标签序列
        seq_len = len(encoding['input_ids'])
        labels = torch.zeros(seq_len, dtype=torch.long)  # [seq_len]
        relation_labels = torch.zeros(seq_len, dtype=torch.long)  # [seq_len]
        
        # 处理SPO列表
        spo_list = []
        if 'spo_list' in item:
            for spo in item['spo_list']:
                # 获取关系模式
                subject_type = spo['subject_type']
                predicate = spo['predicate']
                object_type = spo['object_type']['@value']
                
                pattern = (subject_type, predicate, object_type)
                if pattern not in self.spo2id:
                    logger.warning(f"未知的关系模式: {pattern}")
                    continue
                
                pattern_id = self.spo2id[pattern]
                
                # 获取实体类型ID
                if subject_type not in self.entity_type2id:
                    logger.warning(f"未知的主体实体类型: {subject_type}")
                    continue
                if object_type not in self.entity_type2id:
                    logger.warning(f"未知的客体实体类型: {object_type}")
                    continue
                
                subject_type_id = self.entity_type2id[subject_type]
                object_type_id = self.entity_type2id[object_type]
                
                # 构建新的SPO
                processed_spo = {
                    'spo_pattern': pattern_id,
                    'subject': {
                        'text': spo['subject'],
                        'type': subject_type_id,
                        'start': spo['subject_start_idx'],
                        'end': spo['subject_start_idx'] + len(spo['subject'])
                    },
                    'object': {
                        '@value': spo['object']['@value'],
                        'type': {'@value': object_type_id},
                        'start': spo['object_start_idx'],
                        'end': spo['object_start_idx'] + len(spo['object']['@value'])
                    },
                    'Combined': spo.get('Combined', False)
                }
                spo_list.append(processed_spo)
                
                # 更新标签序列
                # 主体实体标签
                start_pos = spo['subject_start_idx']
                end_pos = start_pos + len(spo['subject'])
                if start_pos < seq_len:
                    labels[start_pos:min(end_pos, seq_len)] = subject_type_id + 1  # 0 表示非实体
                
                # 客体实体标签
                start_pos = spo['object_start_idx']
                end_pos = start_pos + len(spo['object']['@value'])
                if start_pos < seq_len:
                    labels[start_pos:min(end_pos, seq_len)] = object_type_id + 1
                    relation_labels[start_pos:min(end_pos, seq_len)] = pattern_id + 1  # 0 表示无关系
        
        # 返回处理后的样本
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels,
            'relation_labels': relation_labels,
            'spo_list': spo_list
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将batch数据整理成tensor格式
    
    Args:
        batch: 样本列表，每个样本包含：
            - input_ids: torch.LongTensor [seq_len]
            - attention_mask: torch.LongTensor [seq_len]
            - labels: torch.LongTensor [seq_len]
            - relation_labels: torch.LongTensor [seq_len]
            - spo_list: List[Dict]
        
    Returns:
        整理后的batch数据，包含：
            - input_ids: torch.LongTensor [batch_size, seq_len]
            - attention_mask: torch.LongTensor [batch_size, seq_len]
            - labels: torch.LongTensor [batch_size, seq_len]
            - relation_labels: torch.LongTensor [batch_size, seq_len]
            - spo_list: List[List[Dict]]
    """
    # 收集batch中的数据
    # 注意：input_ids 和 attention_mask 已经是 tensor 了
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    relation_labels = torch.stack([item['relation_labels'] for item in batch])
    spo_lists = [item['spo_list'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'relation_labels': relation_labels,
        'spo_list': spo_lists
    }
