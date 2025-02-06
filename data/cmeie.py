"""
CMeIE 数据集加载器：端到端 SPO 三元组提取
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple

import json
import logging

logger = logging.getLogger(__name__)

def load_json_or_jsonl(filename: str) -> List[Dict]:
    """加载 JSON 或 JSONL 格式文件
    
    Args:
        filename: 文件路径
        
    Returns:
        数据列表
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

class CMeIEDataset(Dataset):
    """端到端 SPO 三元组数据集"""
    
    def __init__(self, data_file: str, tokenizer, schema_file: str, max_length: int):
        """初始化数据集
        
        Args:
            data_file: 训练数据文件路径
            tokenizer: 分词器
            schema_file: schema文件路径
            max_length: 最大序列长度
        """
        self.data = load_json_or_jsonl(data_file)
        self.schema = load_json_or_jsonl(schema_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 收集关系模式
        self.spo_patterns = self._collect_spo_patterns()
        
        # 从 1 开始映射，0 预留给"无关系"
        self.spo2id = {pattern: idx+1 for idx, pattern in enumerate(self.spo_patterns)}
        self.id2spo = {idx+1: pattern for pattern, idx in self.spo2id.items()}
        
        self._log_spo_patterns()
    
    def _collect_spo_patterns(self) -> List[Tuple[str, str, str]]:
        """收集所有 SPO 关系模式
        
        Returns:
            排序后的 SPO 关系模式列表
        """
        spo_patterns = []
        for schema_item in self.schema:
            subject_type = schema_item['subject_type']
            predicate = schema_item['predicate']
            object_type = schema_item['object_type']
            
            if isinstance(object_type, list):
                spo_patterns.extend([
                    (subject_type, predicate, obj_type) 
                    for obj_type in object_type
                ])
            else:
                spo_patterns.append((subject_type, predicate, object_type))
        
        return sorted(set(spo_patterns))
    
    def _log_spo_patterns(self):
        """记录 SPO 关系模式"""
        logger.info("SPO 关系模式映射:")
        for pattern, idx in self.spo2id.items():
            subject_type, predicate, object_type = pattern
            logger.info(f"  [{idx}] {subject_type} - {predicate} -> {object_type}")
    
    @property
    def num_relation_types(self) -> int:
        """获取关系类型数量"""
        return len(self.spo2id)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本
        
        Returns:
            处理后的样本，包含：
            - input_ids: torch.LongTensor, 输入token的ID序列
            - attention_mask: torch.LongTensor, 注意力掩码
            - spo_list: List[Dict], SPO三元组列表，用于训练和评估
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
        
        # 处理SPO列表
        spo_list = []
        if 'spo_list' in item:
            for spo in item['spo_list']:
                subject_type = spo['subject_type']
                predicate = spo['predicate']
                object_type = spo['object_type']['@value']
                
                pattern = (subject_type, predicate, object_type)
                if pattern not in self.spo2id:
                    continue
                
                pattern_id = self.spo2id[pattern]
                
                # 获取实体位置
                subject_tokens = self.tokenizer(
                    spo['subject'],
                    add_special_tokens=False
                )
                object_tokens = self.tokenizer(
                    spo['object_type']['@value'],
                    add_special_tokens=False
                )
                
                # 在编码后的文本中找到实体位置
                subject_start = self._find_sublist_index(
                    encoding['input_ids'].tolist(), 
                    subject_tokens['input_ids']
                )
                object_start = self._find_sublist_index(
                    encoding['input_ids'].tolist(), 
                    object_tokens['input_ids']
                )
                
                # 如果未找到位置，跳过此SPO
                if subject_start is None or object_start is None:
                    continue
                
                # 构造SPO三元组
                spo_item = {
                    'predicate': pattern_id,
                    'subject': {
                        'start': subject_start,
                        'end': subject_start + len(subject_tokens['input_ids']) - 1,
                        'text': spo['subject']
                    },
                    'object': {
                        'start': object_start,
                        'end': object_start + len(object_tokens['input_ids']) - 1,
                        'text': spo['object_type']['@value']
                    }
                }
                spo_list.append(spo_item)
        
        # 返回处理后的样本
        return {
            **encoding,
            'spo_list': spo_list
        }
    
    def _find_sublist_index(self, main_list: List[int], sublist: List[int]) -> Optional[int]:
        """在主列表中找到子列表的起始索引
        
        Args:
            main_list: 主列表
            sublist: 子列表
            
        Returns:
            子列表在主列表中的起始索引，如果未找到返回 None
        """
        if not sublist:
            return None
        
        for i in range(len(main_list) - len(sublist) + 1):
            if main_list[i:i+len(sublist)] == sublist:
                return i
        
        return None

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将batch数据整理成tensor格式"""
    input_ids = torch.stack([sample['input_ids'] for sample in batch])
    attention_mask = torch.stack([sample['attention_mask'] for sample in batch])
    spo_list = [sample['spo_list'] for sample in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'spo_list': spo_list
    }
