"""
ModernBERT 端到端 SPO 关系提取模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)

class ModernBertForRelationExtraction(nn.Module):
    """端到端 SPO 关系提取模型"""
    
    def __init__(self, config_path: str, num_spo_patterns: int):
        """
        Args:
            config_path: ModernBERT 预训练模型配置路径
            num_spo_patterns: SPO 关系模式数量（不包括"无关系"）
        """
        super().__init__()
        
        # 加载模型配置和预训练模型
        logger.info(f"初始化 ModernBERT 模型，配置路径: {config_path}")
        start_time = time.time()
        
        try:
            # 加载配置
            self.config = AutoConfig.from_pretrained(config_path)
            
            # 加载主干网络
            self.backbone = AutoModel.from_pretrained(config_path)
            
            # 实体跨度分类器
            self.span_classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, 2)  # 二分类：是否为实体
            )
            
            # SPO 关系模式分类器
            self.spo_pattern_classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, num_spo_patterns + 1)  # 输出维度为 num_spo_patterns + 1，0 表示"无关系"
            )
            
            # 存储 SPO 模式分类器的输出维度
            self.num_spo_patterns = num_spo_patterns + 1
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        spo_list: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            spo_list: 训练时的 SPO 标注
        
        Returns:
            包含损失和预测结果的字典
        """
        # 性能分析开始
        start_time = time.time()
        
        try:
            # 获取 BERT 输出
            bert_output = self.backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # 使用最后一层的 [CLS] token 作为序列表示
            sequence_output = bert_output.last_hidden_state
            
            # 预测实体跨度
            span_logits = self.span_classifier(sequence_output)
            
            # 预测 SPO 关系模式
            spo_start_time = time.time()
            batch_size, seq_len, hidden_size = sequence_output.shape
            
            # 生成所有可能的主语-宾语对
            spo_pattern_logits = []
            for batch_idx in range(batch_size):
                batch_span_logits = span_logits[batch_idx]
                batch_sequence_output = sequence_output[batch_idx]
                
                # 找到可能的主语和宾语位置
                subject_candidates = torch.where(batch_span_logits[:, 0] > 0.5)[0]
                object_candidates = torch.where(batch_span_logits[:, 0] > 0.5)[0]
                
                # 初始化当前批次的 SPO 模式 logits
                batch_spo_pattern_logit = []
                
                # 对每个主语-宾语对，预测关系
                for subj_start in subject_candidates:
                    for obj_start in object_candidates:
                        if subj_start != obj_start:  # 避免同一个实体
                            # 使用主语和宾语的表示拼接
                            spo_repr = torch.cat([
                                batch_sequence_output[subj_start], 
                                batch_sequence_output[obj_start]
                            ])
                            
                            # 预测 SPO 模式
                            spo_pattern_logit = self.spo_pattern_classifier(spo_repr)
                            batch_spo_pattern_logit.append(spo_pattern_logit)
                
                # 如果没有候选对，添加一个全零 logit
                if not batch_spo_pattern_logit:
                    batch_spo_pattern_logit.append(
                        torch.zeros(self.num_spo_patterns, device=input_ids.device)
                    )
                
                spo_pattern_logits.append(torch.stack(batch_spo_pattern_logit))
            
            # 如果是训练模式，计算损失
            if spo_list is not None:
                loss = self._compute_loss(
                    span_logits=span_logits, 
                    spo_pattern_logits=spo_pattern_logits,
                    input_ids=input_ids, 
                    spo_list=spo_list
                )
                
                return {
                    'loss': loss,
                    'span_logits': span_logits,
                    'spo_pattern_logits': spo_pattern_logits
                }
            
            return {
                'span_logits': span_logits,
                'spo_pattern_logits': spo_pattern_logits
            }
        
        except Exception as e:
            logger.error(f"前向传播失败: {e}")
            raise
    
    def _compute_loss(
        self, 
        span_logits: torch.Tensor, 
        spo_pattern_logits: List[torch.Tensor],
        input_ids: torch.Tensor, 
        spo_list: List[Dict]
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            span_logits: 实体跨度预测 logits
            spo_pattern_logits: SPO 关系模式预测 logits
            input_ids: 输入 token IDs
            spo_list: SPO 标注列表
        
        Returns:
            总损失
        """
        
        # 创建实体跨度标签
        batch_size, seq_len, _ = span_logits.shape
        span_labels = torch.zeros((batch_size, seq_len), dtype=torch.long, device=span_logits.device)
        
        # 根据 SPO 标注填充实体跨度标签
        for batch_idx, batch_spos in enumerate(spo_list):
            for spo in batch_spos:
                # 标记实体的起始和结束位置
                start, end = spo.get('subject_start', -1), spo.get('subject_end', -1)
                if start != -1 and end != -1:
                    span_labels[batch_idx, start:end+1] = 1
        
        # 计算实体跨度损失
        span_loss = F.cross_entropy(
            span_logits.view(-1, 2),  # 展平为二分类
            span_labels.view(-1),
            reduction='mean'
        )
        
        return span_loss
    
    def predict_spo(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        threshold: float = 0.5
    ) -> List[List[Dict]]:
        """预测 SPO 三元组
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            threshold: 预测阈值
        
        Returns:
            批次级别的 SPO 三元组列表
        """
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            span_logits = outputs['span_logits']
            spo_pattern_logits = outputs['spo_pattern_logits']
        
        batch_spo_list = []
        for batch_idx in range(input_ids.size(0)):
            batch_span_logits = span_logits[batch_idx]
            
            # 找到可能的主语和宾语
            subject_candidates = torch.where(batch_span_logits[:, 0] > threshold)[0]
            object_candidates = torch.where(batch_span_logits[:, 1] > threshold)[0]
            
            batch_spo = []
            for subject_start in subject_candidates:
                for object_start in object_candidates:
                    if subject_start == object_start:
                        continue
                    
                    # 找到最可能的关系模式
                    spo_repr = torch.cat([
                        batch_span_logits[subject_start], 
                        batch_span_logits[object_start]
                    ])
                    spo_pattern_pred = torch.argmax(
                        self.spo_pattern_classifier(spo_repr)
                    ).item()
                    
                    # 跳过"无关系"
                    if spo_pattern_pred == 0:
                        continue
                    
                    spo_item = {
                        'predicate': spo_pattern_pred,
                        'subject': {
                            'start': subject_start.item(),
                            'end': subject_start.item(),
                            'text': ''  # 需要从 tokenizer 恢复
                        },
                        'object': {
                            'start': object_start.item(),
                            'end': object_start.item(),
                            'text': ''  # 需要从 tokenizer 恢复
                        }
                    }
                    batch_spo.append(spo_item)
            
            batch_spo_list.append(batch_spo)
        
        return batch_spo_list
