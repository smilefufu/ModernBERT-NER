import torch
import torch.nn as nn
from transformers import ModernBertPreTrainedModel, ModernBertModel
from torch.nn import CrossEntropyLoss
import logging

logger = logging.getLogger(__name__)

class ModernBertForRelationExtraction(ModernBertPreTrainedModel):
    """基于 ModernBERT 的实体关系抽取模型"""
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    base_model_prefix = "model"
    
    def __init__(self, config):
        super().__init__(config)
        
        # 设置任务相关的参数
        self.num_labels = config.num_labels if hasattr(config, "num_labels") else 3
        self.num_relations = config.num_relations if hasattr(config, "num_relations") else 53
        self.entity_types = config.entity_types if hasattr(config, "entity_types") else [
            "疾病", "症状", "检查", "手术", "药物", "其他治疗", 
            "部位", "社会学", "流行病学", "预后", "其他"
        ]
        self.num_entity_types = len(self.entity_types)
        
        # ModernBERT 基础模型
        self.model = ModernBertModel(config)
        
        # 实体识别分类器
        self.ner_head = nn.Linear(config.hidden_size, self.num_labels)
        
        # 实体类型分类头
        self.entity_type_head = nn.Linear(config.hidden_size, self.num_entity_types)
        
        # 关系分类器
        self.relation_head = nn.Linear(config.hidden_size * 2, self.num_relations)
        
        # 初始化权重
        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        relations=None,
        entity_spans=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取 BERT 输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        sequence_output = outputs.last_hidden_state
        
        # 将 padding tokens 的特征置为 0
        if attention_mask is not None:
            sequence_output = sequence_output * attention_mask.unsqueeze(-1)
        
        # 实体识别
        ner_logits = self.ner_head(sequence_output)
        
        # 关系分类
        batch_size = sequence_output.size(0)
        max_relations = entity_spans.size(1)
        sequence_lengths = attention_mask.sum(dim=1)
        
        # 初始化关系logits
        relation_logits = torch.zeros(batch_size, max_relations, self.num_relations, device=sequence_output.device)
        
        # 对每个批次分别裁剪到其实际序列长度
        for i in range(batch_size):
            seq_len = sequence_lengths[i].item()
            entity_spans[i] = torch.clamp(entity_spans[i], min=0, max=seq_len-1)
        
        for i in range(batch_size):
            seq_len = sequence_lengths[i].item()
            
            for j in range(max_relations):
                spans = entity_spans[i, j]
                start1, end1, start2, end2 = spans.tolist()
                
                # 跳过填充的实体对
                if start1 == 0 and end1 == 0 and start2 == 0 and end2 == 0:
                    continue
                
                # 确保end不小于start
                end1 = max(end1, start1)
                end2 = max(end2, start2)
                
                # 确保所有索引都在有效范围内
                start1 = min(start1, seq_len - 1)
                end1 = min(end1, seq_len - 1)
                start2 = min(start2, seq_len - 1)
                end2 = min(end2, seq_len - 1)
                
                # 提取实体表示
                entity1_repr = sequence_output[i, start1:end1+1].mean(dim=0)
                entity2_repr = sequence_output[i, start2:end2+1].mean(dim=0)
                
                # 拼接两个实体的表示
                pair_repr = torch.cat([entity1_repr, entity2_repr], dim=0)
                
                # 计算关系logits
                relation_logits[i, j] = self.relation_head(pair_repr)
        
        loss = None
        if labels is not None and relations is not None:
            # 计算NER损失
            loss_fct = CrossEntropyLoss()
            ner_loss = loss_fct(ner_logits.view(-1, self.num_labels), labels.view(-1))
            
            # 计算关系分类损失
            # 只计算有效关系的损失
            valid_relations_mask = (relations != 0).float()  # [batch_size, max_relations]
            relation_loss = loss_fct(
                relation_logits.view(-1, self.num_relations),
                relations.view(-1)
            ) * valid_relations_mask.view(-1)
            relation_loss = relation_loss.sum() / (valid_relations_mask.sum() + 1e-6)
            
            # 总损失
            loss = ner_loss + relation_loss
            
        if return_dict:
            return {
                'loss': loss,
                'ner_logits': ner_logits,
                'relation_logits': relation_logits,
            }
        
        return loss, ner_logits, relation_logits
