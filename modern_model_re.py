"""
ModernBERT for Relation Extraction
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel
import torch.nn.functional as F

class ModernBertForRelationExtraction(PreTrainedModel):
    """ModernBERT for SPO Extraction"""
    
    def __init__(self, config):
        """初始化模型
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        
        # 加载预训练模型
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(config.model_name_or_path)
        
        # 获取隐藏层维度
        hidden_size = config.hidden_size
        
        # SPO 预测层
        self.spo_pattern_classifier = nn.Linear(hidden_size, config.num_spo_patterns + 1)  # 输出维度为 num_spo_patterns + 1，0 表示无关系
        self.entity_type_classifier = nn.Linear(hidden_size, config.num_entity_types + 1)  # 输出维度为 num_entity_types + 1，0 表示非实体
        self.span_classifier = nn.Linear(hidden_size, 2)  # start/end
        
        # 损失函数权重
        self.spo_pattern_loss_weight = getattr(config, 'spo_pattern_loss_weight', 1.0)
        self.entity_loss_weight = getattr(config, 'entity_loss_weight', 1.0)
        self.span_loss_weight = getattr(config, 'span_loss_weight', 1.0)
        
        # 初始化权重
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        relation_labels=None,
        spo_list=None,
        return_dict=None,
    ):
        """前向传播
        
        Args:
            input_ids: torch.LongTensor [batch_size, seq_len] 输入token的ID
            attention_mask: torch.LongTensor [batch_size, seq_len] 注意力掩码
            labels: torch.LongTensor [batch_size, seq_len] NER标签序列
            relation_labels: torch.LongTensor [batch_size, seq_len] 关系标签序列
            spo_list: List[List[Dict]] SPO三元组列表，用于评估
            return_dict: bool 是否返回字典格式
            
        Returns:
            dict:
                - loss: torch.FloatTensor 总损失
                - ner_logits: torch.FloatTensor [batch_size, seq_len, num_entity_types+1] NER预测
                - relation_logits: torch.FloatTensor [batch_size, seq_len, num_spo_patterns+1] 关系预测
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # NER预测
        ner_logits = self.entity_type_classifier(sequence_output)  # [batch_size, seq_len, num_entity_types+1]
        
        # 关系预测
        relation_logits = self.spo_pattern_classifier(sequence_output)  # [batch_size, seq_len, num_spo_patterns+1]
        
        # 计算损失
        total_loss = None
        if labels is not None and relation_labels is not None:
            # NER损失
            ner_loss = F.cross_entropy(
                ner_logits.view(-1, ner_logits.size(-1)),
                labels.view(-1),
                ignore_index=0  # 忽略填充位置
            )
            
            # 关系损失
            relation_loss = F.cross_entropy(
                relation_logits.view(-1, relation_logits.size(-1)),
                relation_labels.view(-1),
                ignore_index=0  # 忽略填充位置
            )
            
            # 总损失
            total_loss = self.entity_loss_weight * ner_loss + self.spo_pattern_loss_weight * relation_loss
        
        if not return_dict:
            output = (ner_logits, relation_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return {
            'loss': total_loss,
            'ner_logits': ner_logits,
            'relation_logits': relation_logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
    
    def predict_spo(self, input_ids, attention_mask, threshold=0.5):
        """预测SPO三元组
        
        Args:
            input_ids: torch.LongTensor [batch_size, seq_len] 输入token的ID
            attention_mask: torch.LongTensor [batch_size, seq_len] 注意力掩码
            threshold: float 实体范围预测的阈值
            
        Returns:
            List[List[Dict]] 预测的SPO三元组列表
        """
        # 获取模型输出
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取预测结果
        ner_logits = outputs['ner_logits']
        relation_logits = outputs['relation_logits']
        
        # 预测实体类型
        predicted_entity_types = torch.argmax(ner_logits, dim=-1)
        
        # 预测关系模式
        predicted_spo_patterns = torch.argmax(relation_logits, dim=-1)
        
        batch_spo_list = []
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            sample_spos = []
            
            # 找到所有可能的实体起始位置
            start_positions = torch.nonzero(predicted_entity_types[batch_idx] > 0).squeeze(-1)
            
            for start_pos in start_positions:
                # 找到对应的结束位置
                end_positions = torch.nonzero(predicted_entity_types[batch_idx, start_pos:] > 0).squeeze(-1)
                if len(end_positions.shape) == 0:
                    end_positions = end_positions.unsqueeze(0)
                
                # 对每个可能的实体范围
                for end_offset in end_positions:
                    end_pos = start_pos + end_offset
                    
                    # 获取该位置的实体类型和关系模式
                    entity_type = predicted_entity_types[batch_idx, start_pos].item()
                    spo_pattern = predicted_spo_patterns[batch_idx, start_pos].item()
                    
                    # 如果预测到了有效的实体类型和关系模式
                    if entity_type > 0 and spo_pattern > 0:
                        # 寻找对应的宾语实体
                        object_start_positions = torch.nonzero(predicted_entity_types[batch_idx, end_pos:] > 0).squeeze(-1)
                        
                        for obj_start_offset in object_start_positions:
                            obj_start_pos = end_pos + obj_start_offset
                            obj_end_positions = torch.nonzero(predicted_entity_types[batch_idx, obj_start_pos:] > 0).squeeze(-1)
                            
                            if len(obj_end_positions.shape) == 0:
                                obj_end_positions = obj_end_positions.unsqueeze(0)
                            
                            for obj_end_offset in obj_end_positions:
                                obj_end_pos = obj_start_pos + obj_end_offset
                                
                                # 获取宾语实体类型
                                obj_entity_type = predicted_entity_types[batch_idx, obj_start_pos].item()
                                
                                # 添加到结果列表
                                spo = {
                                    'spo_pattern': spo_pattern,
                                    'subject': {
                                        'type': entity_type,
                                        'start': start_pos.item(),
                                        'end': end_pos.item()
                                    },
                                    'object': {
                                        'type': obj_entity_type,
                                        'start': obj_start_pos.item(),
                                        'end': obj_end_pos.item()
                                    }
                                }
                                sample_spos.append(spo)
            
            batch_spo_list.append(sample_spos)
        
        return batch_spo_list
