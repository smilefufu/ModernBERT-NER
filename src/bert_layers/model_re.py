import torch
import torch.nn as nn
from .model import FlexBertPreTrainedModel, FlexBertModel

class FlexBertForRelationExtraction(FlexBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = FlexBertModel(config)
        
        # 实体识别的分类头（BIO标注方案）
        self.entity_classifier = nn.Linear(config.hidden_size, 3)
        
        # 实体类型分类头
        self.entity_type_classifier = nn.Linear(config.hidden_size, len(config.entity_types))
        
        # 关系分类头
        self.relation_classifier = nn.Linear(config.hidden_size * 2, config.num_relations)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_positions=None,
        labels=None,
    ):
        # 获取 BERT 输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        if len(sequence_output.shape) == 2:
            # 如果丢失了batch维度，添加一个
            sequence_output = sequence_output.unsqueeze(0)
        sequence_output = self.dropout(sequence_output)
        
        # 实体识别
        entity_logits = self.entity_classifier(sequence_output)  # [batch_size, seq_len, 3]
        
        # 如果提供了实体位置
        if entity_positions is not None and entity_positions.numel() > 0:
            batch_size = sequence_output.size(0)
            seq_len = sequence_output.size(1)
            hidden_size = sequence_output.size(-1)  # 获取隐藏状态的维度
            entity_pairs = []
            
            # 对每个样本处理
            for i in range(min(batch_size, len(entity_positions))):
                # 获取当前样本的主体和客体位置
                subj_pos, obj_pos = entity_positions[i]
                # 确保位置索引不超过序列长度
                subj_pos = min(subj_pos, seq_len - 1)
                obj_pos = min(obj_pos, seq_len - 1)
                
                # 获取对应的隐藏状态，保持最后的hidden_size维度
                subj_hidden = sequence_output[i:i+1, subj_pos]  # [1, hidden_size]
                obj_hidden = sequence_output[i:i+1, obj_pos]    # [1, hidden_size]
                
                # 将主体和客体的表示组合在一起
                entity_pair = torch.cat([subj_hidden, obj_hidden], dim=0)  # [2, hidden_size]
                entity_pairs.append(entity_pair)
            
            if entity_pairs:
                # 将所有样本的实体对堆叠在一起
                entity_hidden = torch.stack(entity_pairs, dim=0)  # [batch_size, 2, hidden_size]
                
                # 实体类型分类（对主体和客体分别进行分类）
                batch_size = entity_hidden.size(0)
                num_entities = entity_hidden.size(1)  # 应该是2
                
                # 重塑张量以进行分类
                entity_hidden_flat = entity_hidden.reshape(-1, hidden_size)  # [batch_size*2, hidden_size]
                
                # 对每个实体进行类型分类
                entity_type_logits = self.entity_type_classifier(entity_hidden_flat)  # [batch_size*2, num_entity_types]
                entity_type_logits = entity_type_logits.reshape(batch_size, num_entities, -1)  # [batch_size, 2, num_entity_types]
                
                # 关系分类（使用实体对的拼接表示）
                relation_logits = self.relation_classifier(
                    torch.cat([entity_hidden[:, 0], entity_hidden[:, 1]], dim=-1)
                )
                
                outputs = (entity_logits, entity_type_logits, relation_logits)
            else:
                outputs = (entity_logits,)
        else:
            outputs = (entity_logits,)
            
        # 如果提供了标签，计算损失
        if labels is not None:
            entity_labels, entity_type_labels, relation_labels = labels
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # 使用-100作为忽略的标签
            
            # 计算实体识别损失
            # 确保entity_logits的batch_size与entity_labels匹配
            if entity_logits.size(0) != entity_labels.size(0):
                # 如果batch_size不匹配，复制entity_logits到正确的batch_size
                entity_logits = entity_logits.expand(entity_labels.size(0), -1, -1)
            
            # 只计算有效的位置（非padding）
            active_loss = attention_mask.reshape(-1) == 1
            active_logits = entity_logits.reshape(-1, entity_logits.size(-1))[active_loss]
            active_labels = entity_labels.reshape(-1)[active_loss]
            entity_loss = loss_fct(active_logits, active_labels)
            
            if entity_positions is not None and entity_positions.numel() > 0 and entity_type_labels is not None:
                # 计算实体类型损失
                # 只计算有效的实体类型标签（非-100）
                active_type_loss = entity_type_labels.reshape(-1) != -100
                active_type_logits = entity_type_logits.reshape(-1, len(self.config.entity_types))[active_type_loss]
                active_type_labels = entity_type_labels.reshape(-1)[active_type_loss]
                entity_type_loss = loss_fct(active_type_logits, active_type_labels)
                
                # 计算关系分类损失
                relation_loss = loss_fct(relation_logits, relation_labels)
                
                total_loss = entity_loss + entity_type_loss + relation_loss
            else:
                total_loss = entity_loss
                
            outputs = (total_loss,) + outputs
            
        return outputs