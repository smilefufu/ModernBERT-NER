import torch
import torch.nn as nn
from transformers import ModernBertPreTrainedModel, ModernBertModel
from torch.nn import CrossEntropyLoss
import math
import logging
from utils.debug_utils import debug_logger

logger = logging.getLogger(__name__)

class ModernBertForRelationExtraction(ModernBertPreTrainedModel):
    """基于 ModernBERT 的实体关系抽取模型"""
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    base_model_prefix = "model"
    
    def __init__(self, config):
        super().__init__(config)
        
        # 检查词表大小
        if config.vocab_size < 150000:  # 如果词表太小，可能不是多语言模型
            logger.warning(f"警告：词表大小({config.vocab_size})可能不足以支持中文处理。建议使用词表大小>150000的多语言模型。")
        
        # 设置任务相关的参数
        self.entity_types = config.entity_types if hasattr(config, "entity_types") else [
            "疾病", "症状", "检查", "手术", "药物", "其他治疗", 
            "部位", "社会学", "流行病学", "预后", "其他"
        ]
        self.num_entity_types = len(self.entity_types)
        # 每个实体类型有 B、I 标签，加上一个 O 标签
        self.num_labels = self.num_entity_types * 2 + 1
        self.num_relations = config.num_relations if hasattr(config, "num_relations") else 53
        
        # 初始化类别计数器
        self.ner_label_counts = torch.zeros(self.num_labels)
        self.relation_counts = torch.zeros(self.num_relations)
        
        # 是否使用自适应权重
        self.use_adaptive_weights = config.use_adaptive_weights if hasattr(config, "use_adaptive_weights") else True
        
        # 加载类别权重(如果配置中提供)
        self.ner_weights = torch.tensor(config.ner_weights) if hasattr(config, "ner_weights") else None
        self.relation_weights = torch.tensor(config.relation_weights) if hasattr(config, "relation_weights") else None
        
        # ModernBERT 基础模型
        self.model = ModernBertModel(config)
        
        # 实体识别分类器
        self.ner_head = nn.Linear(config.hidden_size, self.num_labels)
        
        # 实体类型嵌入
        self.entity_type_embeddings = nn.Embedding(self.num_entity_types, config.hidden_size)
        
        # 关系分类相关组件
        self.relation_layer_norm = nn.LayerNorm(config.hidden_size * 4)  # 实体表示 + 类型表示
        self.relation_head = nn.Linear(config.hidden_size * 4, self.num_relations)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        # 首先调用父类的init_weights()来初始化ModernBERT基础模型
        super().init_weights()
        
        def init_linear(module, name):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=1.0 / math.sqrt(module.weight.data.size(1)))
                if module.bias is not None:
                    module.bias.data.zero_()
        
        # 初始化任务相关的层
        init_linear(self.ner_head, 'ner_head')
        init_linear(self.relation_head, 'relation_head')
        
        # 初始化LayerNorm
        self.relation_layer_norm.weight.data.fill_(1.0)
        self.relation_layer_norm.bias.data.zero_()
        
        # 初始化实体类型嵌入
        self.entity_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        relations=None,
        entity_spans=None,
        return_dict=None,
    ):
        """前向传播"""
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
        
        # 如果没有提供entity_spans，只返回实体识别结果
        if entity_spans is None:
            return RelationExtractionOutput(
                loss=None,
                ner_logits=ner_logits,
                relation_logits=None,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        # 关系分类
        batch_size = sequence_output.size(0)
        max_relations = entity_spans.size(1)
        sequence_lengths = attention_mask.sum(dim=1)
        
        relation_logits = torch.full((batch_size, max_relations, self.num_relations), 
                                  -1e4, device=sequence_output.device)
        
        # 对每个批次分别处理
        for i in range(batch_size):
            seq_len = sequence_lengths[i].item()
            
            for j in range(max_relations):
                spans = entity_spans[i, j]
                start1, end1, start2, end2 = spans.tolist()
                
                # 跳过填充的实体对
                if start1 == 0 and end1 == 0 and start2 == 0 and end2 == 0:
                    continue
                
                # 确保所有索引都在有效范围内
                if start1 >= seq_len or start2 >= seq_len:
                    continue
                
                try:
                    # 获取实体表示并添加eps防止除零
                    eps = 1e-10
                    
                    # 检查切片的有效性
                    if start1 >= sequence_output.size(1) or end1 >= sequence_output.size(1) or \
                       start2 >= sequence_output.size(1) or end2 >= sequence_output.size(1):
                        continue
                    
                    # 获取实体表示
                    entity1_repr = sequence_output[i, start1:end1+1].mean(dim=0)
                    entity2_repr = sequence_output[i, start2:end2+1].mean(dim=0)
                    
                    # 获取实体类型
                    entity1_type = (labels[i, start1] - 1) // 2  # 减1是因为0是O标签
                    entity2_type = (labels[i, start2] - 1) // 2
                    
                    # 获取实体类型表示
                    entity1_type_repr = self.entity_type_embeddings(entity1_type)
                    entity2_type_repr = self.entity_type_embeddings(entity2_type)
                    
                    # 拼接实体表示和类型表示
                    pair_repr = torch.cat([entity1_repr, entity1_type_repr, 
                                        entity2_repr, entity2_type_repr])
                    
                    # LayerNorm
                    pair_repr = self.relation_layer_norm(pair_repr)
                    
                    # 计算关系分数
                    relation_logits[i, j] = self.relation_head(pair_repr)
                    
                except Exception as e:
                    debug_logger.error(f"处理实体对时出错: {str(e)}")
                    debug_logger.error(f"实体对信息: i={i}, j={j}, start1={start1}, end1={end1}, start2={start2}, end2={end2}")
                    debug_logger.error(f"序列长度: {seq_len}")
                    continue
        
        # 计算损失
        total_loss = None
        if labels is not None and relations is not None:
            # 更新类别计数
            if self.use_adaptive_weights and self.training:
                # 更新NER标签计数
                label_counts = torch.bincount(labels.view(-1)[labels.view(-1) != -1], 
                                           minlength=self.num_labels)
                self.ner_label_counts += label_counts.to(self.ner_label_counts.device)
                
                # 更新关系类别计数
                relation_counts = torch.bincount(relations.view(-1)[relations.view(-1) != -1], 
                                              minlength=self.num_relations)
                self.relation_counts += relation_counts.to(self.relation_counts.device)
            
            # 计算自适应权重
            if self.use_adaptive_weights:
                eps = 1e-8  # 防止除零
                ner_weights = 1.0 / (self.ner_label_counts + eps)
                ner_weights = ner_weights / ner_weights.sum() * self.num_labels  # 归一化
                
                relation_weights = 1.0 / (self.relation_counts + eps)
                relation_weights = relation_weights / relation_weights.sum() * self.num_relations
            else:
                ner_weights = None
                relation_weights = None
            
            # NER损失
            ner_loss_fct = CrossEntropyLoss(weight=ner_weights.to(ner_logits.device) if self.use_adaptive_weights else None)
            
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ner_logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(ner_loss_fct.ignore_index).type_as(labels)
                )
                
                loss = ner_loss_fct(active_logits, active_labels)
            else:
                loss = ner_loss_fct(ner_logits.view(-1, self.num_labels), labels.view(-1))
            
            # 关系分类损失
            relation_loss_fct = CrossEntropyLoss(
                weight=relation_weights.to(relation_logits.device) if self.use_adaptive_weights else None,
                ignore_index=-1
            )
            
            # 只计算有效的关系损失
            valid_relations = relations != -1
            if valid_relations.any():
                re_loss = relation_loss_fct(
                    relation_logits[valid_relations].view(-1, self.num_relations),
                    relations[valid_relations].view(-1)
                )
                
                loss = loss + re_loss
                total_loss = loss
        
        return RelationExtractionOutput(
            loss=total_loss,
            ner_logits=ner_logits,
            relation_logits=relation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RelationExtractionOutput:
    def __init__(self, loss, ner_logits, relation_logits, hidden_states, attentions):
        self.loss = loss
        self.ner_logits = ner_logits
        self.relation_logits = relation_logits
        self.hidden_states = hidden_states
        self.attentions = attentions
