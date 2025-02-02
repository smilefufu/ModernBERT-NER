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
        
        # 加载类别权重(如果配置中提供)
        self.ner_weights = torch.tensor(config.ner_weights) if hasattr(config, "ner_weights") else None
        self.relation_weights = torch.tensor(config.relation_weights) if hasattr(config, "relation_weights") else None
        
        # ModernBERT 基础模型
        self.model = ModernBertModel(config)
        
        # 实体识别分类器
        self.ner_head = nn.Linear(config.hidden_size, self.num_labels)
        
        # 实体类型嵌入
        self.entity_type_embeddings = nn.Embedding(self.num_entity_types, config.hidden_size)
        
        # 实体表示相关组件
        self.entity_attention = nn.Linear(config.hidden_size, 1)
        
        # 关系分类相关组件
        self.relation_layer_norm = nn.LayerNorm(config.hidden_size * 4)  # 实体表示 + 类型表示
        self.relation_head = nn.Linear(config.hidden_size * 4, self.num_relations)
        
        # 初始化权重将在 init_weights 方法中完成
        
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
        
    def get_entity_embeddings(self, sequence_output, entity_spans):
        """获取实体表示
        Args:
            sequence_output: ModernBERT 输出，形状为 (batch_size, seq_len, hidden_size)
            entity_spans: 实体位置，形状为 (batch_size, max_relations, 4)
                        每个关系包含两个实体的起始和结束位置 [start1, end1, start2, end2]
        """
        batch_size = sequence_output.size(0)
        hidden_size = sequence_output.size(-1)
        max_relations = entity_spans.size(1)
        seq_len = sequence_output.size(1)
        
        # 初始化输出张量
        entity_embeddings = torch.zeros(
            (batch_size * max_relations * 2, hidden_size),
            dtype=sequence_output.dtype,
            device=sequence_output.device
        )
        
        valid_count = 0
        for i in range(batch_size):
            for j in range(max_relations):
                spans = entity_spans[i, j]
                start1, end1, start2, end2 = spans.tolist()
                
                # 处理第一个实体
                if start1 > 0 and end1 > 0 and start1 < seq_len and end1 < seq_len:
                    # 获取实体内部的token表示
                    entity_tokens = sequence_output[i, start1:end1+1]  # (entity_len, hidden_size)
                    
                    # 计算注意力分数并归一化
                    attention_scores = self.entity_attention(entity_tokens)  # (entity_len, 1)
                    attention_weights = torch.softmax(attention_scores, dim=0)  # (entity_len, 1)
                    
                    # 加权平均得到实体表示
                    entity_embeddings[valid_count] = (entity_tokens * attention_weights).sum(dim=0)  # (hidden_size,)
                
                valid_count += 1
                
                # 处理第二个实体
                if start2 > 0 and end2 > 0 and start2 < seq_len and end2 < seq_len:
                    # 获取实体内部的token表示
                    entity_tokens = sequence_output[i, start2:end2+1]  # (entity_len, hidden_size)
                    
                    # 计算注意力分数并归一化
                    attention_scores = self.entity_attention(entity_tokens)  # (entity_len, 1)
                    attention_weights = torch.softmax(attention_scores, dim=0)  # (entity_len, 1)
                    
                    # 加权平均得到实体表示
                    entity_embeddings[valid_count] = (entity_tokens * attention_weights).sum(dim=0)  # (hidden_size,)
                
                valid_count += 1
        
        return entity_embeddings

    def compute_relation_scores(self, entity_embeddings, entity_types=None):
        """计算关系得分
        Args:
            entity_embeddings: 实体表示，形状为 (batch_size * max_relations * 2, hidden_size)
            entity_types: 实体类型，形状为 (batch_size, max_relations, 2)，如果为None则使用默认类型0
        """
        batch_size = entity_embeddings.size(0) // 8  # 每个样本有4个关系，每个关系2个实体
        max_relations = 4
        hidden_size = entity_embeddings.size(1)
        
        # 重塑张量以配对实体
        # 从 (batch_size * max_relations * 2, hidden_size) 变为 (batch_size * max_relations, hidden_size * 2)
        entity_embeddings = entity_embeddings.view(batch_size, max_relations, 2, hidden_size)
        
        # 获取实体类型嵌入
        if entity_types is None:
            entity_types = torch.zeros(
                (batch_size, max_relations, 2),
                dtype=torch.long,
                device=entity_embeddings.device
            )
        entity_type_embeddings = self.entity_type_embeddings(entity_types)
        
        # 拼接实体表示和类型表示
        entity_pairs = torch.cat([
            entity_embeddings[:, :, 0, :],  # 第一个实体的表示
            entity_type_embeddings[:, :, 0, :],  # 第一个实体的类型表示
            entity_embeddings[:, :, 1, :],  # 第二个实体的表示
            entity_type_embeddings[:, :, 1, :]  # 第二个实体的类型表示
        ], dim=-1)  # 最终维度为 hidden_size * 4
        
        # 应用层归一化
        entity_pairs = self.relation_layer_norm(entity_pairs)
        
        # 展平为 (batch_size * max_relations, hidden_size * 4)
        entity_pairs = entity_pairs.view(-1, hidden_size * 4)
        
        # 计算关系得分
        relation_logits = self.relation_head(entity_pairs)
        
        # 重塑为 (batch_size, max_relations, num_relations)
        relation_logits = relation_logits.view(batch_size, max_relations, -1)
        
        return relation_logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        relations=None,
        entity_spans=None,
        entity_types=None,  # 新增参数
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取 ModernBERT 的输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        
        # NER 任务
        ner_logits = self.ner_head(sequence_output)
        
        # 获取实体表示
        entity_embeddings = self.get_entity_embeddings(sequence_output, entity_spans)
        
        # 计算关系得分，传入实体类型
        relation_logits = self.compute_relation_scores(entity_embeddings, entity_types)
        
        # 计算损失
        total_loss = None
        if labels is not None and relations is not None:
            # NER 损失
            loss_fct = CrossEntropyLoss(
                weight=self.ner_weights.to(labels.device) if self.ner_weights is not None else None,
                reduction='none'  # 先不要求平均，以便后续更精确地处理
            )
            
            # 计算每个位置的损失
            ner_loss_per_token = loss_fct(
                ner_logits.view(-1, self.num_labels),
                labels.view(-1)
            )
            
            # 只保留有效位置的损失（注意力掩码为1且不是特殊token的位置）
            valid_positions = (
                (attention_mask.view(-1) == 1) &  # 非填充位置
                (labels.view(-1) != -100)  # 非特殊token位置
            )
            ner_loss = ner_loss_per_token[valid_positions].mean()
            
            # 关系分类损失
            relation_loss_fct = CrossEntropyLoss(
                weight=self.relation_weights.to(relations.device) if self.relation_weights is not None else None,
                reduction='none'  # 先不要求平均，以便后续更精确地处理
            )
            
            # 计算每个关系的损失
            relation_loss_per_item = relation_loss_fct(
                relation_logits.view(-1, self.num_relations),
                relations.view(-1)
            )
            
            # 只保留有效关系的损失
            valid_relations = relations.view(-1) != -100
            relation_loss = relation_loss_per_item[valid_relations].mean()
            
            # 由于实体识别和关系分类是一个整体任务的两个方面
            # 我们使用简单的算术平均作为总损失
            # 这样可以保持两个子任务的梯度在相同的数量级
            total_loss = (ner_loss + relation_loss) / 2.0
        
        if not return_dict:
            output = (ner_logits, relation_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
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
