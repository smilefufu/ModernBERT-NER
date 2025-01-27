import torch
import torch.nn as nn
import logging
from transformers import ModernBertPreTrainedModel, ModernBertModel
from torch.nn import CrossEntropyLoss

# 设置日志
logger = logging.getLogger(__name__)

class ModernBertForRelationExtraction(ModernBertPreTrainedModel):
    """
    基于 ModernBERT 的实体关系抽取模型
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    base_model_prefix = "model"
    
    def __init__(self, config):
        super().__init__(config)
        
        # 设置任务相关的参数
        self.num_labels = getattr(config, "num_labels", 3)  # BIO标注的默认类别数
        self.num_relations = getattr(config, "num_relations", 53)  # 默认关系类别数
        self.entity_types = getattr(config, "entity_types", [
            "疾病", "症状", "检查", "手术", "药物", "其他治疗", 
            "部位", "社会学", "流行病学", "预后", "其他"
        ])  # 默认实体类型
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
        
        # 记录输入形状
        logger.info(f"ModernBertForRelationExtraction forward - 输入形状:")
        logger.info(f"input_ids shape: {input_ids.shape if input_ids is not None else None}")
        logger.info(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        logger.info(f"labels shape: {labels.shape if labels is not None else None}")
        logger.info(f"relations shape: {relations.shape if relations is not None else None}")
        logger.info(f"entity_spans shape: {entity_spans.shape if entity_spans is not None else None}")
        
        # 获取 BERT 输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        sequence_output = outputs.last_hidden_state
        logger.info(f"sequence_output shape: {sequence_output.shape}")
        
        # 检查数值稳定性
        if torch.isnan(sequence_output).any():
            logger.error("检测到 NaN 在 sequence_output 中")
            raise ValueError("sequence_output contains NaN values")
        
        # 实体识别
        ner_logits = self.ner_head(sequence_output)
        logger.info(f"ner_logits shape: {ner_logits.shape}")
        
        if torch.isnan(ner_logits).any():
            logger.error("检测到 NaN 在 ner_logits 中")
            raise ValueError("ner_logits contains NaN values")
        
        # 关系分类
        batch_size = sequence_output.size(0)
        max_relations = entity_spans.size(1)  # 每个样本的最大关系数
        relation_logits = torch.zeros(batch_size, max_relations, self.num_relations, device=sequence_output.device)
        
        # 获取每个样本的有效序列长度
        sequence_lengths = attention_mask.sum(dim=1)  # [batch_size]
        
        # 统计有效和无效的实体对
        total_spans = 0
        valid_spans = 0
        
        for i in range(batch_size):
            # 检查是否有有效的实体对（所有位置都是0表示无效）
            has_valid_spans = not (entity_spans[i] == 0).all()
            logger.info(f"Batch {i} has_valid_spans: {has_valid_spans}")
            
            if has_valid_spans:
                seq_len = sequence_lengths[i].item()
                # 获取实体对的表示
                for j in range(max_relations):
                    start1, end1, start2, end2 = entity_spans[i, j]
                    total_spans += 1
                    
                    # 跳过无效的实体对（全0）
                    if start1 == 0 and end1 == 0 and start2 == 0 and end2 == 0:
                        continue
                    
                    # 如果发现 511，可能是填充标记，跳过
                    if 511 in [start1, end1, start2, end2]:
                        logger.warning(f"检测到填充标记 511: batch {i}, relation {j}, " + 
                                     f"spans: ({start1}, {end1}, {start2}, {end2})")
                        continue
                    
                    # 确保索引有效且在序列长度范围内
                    if (start1 >= end1 or end1 > seq_len or start1 >= seq_len or
                        start2 >= end2 or end2 > seq_len or start2 >= seq_len):
                        logger.warning(f"无效的实体范围: batch {i}, relation {j}, " + 
                                     f"spans: ({start1}, {end1}, {start2}, {end2}), " +
                                     f"sequence_length: {seq_len}")
                        continue
                    
                    # 使用平均池化获取实体表示
                    entity1_repr = sequence_output[i, start1:end1].mean(dim=0)
                    entity2_repr = sequence_output[i, start2:end2].mean(dim=0)
                    
                    # 检查数值稳定性
                    if torch.isnan(entity1_repr).any() or torch.isnan(entity2_repr).any():
                        logger.warning(f"检测到 NaN 在实体表示中: batch {i}, relation {j}")
                        continue
                    
                    # 拼接实体对表示
                    pair_repr = torch.cat([entity1_repr, entity2_repr])
                    
                    # 计算关系类型概率
                    rel_logits = self.relation_head(pair_repr)
                    
                    # 检查数值稳定性
                    if torch.isnan(rel_logits).any():
                        logger.warning(f"检测到 NaN 在关系 logits 中: batch {i}, relation {j}")
                        continue
                    
                    relation_logits[i, j] = rel_logits
                    valid_spans += 1
        
        # 记录实体对的统计信息
        if total_spans > 0:
            logger.info(f"实体对统计 - 总数: {total_spans}, 有效: {valid_spans}, " +
                       f"有效率: {valid_spans/total_spans*100:.2f}%")
        
        # 计算损失
        total_loss = None
        if labels is not None and relations is not None:
            loss_fct = CrossEntropyLoss(reduction='mean', ignore_index=-100)
            
            # NER loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            # 检查标签的有效性
            if torch.any((active_labels >= self.num_labels) & (active_labels != -100)):
                logger.error(f"检测到无效的标签: {active_labels.max().item()} >= {self.num_labels}")
                raise ValueError("Invalid label values detected")
            
            ner_loss = loss_fct(active_logits, active_labels)
            logger.info(f"ner_loss: {ner_loss.item()}")
            
            # 关系 loss
            # 只考虑有效序列长度内的实体对
            valid_relations = torch.zeros_like(entity_spans[..., 0], dtype=torch.bool)
            for i in range(batch_size):
                seq_len = sequence_lengths[i].item()
                for j in range(max_relations):
                    start1, end1, start2, end2 = entity_spans[i, j]
                    # 跳过 511 标记
                    if 511 in [start1, end1, start2, end2]:
                        continue
                    if (start1 < seq_len and end1 <= seq_len and
                        start2 < seq_len and end2 <= seq_len and
                        start1 < end1 and start2 < end2):
                        valid_relations[i, j] = True
            
            valid_relation_logits = relation_logits[valid_relations]
            valid_relation_labels = relations[valid_relations]
            
            # 检查是否有足够的有效关系进行训练
            if valid_relation_logits.size(0) > 0:
                # 检查标签的有效性
                if torch.any((valid_relation_labels >= self.num_relations) & (valid_relation_labels != -100)):
                    logger.error(f"检测到无效的关系标签: {valid_relation_labels.max().item()} >= {self.num_relations}")
                    raise ValueError("Invalid relation label values detected")
                
                relation_loss = loss_fct(valid_relation_logits, valid_relation_labels)
                logger.info(f"relation_loss: {relation_loss.item()}")
                
                # 平衡两个损失
                total_loss = ner_loss + relation_loss
            else:
                logger.warning("没有有效的关系用于训练")
                total_loss = ner_loss
            
            logger.info(f"total_loss: {total_loss.item()}")
            
            # 最终检查损失值
            if torch.isnan(total_loss):
                logger.error("检测到 NaN 在总损失中")
                raise ValueError("Total loss is NaN")
        
        if not return_dict:
            output = (ner_logits, relation_logits)
            return ((total_loss,) + output) if total_loss is not None else output
        
        return {
            'loss': total_loss,
            'ner_logits': ner_logits,
            'relation_logits': relation_logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
