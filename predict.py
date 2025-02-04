"""
ModernBERT 医学实体关系提取模型推理代码
"""
import os
import torch
import yaml
import logging
import json
from transformers import AutoTokenizer, AutoConfig
from modern_model_re import ModernBertForRelationExtraction

# 配置日志
logger = logging.getLogger('predict')
logger.setLevel(logging.DEBUG)

# 创建文件处理器
debug_handler = logging.FileHandler('debug.txt', mode='w', encoding='utf-8')
debug_handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter('%(message)s')
debug_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(debug_handler)

def check_model_weights(model, stage=""):
    """检查模型权重的状态"""
    logger.info(f"\n{stage}的模型权重统计:")
    
    for name, param in model.named_parameters():
        stats = {
            'shape': param.shape,
            'mean': param.float().mean().item() if param.numel() > 0 else None,
            'std': param.float().std().item() if param.numel() > 0 else None,
            'min': param.float().min().item() if param.numel() > 0 else None,
            'max': param.float().max().item() if param.numel() > 0 else None,
            'has_nan': torch.isnan(param).any().item(),
            'has_inf': torch.isinf(param).any().item()
        }
        
        logger.info(f"{name}:")
        for k, v in stats.items():
            if v is not None:
                logger.info(f"  {k}: {v}")
        
        if stats['has_nan']:
            logger.error(f"{name} 包含NaN值!")
        if stats['has_inf']:
            logger.error(f"{name} 包含Inf值!")
        if stats['std'] is not None and stats['std'] > 1e3:
            logger.warning(f"{name} 的标准差过大: {stats['std']}")

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载模型和分词器"""
    logger.info(f"开始加载模型，路径: {model_path}")
    logger.info(f"使用设备: {device}")
    
    # 加载配置
    config = AutoConfig.from_pretrained(model_path)
    
    # 创建模型实例
    model = ModernBertForRelationExtraction(config)
    # logger.info("模型实例创建完成，检查初始权重...")
    # check_model_weights(model, "初始化后")
    
    # 加载模型权重
    if os.path.isdir(model_path):
        # 如果是目录，查找权重文件
        for weight_file in ['pytorch_model.bin', 'model.pt', 'model.safetensors']:
            weight_path = os.path.join(model_path, weight_file)
            if os.path.exists(weight_path):
                logger.info(f"找到权重文件: {weight_path}")
                # 加载权重
                if weight_file.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_path)
                else:
                    state_dict = torch.load(weight_path, map_location='cpu')
                logger.info(f"权重文件包含 {len(state_dict)} 个参数")
                
                # 检查每个权重
                for key, value in state_dict.items():
                    if torch.isnan(value).any():
                        logger.error(f"权重文件中的参数 {key} 包含NaN值")
                    if torch.isinf(value).any():
                        logger.error(f"权重文件中的参数 {key} 包含Inf值")
                    if value.abs().max() > 1e6:
                        logger.warning(f"权重文件中的参数 {key} 包含过大的值: {value.abs().max()}")
                
                # 加载权重
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                logger.info(f"缺失的键: {missing_keys}")
                logger.info(f"意外的键: {unexpected_keys}")
                break
        else:
            raise ValueError(f"在 {model_path} 中找不到权重文件")
    else:
        raise ValueError(f"模型路径不存在: {model_path}")
    
    # # 检查加载后的权重
    # logger.info("权重加载完成，检查最终权重...")
    # check_model_weights(model, "加载权重后")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=config.max_position_embeddings,
        use_fast=True,
        do_lower_case=False,
        clean_up_tokenization_spaces=False,
        tokenize_chinese_chars=True,
        strip_accents=False
    )
    
    # 输出分词器配置信息
    logger.debug("\n分词器配置信息:")
    logger.debug(f"词表大小: {len(tokenizer)}")
    logger.debug(f"分词器类型: {type(tokenizer).__name__}")
    logger.debug(f"特殊token: {tokenizer.all_special_tokens}")
    logger.debug(f"特殊token IDs: {tokenizer.all_special_ids}")
    
    # 获取中文token示例
    chinese_tokens = []
    for token, _ in list(tokenizer.get_vocab().items())[:1000]:
        if '\u4e00' <= token <= '\u9fff':
            chinese_tokens.append(token)
        if len(chinese_tokens) >= 10:
            break
    logger.debug(f"词表示例(前10个中文token): {chinese_tokens}")
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def load_schema(schema_file):
    """从schema文件中加载实体类型和关系类型"""
    entity_types = set()
    with open(schema_file, 'r', encoding='utf-8') as f:
        for line in f:
            schema = json.loads(line.strip())
            entity_types.add(schema['subject_type'])
            if isinstance(schema['object_type'], dict):
                entity_types.add(schema['object_type']['@value'])
            else:
                entity_types.add(schema['object_type'])
    return sorted(list(entity_types))

def predict(text, model, tokenizer, device, max_length=512):
    """对输入文本进行推理"""
    model.eval()
    
    # 预处理文本
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    
    # 添加调试信息
    logger.debug("\n分词结果:")
    for i, (token_id, offset) in enumerate(zip(inputs["input_ids"][0], inputs["offset_mapping"][0])):
        token = tokenizer.convert_ids_to_tokens([token_id.item()])[0]
        start, end = offset.numpy()
        text_span = text[start:end] if start < end else f"[{token}]"
        logger.debug(f"位置 {i}: Token='{token}' (ID={token_id.item()}, 原文='{text_span}', 范围={start}:{end})")
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    offset_mapping = inputs["offset_mapping"][0].numpy()
    
    # 模型推理
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # 提取实体和关系
    entities, relations = extract_entities_and_relations(outputs, text, tokenizer, offset_mapping, input_ids)
    
    return {
        "text": text,
        "entities": entities,
        "relations": relations
    }

def extract_entities_and_relations(outputs, text, tokenizer, offset_mapping, input_ids):
    """从模型输出中提取实体和关系"""
    logger.debug("\n" + "="*50)
    logger.debug("开始处理新文本")
    logger.debug("="*50)
    logger.debug(f"原始文本: {text}")
    
    # 获取NER标签
    ner_logits = outputs.get('ner_logits')
    if ner_logits is not None:
        labels = torch.argmax(ner_logits, dim=-1)  # [seq_len]
         
        # 将预测结果转换为实体
        predictions = labels.cpu().numpy()
    
    logger.debug("\nToken 和标签对应关系:")
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        logger.debug(f"位置 {i}: Token='{token}' (长度={len(token)}), 标签={pred}")
    
    # 从schema中获取实体类型列表
    schema_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '53_schemas.jsonl')
    entity_types = load_schema(schema_file)
    
    entities = []
    current_entities = {}  # 每种类型的当前实体
    
    logger.debug("\n实体识别过程:")
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        if pred == 0:  # O tag
            # 结束所有当前实体
            for entity_type, current_entity in current_entities.items():
                if current_entity is not None:
                    logger.debug(f"完成{entity_type}实体: {current_entity}")
                    entities.append(current_entity)
            current_entities = {}
        else:
            # 计算实体类型
            type_idx = (pred - 1) // 2
            is_b_tag = (pred - 1) % 2 == 0
            entity_type = entity_types[type_idx]
            
            if is_b_tag:  # B tag
                # 如果该类型已有实体，先结束它
                if entity_type in current_entities and current_entities[entity_type] is not None:
                    logger.debug(f"完成{entity_type}实体: {current_entities[entity_type]}")
                    entities.append(current_entities[entity_type])
                
                # 获取当前token对应的原始文本范围
                token_start, token_end = offset_mapping[i]
                current_entities[entity_type] = {
                    "start_idx": int(token_start),
                    "end_idx": int(token_end),
                    "text": text[token_start:token_end],
                    "tokens": [token],
                    "type": entity_type
                }
                logger.debug(f"开始新{entity_type}实体: 位置={i}, Token='{token}', 文本范围={token_start}:{token_end}")
                
            else:  # I tag
                if entity_type in current_entities and current_entities[entity_type] is not None:
                    # 获取当前token对应的原始文本范围
                    token_start, token_end = offset_mapping[i]
                    current_entities[entity_type]["end_idx"] = int(token_end)
                    current_entities[entity_type]["tokens"].append(token)
                    current_entities[entity_type]["text"] = text[current_entities[entity_type]["start_idx"]:token_end]
                    logger.debug(f"扩展{entity_type}实体: 添加Token='{token}', 当前文本='{current_entities[entity_type]['text']}'")
                else:
                    # 如果遇到孤立的I标签，当作B标签处理
                    token_start, token_end = offset_mapping[i]
                    current_entities[entity_type] = {
                        "start_idx": int(token_start),
                        "end_idx": int(token_end),
                        "text": text[token_start:token_end],
                        "tokens": [token],
                        "type": entity_type
                    }
                    logger.debug(f"开始新{entity_type}实体(I): 位置={i}, Token='{token}', 文本范围={token_start}:{token_end}")
    
    # 处理最后的实体
    for entity_type, current_entity in current_entities.items():
        if current_entity is not None:
            logger.debug(f"完成最后一个{entity_type}实体: {current_entity}")
            entities.append(current_entity)
    
    # 过滤掉特殊token（[CLS], [SEP], [PAD]）产生的实体
    entities = [e for e in entities if not any(special in e["tokens"] for special in ["[CLS]", "[SEP]", "[PAD]"])]
    
    logger.debug("\n最终识别的实体:")
    for e in entities:
        logger.debug(f"{e['type']}实体: '{e['text']}', 位置: {e['start_idx']}-{e['end_idx']}")
    
    # 如果有关系预测结果，处理关系
    relations = []
    if outputs.relation_logits is not None:
        relation_logits = outputs.relation_logits[0]  # [num_relations, num_relations]
        relation_preds = torch.argmax(relation_logits, dim=-1)
        
        # 遍历所有实体对
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    relation_idx = relation_preds[i, j].item()
                    if relation_idx > 0:  # 0 表示无关系
                        relations.append({
                            "subject": entity1["text"],
                            "subject_type": entity1["type"],
                            "object": entity2["text"],
                            "object_type": entity2["type"],
                            "relation": relation_idx
                        })
    
    logger.debug("="*50 + "\n")
    return entities, relations

if __name__ == '__main__':
    # 示例用法
    model_path = './outputs/model_epoch_6_f1_0.2514'  # 修改为你的模型路径
    text = "休克中晚期：面色苍白、四肢厥冷、脉搏细弱、尿量减少，有严重缺氧和循环衰竭表现，如呼吸急促、唇周发绀、烦躁不安、意识障碍、动脉血氧分压降低、血氧饱和度下降、代谢性酸中毒；心率增快、四肢及皮肤湿冷、出现花纹，血压降低，收缩压可低于40mmHg以下，尿量减少或无尿，晚期可有DIC表现"  # 修改为你要分析的文本
    
    # 加载模型
    model, tokenizer, device = load_model(model_path, "mps")
    
    # 进行推理
    result = predict(text, model, tokenizer, device)

    print(result)
