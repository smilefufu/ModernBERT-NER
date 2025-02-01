"""
ModernBERT 医学实体关系提取模型推理代码
"""
import os
import torch
import yaml
import logging
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
    logger.info("模型实例创建完成，检查初始权重...")
    check_model_weights(model, "初始化后")
    
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
    
    # 检查加载后的权重
    logger.info("权重加载完成，检查最终权重...")
    check_model_weights(model, "加载权重后")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def extract_entities_and_relations(outputs, text, tokenizer, offset_mapping):
    """从模型输出中提取实体和关系"""
    logger.debug("\n" + "="*50)
    logger.debug("开始处理新文本")
    logger.debug("="*50)
    logger.debug(f"原始文本: {text}")
    
    # 获取NER标签
    ner_logits = outputs.ner_logits[0]  # [seq_len, num_labels]
    ner_labels = torch.argmax(ner_logits, dim=-1)  # [seq_len]
    
    # 提取实体span
    entities = []
    current_entity = None
    last_end = -1
    
    for i, (label, (start, end)) in enumerate(zip(ner_labels, offset_mapping)):
        # 跳过特殊token
        if start == end:
            continue
            
        if label == 1:  # B
            # 如果有正在处理的实体，先保存它
            if current_entity is not None:
                entities.append(current_entity)
            
            # 开始新实体
            current_entity = {
                'start_idx': int(start),
                'text': text[start:end],
                'token_start': i,
                'token_end': i
            }
            last_end = end
        elif label == 2 and current_entity is not None:  # I
            # 只有当这个token紧接着上一个token时才合并
            if start == last_end:
                current_entity['text'] = text[current_entity['start_idx']:end]
                current_entity['token_end'] = i
                last_end = end
            else:
                # 如果不连续，保存当前实体并开始新实体
                entities.append(current_entity)
                current_entity = None
        elif label == 0:  # O
            # 保存当前实体
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    
    # 处理最后一个实体
    if current_entity is not None:
        entities.append(current_entity)
    
    # 过滤掉空文本的实体
    entities = [e for e in entities if e['text'].strip()]
    
    # 提取关系
    relations = []
    if len(entities) > 1:
        # 构建实体对
        batch_size = 1
        max_relations = len(entities) * (len(entities) - 1)
        entity_spans = torch.zeros((batch_size, max_relations, 4), dtype=torch.long)
        
        relation_idx = 0
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i != j:
                    entity_spans[0, relation_idx] = torch.tensor([
                        entities[i]['token_start'], entities[i]['token_end'],
                        entities[j]['token_start'], entities[j]['token_end']
                    ])
                    relation_idx += 1
        
        # 再次进行推理，这次带上实体spans
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                entity_spans=entity_spans.to(device)
            )
        
        # 提取关系
        relation_logits = outputs.relation_logits[0]  # [num_pairs, num_relations]
        relation_scores = torch.softmax(relation_logits[:relation_idx], dim=-1)
        
        relation_idx = 0
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i != j:
                    rel_type = torch.argmax(relation_scores[relation_idx]).item()
                    if rel_type > 0:  # 0表示无关系
                        relations.append({
                            'subject': entities[i]['text'],
                            'subject_start_idx': entities[i]['start_idx'],
                            'object': {
                                '@value': entities[j]['text']
                            },
                            'object_start_idx': entities[j]['start_idx'],
                            'relation_id': rel_type,
                            'score': relation_scores[relation_idx, rel_type].item()
                        })
                    relation_idx += 1
    
    logger.debug("\nToken 和标签对应关系:")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predictions = ner_labels.cpu().numpy()
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        logger.debug(f"位置 {i}: Token='{token}' (长度={len(token)}), 标签={pred}")
    
    logger.debug("\n实体合并过程:")
    current_entity = None
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        if pred == 0:  # O tag
            if current_entity is not None:
                logger.debug(f"\n完成实体: {current_entity}")
                entities.append(current_entity)
                current_entity = None
        elif pred == 1:  # B tag
            if current_entity is not None:
                logger.debug(f"\n完成实体: {current_entity}")
                entities.append(current_entity)
            start = i
            current_entity = {
                "start": start,
                "end": start + 1,
                "tokens": [token]
            }
            logger.debug(f"\n开始新实体: 位置={start}, 首个Token='{token}'")
        elif pred == 2:  # I tag
            if current_entity is not None and i == current_entity["end"]:
                current_entity["end"] = i + 1
                current_entity["tokens"].append(token)
                logger.debug(f"扩展实体: 添加Token='{token}', 当前tokens={current_entity['tokens']}")
            else:
                if current_entity is not None:
                    logger.debug(f"\n完成实体: {current_entity}")
                    entities.append(current_entity)
                start = i
                current_entity = {
                    "start": start,
                    "end": start + 1,
                    "tokens": [token]
                }
                logger.debug(f"\n开始新实体(I): 位置={start}, Token='{token}'")
    
    if current_entity is not None:
        logger.debug(f"\n完成最后一个实体: {current_entity}")
        entities.append(current_entity)
    
    logger.debug("\n最终识别的实体:")
    for e in entities:
        logger.debug(f"实体: '{e['text']}', 位置: {e['start_idx']}-{e['token_end']}")
    
    logger.debug("="*50 + "\n")
    return entities, relations

def predict(text, model, tokenizer, device, max_length=512):
    """对输入文本进行推理"""
    logger.info(f"处理文本: {text}")
    
    # 预处理文本
    global inputs  # 使其可以在 extract_entities_and_relations 中访问
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt',
        return_offsets_mapping=True
    )
    
    logger.debug(f"分词结果: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    offset_mapping = inputs.pop('offset_mapping')[0]
    
    # 将输入移到指定设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 第一步：识别实体
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 提取实体和关系
    entities, relations = extract_entities_and_relations(outputs, text, tokenizer, offset_mapping)
    
    logger.info(f"识别出 {len(entities)} 个实体:")
    for entity in entities:
        logger.info(f"  - {entity['text']} (位置: {entity['start_idx']})")
    
    logger.info(f"识别出 {len(relations)} 个关系:")
    for relation in relations:
        logger.info(f"  - {relation['subject']} -> {relation['object']['@value']} (得分: {relation['score']:.4f})")
    
    return entities, relations

if __name__ == '__main__':
    # 示例用法
    model_path = '/Users/fufu/codes/playgruond/test-modernbert/workplace/outputs/model_epoch_1_f1_0.2377'  # 修改为你的模型路径
    text = "在2型糖尿病的治疗中，二甲双胍可以有效降低血糖水平。"  # 修改为你要分析的文本
    
    # 加载模型
    model, tokenizer, device = load_model(model_path, "mps")
    
    # 进行推理
    result = predict(text, model, tokenizer, device)

    print(result)
