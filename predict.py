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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info("提取实体和关系...")
    
    # 获取NER标签
    ner_logits = outputs.ner_logits[0]  # [seq_len, num_labels]
    ner_labels = torch.argmax(ner_logits, dim=-1)  # [seq_len]
    
    # 提取实体span
    entities = []
    current_entity = None
    
    for i, (label, (start, end)) in enumerate(zip(ner_labels, offset_mapping)):
        if label == 0:  # O
            if current_entity is not None:
                # 找到实体的token范围
                entity_end = current_entity['start_idx'] + len(current_entity['text'])
                token_start = None
                token_end = None
                for j, (tok_start, tok_end) in enumerate(offset_mapping):
                    if tok_start <= current_entity['start_idx'] < tok_end:
                        token_start = j
                    if tok_start < entity_end <= tok_end:
                        token_end = j
                        break
                
                if token_start is not None and token_end is not None:
                    current_entity['token_start'] = token_start
                    current_entity['token_end'] = token_end
                    entities.append(current_entity)
                current_entity = None
        elif label == 1:  # B
            # 处理之前的实体
            if current_entity is not None:
                entity_end = current_entity['start_idx'] + len(current_entity['text'])
                token_start = None
                token_end = None
                for j, (tok_start, tok_end) in enumerate(offset_mapping):
                    if tok_start <= current_entity['start_idx'] < tok_end:
                        token_start = j
                    if tok_start < entity_end <= tok_end:
                        token_end = j
                        break
                
                if token_start is not None and token_end is not None:
                    current_entity['token_start'] = token_start
                    current_entity['token_end'] = token_end
                    entities.append(current_entity)
            
            # 开始新实体
            current_entity = {
                'start_idx': int(start),
                'text': text[start:end]
            }
        elif label == 2:  # I
            # 只有当前实体存在时才处理 I 标签
            if current_entity is not None:
                # 检查当前token是否紧接着上一个token
                prev_end = current_entity['start_idx'] + len(current_entity['text'])
                if start == prev_end:
                    # 扩展当前实体
                    current_entity['text'] = text[current_entity['start_idx']:end]
                else:
                    # 如果不连续，结束当前实体并忽略这个 I 标签
                    entity_end = current_entity['start_idx'] + len(current_entity['text'])
                    token_start = None
                    token_end = None
                    for j, (tok_start, tok_end) in enumerate(offset_mapping):
                        if tok_start <= current_entity['start_idx'] < tok_end:
                            token_start = j
                        if tok_start < entity_end <= tok_end:
                            token_end = j
                            break
                    
                    if token_start is not None and token_end is not None:
                        current_entity['token_start'] = token_start
                        current_entity['token_end'] = token_end
                        entities.append(current_entity)
                    current_entity = None
    
    # 处理最后一个实体
    if current_entity is not None:
        entity_end = current_entity['start_idx'] + len(current_entity['text'])
        token_start = None
        token_end = None
        for j, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start <= current_entity['start_idx'] < tok_end:
                token_start = j
            if tok_start < entity_end <= tok_end:
                token_end = j
                break
        
        if token_start is not None and token_end is not None:
            current_entity['token_start'] = token_start
            current_entity['token_end'] = token_end
            entities.append(current_entity)
    
    # 过滤掉空文本的实体和没有找到token范围的实体
    entities = [e for e in entities if e['text'].strip() and 'token_start' in e]
    
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
