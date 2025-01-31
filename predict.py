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
    
    # 获取关系
    relation_logits = outputs.relation_logits[0]  # [num_entities, num_entities, num_relations]
    
    # 提取实体span
    entities = []
    current_entity = None
    
    for i, (label, (start, end)) in enumerate(zip(ner_labels, offset_mapping)):
        if label == 0:  # O
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
        elif label == 1:  # B
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {
                'start_idx': int(start),
                'end_idx': int(end),
                'start': start,
                'end': end,
                'text': text[start:end]
            }
        elif label == 2:  # I
            if current_entity is not None:
                current_entity['end_idx'] = int(end)
                current_entity['end'] = end
                current_entity['text'] = text[current_entity['start']:end]
    
    if current_entity is not None:
        entities.append(current_entity)
    
    # 过滤掉空文本的实体
    entities = [e for e in entities if e['text'].strip()]
    
    # 提取关系
    relations = []
    if len(entities) > 1:  # 至少需要两个实体才可能有关系
        relation_scores = torch.softmax(relation_logits[:len(entities), :len(entities)], dim=-1)
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i != j:
                    rel_type = torch.argmax(relation_scores[i, j]).item()
                    if rel_type > 0:  # 0表示无关系
                        relations.append({
                            'subject': entities[i]['text'],
                            'object': entities[j]['text'],
                            'relation_id': rel_type,
                            'score': relation_scores[i, j, rel_type].item()
                        })
    
    return entities, relations

def predict(text, model, tokenizer, device, max_length=512):
    """对输入文本进行推理"""
    logger.info(f"处理文本: {text}")
    
    # 预处理文本
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
        # 先不传入entity_spans，只获取NER结果
        outputs = model(**inputs)
        
    # 从输出中提取实体
    ner_logits = outputs.ner_logits[0]  # [seq_len, num_labels]
    ner_labels = torch.argmax(ner_logits, dim=-1)  # [seq_len]
    
    # 提取实体span
    entities = []
    current_entity = None
    
    for i, (label, (start, end)) in enumerate(zip(ner_labels.cpu(), offset_mapping)):
        if label == 0:  # O
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
        elif label == 1:  # B
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {
                'start_idx': i,
                'end_idx': i + 1,
                'start': start,
                'end': end,
                'text': text[start:end]
            }
        elif label == 2:  # I
            if current_entity is not None:
                current_entity['end_idx'] = i + 1
                current_entity['end'] = end
                current_entity['text'] = text[current_entity['start']:end]
    
    if current_entity is not None:
        entities.append(current_entity)
    
    logger.info(f"识别出的实体: {entities}")
    
    # 过滤掉空文本的实体
    entities = [e for e in entities if e['text'].strip()]
    
    # 第二步：构建实体对并预测关系
    batch_size = inputs['input_ids'].size(0)
    max_relations = len(entities) * (len(entities) - 1) if len(entities) > 1 else 1
    entity_spans = torch.zeros((batch_size, max_relations, 4), dtype=torch.long, device=device)
    
    # 构建所有可能的实体对
    relation_idx = 0
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i != j:
                entity_spans[0, relation_idx] = torch.tensor([
                    entities[i]['start_idx'],
                    entities[i]['end_idx'],
                    entities[j]['start_idx'],
                    entities[j]['end_idx']
                ], device=device)
                relation_idx += 1
    
    logger.info(f"构建的实体对数量: {relation_idx}")
    
    # 第三步：预测关系
    with torch.no_grad():
        outputs = model(**inputs, entity_spans=entity_spans)
    
    # 解析结果
    relations = []
    if len(entities) > 1:
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
                            'object': entities[j]['text'],
                            'relation_id': rel_type,
                            'score': relation_scores[relation_idx, rel_type].item()
                        })
                    relation_idx += 1
    
    return {
        'text': text,
        'entities': entities,
        'relations': relations
    }

if __name__ == '__main__':
    # 示例用法
    model_path = '/Users/fufu/codes/playgruond/test-modernbert/workplace/outputs/model_epoch_1_f1_0.2377'  # 修改为你的模型路径
    text = "在2型糖尿病的治疗中，二甲双胍可以有效降低血糖水平。"  # 修改为你要分析的文本
    
    # 加载模型
    model, tokenizer, device = load_model(model_path, "mps")
    
    # 进行推理
    result = predict(text, model, tokenizer, device)

    print(result)
    
    # # 打印结果
    # print("\n输入文本:", result['text'])
    # print("\n识别的实体:")
    # for entity in result['entities']:
    #     print(f"- {entity['text']} ({entity['start']}:{entity['end']})")
    
    # print("\n识别的关系:")
    # for relation in result['relations']:
    #     print(f"- {relation['subject']} -> {relation['relation_id']} -> {relation['object']} (score: {relation['score']:.3f})")
