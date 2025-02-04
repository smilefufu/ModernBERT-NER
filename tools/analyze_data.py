import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_or_jsonl(filename: str):
    """加载JSON或JSONL格式的文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        if filename.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

def analyze_data(data_file: str, schema_file: str):
    """分析训练数据和schema的一致性"""
    logger.info(f"开始分析数据文件: {data_file}")
    logger.info(f"使用schema文件: {schema_file}")
    
    # 加载数据
    data = load_json_or_jsonl(data_file)
    schema = load_json_or_jsonl(schema_file)
    
    # 从schema中收集实体类型和关系约束
    schema_entity_types = set()
    type_constraints = {}
    
    for item in schema:
        predicate = item['predicate']
        subject_type = item['subject_type']
        object_type = item['object_type']
        
        # 收集主实体类型
        if isinstance(subject_type, list):
            schema_entity_types.update(subject_type)
        else:
            schema_entity_types.add(subject_type)
        
        # 收集客实体类型
        if isinstance(object_type, dict) and '@value' in object_type:
            value_types = object_type['@value']
            if isinstance(value_types, list):
                schema_entity_types.update(value_types)
            else:
                schema_entity_types.add(value_types)
        elif isinstance(object_type, list):
            schema_entity_types.update(object_type)
        else:
            schema_entity_types.add(object_type)
            
        # 记录关系约束
        type_constraints[predicate] = (subject_type, object_type)
    
    logger.info(f"Schema中定义的实体类型: {sorted(list(schema_entity_types))}")
    
    # 分析训练数据
    entity_types_in_data = set()  # 训练数据中出现的所有实体类型
    entity_type_stats = defaultdict(int)  # 每种实体类型出现的次数
    entity_multiple_types = defaultdict(set)  # 记录每个实体被标注的所有类型
    entity_type_by_predicate = defaultdict(lambda: defaultdict(set))  # predicate -> entity -> types
    predicate_stats = defaultdict(int)  # 每种关系出现的次数
    type_mismatch_stats = defaultdict(int)  # 类型不匹配的统计
    
    # 收集schema中的关系约束
    predicate_constraints = {}
    for schema_item in schema:
        predicate = schema_item['predicate']
        subject_type = schema_item['subject_type']
        object_type = schema_item['object_type']
        if isinstance(object_type, dict) and '@value' in object_type:
            object_type = object_type['@value']
            
        # 对于同义词关系，主客体类型应该一致
        if predicate == '同义词':
            predicate_constraints[f"{predicate}_{subject_type}"] = (subject_type, subject_type)
        else:
            predicate_constraints[predicate] = (subject_type, object_type)
    
    for idx, item in enumerate(data):
        if 'spo_list' not in item:
            continue
            
        text = item['text']
        for spo in item['spo_list']:
            # 记录主实体信息
            subject = spo['subject']
            subject_type = spo['subject_type']
            subject_key = (subject, spo['subject_start_idx'])
            entity_types_in_data.add(subject_type)
            entity_type_stats[subject_type] += 1
            entity_multiple_types[subject_key].add(subject_type)
            
            # 记录客实体信息
            object_value = spo['object']['@value']
            object_type = spo['object_type']['@value']
            object_key = (object_value, spo['object_start_idx'])
            entity_types_in_data.add(object_type)
            entity_type_stats[object_type] += 1
            entity_multiple_types[object_key].add(object_type)
            
            # 记录关系信息
            predicate = spo['predicate']
            predicate_stats[predicate] += 1
            
            # 记录每个谓词下实体的类型
            entity_type_by_predicate[predicate][subject_key].add(subject_type)
            entity_type_by_predicate[predicate][object_key].add(object_type)
            
            # 检查类型约束
            if predicate == '同义词':
                constraint_key = f"{predicate}_{subject_type}"
                if constraint_key in predicate_constraints:
                    expected_subject_type, expected_object_type = predicate_constraints[constraint_key]
                    
                    # 检查主实体类型
                    if subject_type != expected_subject_type:
                        type_mismatch_stats[f"在关系'{predicate}'中，主实体'{subject}'的类型错误: 期望 {expected_subject_type}，实际为 {subject_type}"] += 1
                    
                    # 检查客实体类型
                    if object_type != expected_object_type:
                        type_mismatch_stats[f"在关系'{predicate}'中，客实体'{object_value}'的类型错误: 期望 {expected_object_type}，实际为 {object_type}"] += 1
            elif predicate in predicate_constraints:
                expected_subject_type, expected_object_type = predicate_constraints[predicate]
                
                # 检查主实体类型
                if subject_type != expected_subject_type:
                    type_mismatch_stats[f"在关系'{predicate}'中，主实体'{subject}'的类型错误: 期望 {expected_subject_type}，实际为 {subject_type}"] += 1
                
                # 检查客实体类型
                if object_type != expected_object_type:
                    type_mismatch_stats[f"在关系'{predicate}'中，客实体'{object_value}'的类型错误: 期望 {expected_object_type}，实际为 {object_type}"] += 1
    
    # 输出分析结果
    logger.info("\n=== 数据分析结果 ===")
    logger.info(f"总样本数: {len(data)}")
    
    # 输出实体类型统计
    logger.info("\n实体类型统计:")
    for entity_type, count in sorted(entity_type_stats.items(), key=lambda x: x[1], reverse=True):
        valid_mark = "✓" if entity_type in schema_entity_types else "✗"
        logger.info(f"{entity_type}: {count} ({valid_mark})")
    
    # 输出关系统计
    logger.info("\n关系统计:")
    for predicate, count in sorted(predicate_stats.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{predicate}: {count}")
    
    # 输出多重类型实体信息
    logger.info("\n实体多重类型情况:")
    multiple_type_count = sum(1 for types in entity_multiple_types.values() if len(types) > 1)
    logger.info(f"有 {multiple_type_count} 个实体在不同关系中扮演不同角色")
    for (entity, pos), types in entity_multiple_types.items():
        if len(types) > 1:
            # 收集这个实体在每个关系中的角色
            roles = []
            for pred, entities in entity_type_by_predicate.items():
                if (entity, pos) in entities:
                    roles.append(f"在关系'{pred}'中作为{'/'.join(entities[(entity, pos)])}")
            logger.info(f"实体 '{entity}' (位置 {pos}) 的角色: {sorted(list(types))}")
            logger.info(f"  详细信息: {'; '.join(roles)}")
    
    # 输出类型不匹配警告
    if type_mismatch_stats:
        logger.info("\n类型不匹配警告:")
        for mismatch, count in sorted(type_mismatch_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{mismatch}: {count}次")
    
    # 检查schema中未使用的类型
    unused_types = schema_entity_types - entity_types_in_data
    if unused_types:
        logger.warning(f"\nSchema中定义但未在数据中使用的类型: {sorted(list(unused_types))}")
    
    # 检查数据中未定义的类型
    undefined_types = entity_types_in_data - schema_entity_types
    if undefined_types:
        logger.warning(f"\n数据中使用但未在Schema中定义的类型: {sorted(list(undefined_types))}")

if __name__ == '__main__':
    # 假设我们在项目根目录运行这个脚本
    data_file = "data/CMeIE_train_with_idx.jsonl"  # 根据实际路径调整
    schema_file = "data/53_schemas.jsonl"  # 根据实际路径调整
    analyze_data(data_file, schema_file)
