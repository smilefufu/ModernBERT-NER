#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict
from typing import Dict, Set, Tuple

def load_training_data(file_path: str) -> list:
    """加载训练数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_schema(file_path: str) -> list:
    """加载现有的schema定义"""
    schema = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            schema.append(json.loads(line.strip()))
    return schema

def analyze_training_data(data: list) -> Tuple[Set[str], Dict[Tuple[str, str, str], int]]:
    """分析训练数据中的实体类型和关系约束"""
    entity_types = set()
    relation_constraints = defaultdict(int)
    
    for item in data:
        for spo in item['spo_list']:
            # 处理主实体类型
            subject_type = spo['subject_type']
            entity_types.add(subject_type)
            
            # 处理客实体类型
            if isinstance(spo['object_type'], dict):
                object_type = spo['object_type']['@value']
            else:
                object_type = spo['object_type']
            entity_types.add(object_type)
            
            # 记录关系约束
            relation_key = (subject_type, spo['predicate'], object_type)
            relation_constraints[relation_key] += 1
    
    return entity_types, relation_constraints

def compare_schema(existing_schema: list, entity_types: Set[str], relation_constraints: Dict[Tuple[str, str, str], int]):
    """比较现有schema和训练数据中的差异"""
    print("\n=== Schema 比较结果 ===")
    
    # 从现有schema中提取信息
    existing_entity_types = set()
    existing_relations = set()
    for schema in existing_schema:
        existing_entity_types.add(schema['subject_type'])
        if isinstance(schema['object_type'], dict):
            existing_entity_types.add(schema['object_type']['@value'])
        else:
            existing_entity_types.add(schema['object_type'])
        existing_relations.add((schema['subject_type'], schema['predicate'], 
                              schema['object_type']['@value'] if isinstance(schema['object_type'], dict) else schema['object_type']))
    
    # 比较实体类型
    print("\n实体类型比较:")
    print(f"训练数据中的实体类型数量: {len(entity_types)}")
    print(f"现有Schema中的实体类型数量: {len(existing_entity_types)}")
    
    missing_types = entity_types - existing_entity_types
    if missing_types:
        print("\n在Schema中缺失的实体类型:")
        for t in sorted(missing_types):
            print(f"  - {t}")
    
    extra_types = existing_entity_types - entity_types
    if extra_types:
        print("\nSchema中多余的实体类型:")
        for t in sorted(extra_types):
            print(f"  - {t}")
    
    # 比较关系约束
    print("\n关系约束比较:")
    print(f"训练数据中的关系约束数量: {len(relation_constraints)}")
    print(f"现有Schema中的关系约束数量: {len(existing_relations)}")
    
    missing_relations = set(relation_constraints.keys()) - existing_relations
    if missing_relations:
        print("\n在Schema中缺失的关系约束:")
        for r in sorted(missing_relations):
            print(f"  - {r[0]} --[{r[1]}]--> {r[2]} (出现次数: {relation_constraints[r]})")
    
    extra_relations = existing_relations - set(relation_constraints.keys())
    if extra_relations:
        print("\nSchema中多余的关系约束:")
        for r in sorted(extra_relations):
            print(f"  - {r[0]} --[{r[1]}]--> {r[2]}")

def generate_new_schema(entity_types: Set[str], relation_constraints: Dict[Tuple[str, str, str], int], output_file: str):
    """生成新的schema文件"""
    new_schema = []
    for (subject_type, predicate, object_type), count in sorted(relation_constraints.items(), key=lambda x: (-x[1], x[0])):
        schema_item = {
            "subject_type": subject_type,
            "predicate": predicate,
            "object_type": object_type
        }
        new_schema.append(schema_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in new_schema:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n新的schema文件已生成: {output_file}")
    print(f"包含 {len(new_schema)} 个关系约束")

def main():
    # 设置文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    training_file = os.path.join(base_dir, 'data', 'CMeIE_train_with_idx.jsonl')
    schema_file = os.path.join(base_dir, 'data', '53_schemas.jsonl')
    new_schema_file = os.path.join(base_dir, 'data', 'new_schemas.jsonl')
    
    # 加载数据
    print("正在加载训练数据...")
    training_data = load_training_data(training_file)
    print(f"加载了 {len(training_data)} 条训练数据")
    
    print("\n正在加载现有schema...")
    existing_schema = load_schema(schema_file)
    print(f"加载了 {len(existing_schema)} 条schema定义")
    
    # 分析训练数据
    print("\n正在分析训练数据...")
    entity_types, relation_constraints = analyze_training_data(training_data)
    
    # 比较差异
    compare_schema(existing_schema, entity_types, relation_constraints)
    
    # 生成新的schema
    generate_new_schema(entity_types, relation_constraints, new_schema_file)

if __name__ == '__main__':
    main()
