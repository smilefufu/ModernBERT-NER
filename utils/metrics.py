import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from utils.debug_utils import debug_logger

def calculate_ner_metrics(predictions: List[List[int]], 
                        labels: List[List[int]], 
                        entity_types: List[str]) -> Dict:
    """计算NER的评估指标
    
    Args:
        predictions: List[List[int]] 预测的标签序列
        labels: List[List[int]] 真实的标签序列
        entity_types: List[str] 实体类型列表
        
    Returns:
        dict: {
            'precision': float 精确率
            'recall': float 召回率
            'f1': float F1分数
            'type_metrics': dict 每种实体类型的指标
        }
    """
    # 初始化计数器
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative
    
    # 每种实体类型的计数器
    type_metrics = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in entity_types}
    
    # 遍历每个序列
    for pred_seq, true_seq in zip(predictions, labels):
        # 找到预测的实体
        pred_entities = []
        start = None
        current_type = None
        for i, label in enumerate(pred_seq):
            if label > 0:  # 有实体标签
                if start is None:  # 新实体开始
                    start = i
                    current_type = label - 1
                elif label - 1 != current_type:  # 实体类型变化
                    pred_entities.append((start, i, current_type))
                    start = i
                    current_type = label - 1
            elif start is not None:  # 实体结束
                pred_entities.append((start, i, current_type))
                start = None
                current_type = None
        if start is not None:  # 处理序列末尾的实体
            pred_entities.append((start, len(pred_seq), current_type))
        
        # 找到真实的实体
        true_entities = []
        start = None
        current_type = None
        for i, label in enumerate(true_seq):
            if label > 0:  # 有实体标签
                if start is None:  # 新实体开始
                    start = i
                    current_type = label - 1
                elif label - 1 != current_type:  # 实体类型变化
                    true_entities.append((start, i, current_type))
                    start = i
                    current_type = label - 1
            elif start is not None:  # 实体结束
                true_entities.append((start, i, current_type))
                start = None
                current_type = None
        if start is not None:  # 处理序列末尾的实体
            true_entities.append((start, len(true_seq), current_type))
        
        # 计算指标
        for p_start, p_end, p_type in pred_entities:
            found_match = False
            for t_start, t_end, t_type in true_entities:
                if p_start == t_start and p_end == t_end and p_type == t_type:
                    tp += 1
                    type_metrics[entity_types[p_type]]['tp'] += 1
                    found_match = True
                    break
            if not found_match:
                fp += 1
                type_metrics[entity_types[p_type]]['fp'] += 1
        
        # 计算漏检的实体
        for t_start, t_end, t_type in true_entities:
            found_match = False
            for p_start, p_end, p_type in pred_entities:
                if p_start == t_start and p_end == t_end and p_type == t_type:
                    found_match = True
                    break
            if not found_match:
                fn += 1
                type_metrics[entity_types[t_type]]['fn'] += 1
    
    # 计算总体指标
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # 计算每种类型的指标
    for t in type_metrics:
        m = type_metrics[t]
        m['precision'] = m['tp'] / (m['tp'] + m['fp']) if m['tp'] + m['fp'] > 0 else 0
        m['recall'] = m['tp'] / (m['tp'] + m['fn']) if m['tp'] + m['fn'] > 0 else 0
        m['f1'] = 2 * m['precision'] * m['recall'] / (m['precision'] + m['recall']) if m['precision'] + m['recall'] > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'type_metrics': type_metrics
    }

def calculate_re_metrics(predictions: List[List[int]], 
                       labels: List[List[int]], 
                       schema: List[Dict]) -> Dict:
    """计算关系抽取的评估指标
    
    Args:
        predictions: List[List[int]] 预测的标签序列
        labels: List[List[int]] 真实的标签序列
        schema: List[Dict] 关系模式列表
        
    Returns:
        dict: {
            'precision': float 精确率
            'recall': float 召回率
            'f1': float F1分数
            'relation_metrics': dict 每种关系类型的指标
        }
    """
    # 初始化计数器
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative
    
    # 每种关系类型的计数器
    relation_metrics = {s['predicate']: {'tp': 0, 'fp': 0, 'fn': 0} for s in schema}
    
    # 遍历每个序列
    for pred_seq, true_seq in zip(predictions, labels):
        # 找到预测的关系
        pred_relations = []
        start = None
        current_relation = None
        for i, label in enumerate(pred_seq):
            if label > 0:  # 有关系标签
                if start is None:  # 新关系开始
                    start = i
                    current_relation = label - 1
                elif label - 1 != current_relation:  # 关系类型变化
                    pred_relations.append((start, i, current_relation))
                    start = i
                    current_relation = label - 1
            elif start is not None:  # 关系结束
                pred_relations.append((start, i, current_relation))
                start = None
                current_relation = None
        if start is not None:  # 处理序列末尾的关系
            pred_relations.append((start, len(pred_seq), current_relation))
        
        # 找到真实的关系
        true_relations = []
        start = None
        current_relation = None
        for i, label in enumerate(true_seq):
            if label > 0:  # 有关系标签
                if start is None:  # 新关系开始
                    start = i
                    current_relation = label - 1
                elif label - 1 != current_relation:  # 关系类型变化
                    true_relations.append((start, i, current_relation))
                    start = i
                    current_relation = label - 1
            elif start is not None:  # 关系结束
                true_relations.append((start, i, current_relation))
                start = None
                current_relation = None
        if start is not None:  # 处理序列末尾的关系
            true_relations.append((start, len(true_seq), current_relation))
        
        # 计算指标
        for p_start, p_end, p_relation in pred_relations:
            found_match = False
            for t_start, t_end, t_relation in true_relations:
                if p_start == t_start and p_end == t_end and p_relation == t_relation:
                    tp += 1
                    relation_metrics[schema[p_relation]['predicate']]['tp'] += 1
                    found_match = True
                    break
            if not found_match:
                fp += 1
                relation_metrics[schema[p_relation]['predicate']]['fp'] += 1
        
        # 计算漏检的关系
        for t_start, t_end, t_relation in true_relations:
            found_match = False
            for p_start, p_end, p_relation in pred_relations:
                if p_start == t_start and p_end == t_end and p_relation == t_relation:
                    found_match = True
                    break
            if not found_match:
                fn += 1
                relation_metrics[schema[t_relation]['predicate']]['fn'] += 1
    
    # 计算总体指标
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # 计算每种关系的指标
    for r in relation_metrics:
        m = relation_metrics[r]
        m['precision'] = m['tp'] / (m['tp'] + m['fp']) if m['tp'] + m['fp'] > 0 else 0
        m['recall'] = m['tp'] / (m['tp'] + m['fn']) if m['tp'] + m['fn'] > 0 else 0
        m['f1'] = 2 * m['precision'] * m['recall'] / (m['precision'] + m['recall']) if m['precision'] + m['recall'] > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'relation_metrics': relation_metrics
    }

def calculate_class_weights(data_loader, num_classes, device):
    """计算每个类别的权重
    
    在 BIO 标注方案中：
    - 0 表示 O 标签
    - 对于每个实体类型 i，2i+1 表示 B 标签，2i+2 表示 I 标签
    
    Args:
        data_loader: DataLoader 对象
        num_classes: 标签类别数量
        device: 计算设备
        
    Returns:
        torch.Tensor: 类别权重
    """
    import torch
    from tqdm import tqdm
    import logging
    
    logger = logging.getLogger(__name__)
    
    # 统计每个标签的样本数
    label_counts = torch.zeros(num_classes, device=device)
    total_samples = 0
    
    for batch in tqdm(data_loader, desc="计算类别权重"):
        labels = batch['labels']  # [batch_size, seq_len]
        # 统计每个标签的出现次数
        for i in range(num_classes):
            label_counts[i] += (labels == i).sum().item()
        total_samples += labels.numel()
    
    # 计算标签权重
    weights = torch.zeros(num_classes, device=device)
    
    # 处理 O 标签
    if label_counts[0] > 0:
        weights[0] = total_samples / (3 * label_counts[0])  # O 标签权重
    else:
        weights[0] = 1.0
    
    # 处理实体标签
    num_entity_types = (num_classes - 1) // 2
    for i in range(num_entity_types):
        b_idx = 2 * i + 1  # B 标签索引
        i_idx = 2 * i + 2  # I 标签索引
        
        # B 标签权重
        if label_counts[b_idx] > 0:
            weights[b_idx] = total_samples / (3 * num_entity_types * label_counts[b_idx])
        else:
            weights[b_idx] = 1.0
            
        # I 标签权重
        if label_counts[i_idx] > 0:
            weights[i_idx] = total_samples / (3 * num_entity_types * label_counts[i_idx])
        else:
            weights[i_idx] = 1.0
    
    # 归一化权重，使其平均值为1
    weights = weights * (num_classes / weights.sum())
    
    # 输出标签统计信息
    logger.info("\n标签分布统计:")
    logger.info(f"O标签 (0): {label_counts[0]:.0f} 样本, 权重: {weights[0]:.4f}")
    for i in range(num_entity_types):
        b_idx = 2 * i + 1
        i_idx = 2 * i + 2
        entity_type = data_loader.dataset.entity_types[i]
        logger.info(f"{entity_type}:")
        logger.info(f"  B标签 ({b_idx}): {label_counts[b_idx]:.0f} 样本, 权重: {weights[b_idx]:.4f}")
        logger.info(f"  I标签 ({i_idx}): {label_counts[i_idx]:.0f} 样本, 权重: {weights[i_idx]:.4f}")
    
    return weights

def format_metrics(metrics: Dict) -> str:
    """格式化指标输出
    
    Args:
        metrics: 指标字典
        
    Returns:
        格式化的字符串
    """
    output = []
    
    # 添加总体指标
    output.append(f"总体指标:")
    output.append(f"  Precision: {metrics['precision']:.4f}")
    output.append(f"  Recall: {metrics['recall']:.4f}")
    output.append(f"  F1: {metrics['f1']:.4f}")
    
    # 添加实体类型指标（如果存在）
    if 'type_metrics' in metrics:
        output.append("\n实体类型指标:")
        for entity_type, scores in metrics['type_metrics'].items():
            output.append(f"  {entity_type}:")
            output.append(f"    Precision: {scores['precision']:.4f}")
            output.append(f"    Recall: {scores['recall']:.4f}")
            output.append(f"    F1: {scores['f1']:.4f}")
    
    # 添加关系类型指标（如果存在）
    if 'relation_metrics' in metrics:
        output.append("\n关系类型指标:")
        for relation, scores in metrics['relation_metrics'].items():
            output.append(f"  {relation}:")
            output.append(f"    Precision: {scores['precision']:.4f}")
            output.append(f"    Recall: {scores['recall']:.4f}")
            output.append(f"    F1: {scores['f1']:.4f}")
    
    return "\n".join(output)

def compute_spo_metrics(
    predictions: List[List[Dict]], 
    labels: List[List[Dict]]
) -> Dict:
    """计算SPO三元组的评估指标
    
    Args:
        predictions: 预测的SPO三元组列表的列表
        labels: 真实的SPO三元组列表的列表
        
    Returns:
        包含precision, recall, f1等指标的字典
    """
    def normalize_spo(spo: Dict) -> Tuple:
        """将SPO三元组转换为可比较的格式"""
        return (
            spo.get('predicate', ''),
            spo.get('subject', {}).get('type', ''),
            spo.get('subject', {}).get('start', -1),
            spo.get('subject', {}).get('end', -1),
            spo.get('object', {}).get('type', ''),
            spo.get('object', {}).get('start', -1),
            spo.get('object', {}).get('end', -1)
        )
    
    # 统计正确的预测数、总预测数和总标签数
    correct = 0
    total_pred = 0
    total_gold = 0
    
    # 按样本统计指标
    for pred_spos, gold_spos in zip(predictions, labels):
        # 转换为可比较的格式
        pred_set = {normalize_spo(spo) for spo in pred_spos}
        gold_set = {normalize_spo(spo) for spo in gold_spos}
        
        # 统计正确的预测
        correct += len(pred_set & gold_set)
        total_pred += len(pred_set)
        total_gold += len(gold_set)
    
    # 计算精确率、召回率和F1
    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    # 记录详细指标
    logger = debug_logger
    logger.info("\nSPO三元组指标:")
    logger.info(f"  总体精确率: {precision:.4f}")
    logger.info(f"  总体召回率: {recall:.4f}")
    logger.info(f"  总体F1分数: {f1:.4f}")
    logger.info(f"  总预测数: {total_pred}")
    logger.info(f"  总标签数: {total_gold}")
    logger.info(f"  正确预测数: {correct}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_pred': total_pred,
        'total_gold': total_gold,
        'correct': correct
    }

def format_spo_metrics(metrics: Dict) -> str:
    """格式化SPO指标输出
    
    Args:
        metrics: 指标字典
        
    Returns:
        格式化的字符串
    """
    lines = []
    
    # 添加总体指标
    lines.append("总体指标:")
    lines.append(f"  Precision: {metrics['precision']:.4f}")
    lines.append(f"  Recall: {metrics['recall']:.4f}")
    lines.append(f"  F1: {metrics['f1']:.4f}")
    
    # 添加关系指标
    if 'relation_metrics' in metrics:
        lines.append("\n关系抽取指标:")
        for rel_name, rel_metrics in metrics['relation_metrics'].items():
            lines.append(f"{rel_name}:")
            lines.append(f"  Precision: {rel_metrics['precision']:.4f}")
            lines.append(f"  Recall: {rel_metrics['recall']:.4f}")
            lines.append(f"  F1: {rel_metrics['f1']:.4f}")
            lines.append(f"  Support: {rel_metrics['support']}")
    
    # 添加实体指标
    if 'entity_metrics' in metrics:
        lines.append("\n实体识别指标:")
        for type_name, type_metrics in metrics['entity_metrics'].items():
            lines.append(f"{type_name}:")
            lines.append(f"  Precision: {type_metrics['precision']:.4f}")
            lines.append(f"  Recall: {type_metrics['recall']:.4f}")
            lines.append(f"  F1: {type_metrics['f1']:.4f}")
            lines.append(f"  Support: {type_metrics['support']}")
    
    return "\n".join(lines)
