import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from utils.debug_utils import debug_logger

def calculate_ner_metrics(predictions: List[np.ndarray], 
                        labels: List[np.ndarray], 
                        entity_types: List[str],
                        attention_mask: Optional[List[np.ndarray]] = None) -> Dict:
    """计算NER任务的评估指标
    
    Args:
        predictions: 预测的标签序列列表，每个元素是一个batch的预测
        labels: 真实的标签序列列表，每个元素是一个batch的标签
        entity_types: 实体类型列表
        attention_mask: 注意力掩码列表（可选）
        
    Returns:
        包含precision, recall, f1等指标的字典
    """
    # 展平预测和标签
    pred_flat = np.concatenate([p.flatten() for p in predictions])
    label_flat = np.concatenate([l.flatten() for l in labels])
    
    # 将BIO标签映射到实体类型
    def map_to_entity_type(label):
        if label == 0:  # O标签
            return -1  # 表示不是实体
        return (label - 1) // 2  # 每个实体类型有两个标签(B和I)
    
    # 转换预测和真实标签
    pred_types = np.array([map_to_entity_type(p) for p in pred_flat])
    label_types = np.array([map_to_entity_type(l) for l in label_flat])
    
    # 只考虑非O标签的位置
    mask = label_types != -1
    pred_types = pred_types[mask]
    label_types = label_types[mask]
    
    # 记录标签分布
    debug_logger.debug("\n标签分布统计:")
    debug_logger.debug("真实标签分布:")
    for i, entity_type in enumerate(entity_types):
        count = np.sum(label_types == i)
        debug_logger.debug(f"  {entity_type}: {count}")
    
    debug_logger.debug("\n预测标签分布:")
    for i, entity_type in enumerate(entity_types):
        count = np.sum(pred_types == i)
        debug_logger.debug(f"  {entity_type}: {count}")
    
    # 如果没有实体，返回全0指标
    if len(label_types) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'entity_metrics': {et: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} 
                             for et in entity_types}
        }
    
    # 计算各项指标
    precision, recall, f1, support = precision_recall_fscore_support(
        label_types,
        pred_types,
        average='macro',
        labels=list(range(len(entity_types)))  # 实体类型的索引
    )
    
    # 计算每个类别的指标
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        label_types,
        pred_types,
        average=None,
        labels=list(range(len(entity_types)))  # 实体类型的索引
    )
    
    # 计算混淆矩阵
    conf_mat = confusion_matrix(
        label_types,
        pred_types,
        labels=list(range(len(entity_types)))  # 实体类型的索引
    )
    
    # 记录混淆矩阵
    debug_logger.debug("\n实体类型混淆矩阵:")
    debug_logger.debug("预测 →")
    debug_logger.debug("实际 ↓")
    header = "     " + " ".join(f"{et:10}" for et in entity_types)
    debug_logger.debug(header)
    for i, row in enumerate(conf_mat):
        debug_logger.debug(f"{entity_types[i]:5}" + " ".join(f"{x:10d}" for x in row))
    
    # 整理每个实体类型的指标
    entity_metrics = {}
    for i, entity_type in enumerate(entity_types):
        entity_metrics[entity_type] = {
            'precision': class_precision[i],
            'recall': class_recall[i],
            'f1': class_f1[i]
        }
        debug_logger.debug(f"\n{entity_type}类型的指标:")
        debug_logger.debug(f"  Precision: {class_precision[i]:.4f}")
        debug_logger.debug(f"  Recall: {class_recall[i]:.4f}")
        debug_logger.debug(f"  F1: {class_f1[i]:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'entity_metrics': entity_metrics
    }

def calculate_re_metrics(predictions: List[np.ndarray],
                       labels: List[np.ndarray],
                       relations: List[str],
                       attention_mask: Optional[List[np.ndarray]] = None) -> Dict:
    """计算关系抽取任务的评估指标
    
    Args:
        predictions: 预测的关系标签序列列表，每个元素是一个batch的预测
        labels: 真实的关系标签序列列表，每个元素是一个batch的标签
        relations: 关系类型列表
        attention_mask: 注意力掩码列表（可选）
        
    Returns:
        包含precision, recall, f1等指标的字典
    """
    # 展平预测和标签
    pred_flat = np.concatenate([p.flatten() for p in predictions])
    label_flat = np.concatenate([l.flatten() for l in labels])
    
    # 记录标签分布
    debug_logger.log_distribution("关系标签分布", label_flat, labels=relations)
    debug_logger.log_distribution("关系预测分布", pred_flat, labels=relations)
    
    # 计算整体指标
    precision, recall, f1, support = precision_recall_fscore_support(
        label_flat,
        pred_flat,
        average='macro',
        labels=list(range(len(relations)))  # 确保包含所有可能的标签
    )
    
    # 计算每个类别的指标
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        label_flat,
        pred_flat,
        average=None,
        labels=list(range(len(relations)))  # 确保包含所有可能的标签
    )
    
    # 计算混淆矩阵
    conf_mat = confusion_matrix(
        label_flat,
        pred_flat,
        labels=list(range(len(relations)))  # 确保包含所有可能的标签
    )
    
    # 记录混淆矩阵
    debug_logger.debug("\n关系抽取混淆矩阵:")
    for i, row in enumerate(conf_mat):
        debug_logger.debug(f"  {relations[i]}: {row.tolist()}")
    
    # 整理每个关系类型的指标
    relation_metrics = {}
    for i, relation in enumerate(relations):
        relation_metrics[relation] = {
            'precision': class_precision[i],
            'recall': class_recall[i],
            'f1': class_f1[i]
        }
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'relation_metrics': relation_metrics
    }

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
    if 'entity_metrics' in metrics:
        output.append("\n实体类型指标:")
        for entity_type, scores in metrics['entity_metrics'].items():
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
