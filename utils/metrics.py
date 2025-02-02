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
    
    # 记录标签分布
    debug_logger.log_distribution("NER标签分布", label_flat, labels=entity_types)
    debug_logger.log_distribution("NER预测分布", pred_flat, labels=entity_types)
    
    # 计算各项指标
    precision, recall, f1, support = precision_recall_fscore_support(
        label_flat,
        pred_flat,
        average='macro',
        labels=list(range(len(entity_types)))  # 确保包含所有可能的标签
    )
    
    # 计算每个类别的指标
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        label_flat,
        pred_flat,
        average=None,
        labels=list(range(len(entity_types)))  # 确保包含所有可能的标签
    )
    
    # 计算混淆矩阵
    conf_mat = confusion_matrix(
        label_flat,
        pred_flat,
        labels=list(range(len(entity_types)))  # 确保包含所有可能的标签
    )
    
    # 记录混淆矩阵
    debug_logger.debug("\nNER混淆矩阵:")
    for i, row in enumerate(conf_mat):
        debug_logger.debug(f"  {entity_types[i]}: {row.tolist()}")
    
    # 整理每个实体类型的指标
    entity_metrics = {}
    for i, entity_type in enumerate(entity_types):
        entity_metrics[entity_type] = {
            'precision': class_precision[i],
            'recall': class_recall[i],
            'f1': class_f1[i]
        }
    
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
