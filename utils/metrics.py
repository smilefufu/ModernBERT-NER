import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def calculate_ner_metrics(predictions: List[np.ndarray], 
                        labels: List[np.ndarray], 
                        entity_types: List[str]) -> Dict:
    """计算NER任务的评估指标
    
    Args:
        predictions: 预测的标签序列列表，每个元素是一个batch的预测
        labels: 真实的标签序列列表，每个元素是一个batch的标签
        entity_types: 实体类型列表
        
    Returns:
        包含precision, recall, f1等指标的字典
    """
    # 将预测和标签展平
    pred_flat = np.concatenate([p.flatten() for p in predictions])
    label_flat = np.concatenate([l.flatten() for l in labels])
    
    # 计算各项指标
    precision, recall, f1, support = precision_recall_fscore_support(
        label_flat, 
        pred_flat, 
        labels=list(range(len(entity_types))),  # 确保包含所有可能的标签
        average='weighted',
        zero_division=0  # 当某个类别没有预测样本时返回0
    )
    
    # 计算每个类别的指标
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        label_flat,
        pred_flat,
        labels=list(range(len(entity_types))),  # 确保包含所有可能的标签
        average=None,
        zero_division=0  # 当某个类别没有预测样本时返回0
    )
    
    # 计算混淆矩阵
    conf_mat = confusion_matrix(
        label_flat, 
        pred_flat,
        labels=list(range(len(entity_types)))  # 确保包含所有可能的标签
    )
    
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
        'confusion_matrix': conf_mat,
        'entity_metrics': entity_metrics
    }

def calculate_re_metrics(predictions: List[np.ndarray], 
                       labels: List[np.ndarray], 
                       relations: List[str]) -> Dict:
    """计算关系抽取任务的评估指标
    
    Args:
        predictions: 预测的关系标签序列列表，每个元素是一个batch的预测
        labels: 真实的关系标签序列列表，每个元素是一个batch的标签
        relations: 关系类型列表
        
    Returns:
        包含precision, recall, f1等指标的字典
    """
    # 将预测和标签展平
    pred_flat = np.concatenate([p.flatten() for p in predictions])
    label_flat = np.concatenate([l.flatten() for l in labels])
    
    # 计算整体指标
    precision, recall, f1, support = precision_recall_fscore_support(
        label_flat,
        pred_flat,
        labels=list(range(len(relations))),  # 确保包含所有可能的标签
        average='weighted',
        zero_division=0  # 当某个类别没有预测样本时返回0
    )
    
    # 计算每个关系类型的指标
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        label_flat,
        pred_flat,
        labels=list(range(len(relations))),  # 确保包含所有可能的标签
        average=None,
        zero_division=0  # 当某个类别没有预测样本时返回0
    )
    
    # 计算混淆矩阵
    conf_mat = confusion_matrix(
        label_flat, 
        pred_flat,
        labels=list(range(len(relations)))  # 确保包含所有可能的标签
    )
    
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
        'confusion_matrix': conf_mat,
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
