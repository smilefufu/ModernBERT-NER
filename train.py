import json
import torch
import yaml
import logging
import os
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoConfig
from modern_model_re import ModernBertForRelationExtraction
from tqdm import tqdm, trange
from utils.metrics import calculate_ner_metrics, calculate_re_metrics, format_metrics, compute_spo_metrics
from data.cmeie import CMeIEDataset, collate_fn
import math
import safetensors.torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
from typing import Union
from utils import plot_training_metrics

# 设置tokenizer并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_json_or_jsonl(file_path):
    """通用的json/jsonl加载函数
    
    Args:
        file_path: 文件路径
        
    Returns:
        list: 加载的数据列表
    """
    if file_path.endswith('.jsonl'):
        # jsonl格式，按行读取
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line.strip()))
        return data
    else:
        # json格式，直接加载
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_device():
    """设置设备，优先使用 CUDA，其次 MPS，最后 CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f'使用设备: {device}')
    return device

def get_optimizer(model: torch.nn.Module, learning_rate: Union[float, dict]) -> torch.optim.Optimizer:
    """获取优化器

    Args:
        model: 模型
        learning_rate: 学习率或学习率配置字典

    Returns:
        优化器
    """
    # 确保 learning_rate 是浮点数
    learning_rate = float(learning_rate)
    
    # 为不同参数设置不同的权重衰减
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    # 创建优化器
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    return optimizer

def load_schema(schema_file):
    """加载关系模式文件
    
    Args:
        schema_file: schema文件路径
        
    Returns:
        relations: 关系列表
    """
    relations = []
    with open(schema_file, 'r', encoding='utf-8') as f:
        for line in f:
            schema = json.loads(line.strip())
            if isinstance(schema, list):
                for item in schema:
                    relations.append(item['predicate'])
            else:
                relations.append(schema['predicate'])
    return list(set(relations))  # 去重

def initialize_training(config, device):
    """初始化训练所需的模型、优化器和调度器
    
    Args:
        config: 配置字典
        device: 计算设备
        
    Returns:
        model: 模型实例
        train_loader: 训练数据加载器
        eval_loader: 验证数据加载器
        optimizer: 优化器
    """
    # 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name_or_path'])
        logger.info(f"成功加载tokenizer: {type(tokenizer)}")
    except Exception as e:
        logger.error(f"加载tokenizer失败: {str(e)}")
        raise
    
    # 加载模型配置
    try:
        model_config = AutoConfig.from_pretrained(config['model']['model_name_or_path'])
        logger.info(f"成功加载模型配置: {type(model_config)}")
        
        # 添加 model_name_or_path 属性
        model_config.model_name_or_path = config['model']['model_name_or_path']
        
        # 从 schema 中获取实体类型
        temp_dataset = CMeIEDataset(
            data_file=config['data']['train_file'],
            tokenizer=tokenizer,
            schema_file=config['data']['schema_file'],
            max_length=config['data']['max_seq_length']
        )
        entity_types = temp_dataset.entity_types
        
        # 添加任务相关的配置
        model_config.num_labels = len(entity_types) * 2 + 1  # BIO标注方案
        model_config.num_entity_types = len(entity_types)
        model_config.num_spo_patterns = len(temp_dataset.spo2id)  # 使用数据集中的 SPO 关系模式数量
        
        logger.info(f"\n模型配置:")
        logger.info(f"  实体类型数量: {len(entity_types)}")
        logger.info(f"  标签数量: {model_config.num_labels} (BIO标注方案)")
        logger.info(f"  SPO关系模式数量: {model_config.num_spo_patterns}")
        logger.info(f"  SPO关系模式列表:")
        for spo_pattern, pattern_id in temp_dataset.spo2id.items():
            logger.info(f"    - {spo_pattern}: {pattern_id}")
    except Exception as e:
        logger.error(f"加载模型配置失败: {str(e)}")
        raise
    
    # 初始化模型
    try:
        model = ModernBertForRelationExtraction(model_config)
        model.to(device)
        logger.info(f"成功初始化模型: {type(model)}")
    except Exception as e:
        logger.error(f"初始化模型失败: {str(e)}")
        raise
    
    # 加载数据
    train_loader, eval_loader = load_data(config, tokenizer, device)
    
    # 初始化优化器
    learning_rate = config['training']['learning_rate']  # 必需参数，不使用默认值
    logger.info(f"使用学习率: {learning_rate}")
    optimizer = get_optimizer(model, learning_rate)
    
    return model, train_loader, eval_loader, optimizer

def load_data(config, tokenizer, device):
    """加载训练和验证数据
    
    Args:
        config: 配置字典
        tokenizer: tokenizer实例
        device: 设备
        
    Returns:
        train_loader: 训练数据加载器
        eval_loader: 验证数据加载器
    """
    # 加载完整数据集
    full_dataset = CMeIEDataset(
        data_file=config['data']['train_file'],
        tokenizer=tokenizer,
        schema_file=config['data']['schema_file'],
        max_length=config['data']['max_seq_length']
    )
    
    # 计算训练集和验证集大小
    total_size = len(full_dataset)
    eval_size = int(total_size * config['training']['eval_ratio'])
    train_size = total_size - eval_size
    
    # 划分训练集和验证集
    train_dataset, dev_dataset = random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(config['training']['random_seed'])
    )
    
    logger.info(f"数据集大小:")
    logger.info(f"  总数: {total_size}")
    logger.info(f"  训练集: {len(train_dataset)}")
    logger.info(f"  验证集: {len(dev_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    eval_loader = DataLoader(
        dev_dataset,
        batch_size=config['training']['eval_batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, eval_loader

def weighted_cross_entropy_loss(logits, labels, class_weights):
    """计算加权交叉熵损失
    
    Args:
        logits: 模型输出的logits，形状为 [batch_size, seq_len, num_classes]
        labels: 真实标签，形状为 [batch_size, seq_len]
        class_weights: 类别权重，形状为 [num_classes]
        
    Returns:
        加权交叉熵损失
    """
    # 将logits转换为log概率
    log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, num_classes]
    
    # 计算每个位置的损失
    nll_loss = F.nll_loss(
        log_probs.view(-1, log_probs.size(-1)),
        labels.view(-1),
        weight=class_weights,
        ignore_index=-100,
        reduction='none'
    )
    
    # 计算非填充位置的掩码
    mask = (labels != -100).float().view(-1)
    
    # 计算最终损失
    loss = (nll_loss * mask).sum() / (mask.sum() + 1e-8)
    
    return loss

def calculate_metrics(logits, labels):
    """计算评估指标
    
    Args:
        logits: 模型输出的 logits，形状为 (batch_size, seq_len, num_classes)
        labels: 真实标签，形状为 (batch_size, seq_len)
        
    Returns:
        dict: 包含 accuracy 和 f1 的字典
    """
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    
    # 计算 accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # 计算 macro f1
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, config):
    """训练一个epoch
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        epoch: 当前epoch
        config: 配置字典
        
    Returns:
        float: 平均训练损失
    """
    model.train()
    total_loss = 0
    accumulated_loss = 0
    
    # 创建进度条
    progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
    
    # 获取必需的配置参数
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    
    for batch_idx, batch in enumerate(progress_bar):
        # 将需要的tensor数据移动到设备上
        model_inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['labels'].to(device),
            'relation_labels': batch['relation_labels'].to(device),
            'spo_list': batch['spo_list']  # 保持原始Python列表
        }
        
        # 前向传播
        outputs = model(**model_inputs)
        loss = outputs['loss']
        
        # 如果使用梯度累积，需要对损失进行缩放
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # 记录未缩放的损失
        batch_loss = loss.item() * gradient_accumulation_steps
        total_loss += batch_loss
        accumulated_loss += batch_loss
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': batch_loss,
            'grad': torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item(),
            'lr': optimizer.param_groups[0]['lr'] if optimizer.param_groups else 'N/A'
        })
        
        # 梯度累积更新
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        # 定期记录训练状态
        if (batch_idx + 1) % config['training']['logging_steps'] == 0:
            avg_loss = accumulated_loss / config['training']['logging_steps']
            logger.info(
                f"Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)} - "
                f"平均损失: {avg_loss:.4f}, "
                f"梯度范数: {torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item():.4f}"
            )
            accumulated_loss = 0
    
    # 处理最后一个不完整的梯度累积周期
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    logger.info(
        f"Epoch {epoch} 训练完成 - "
        f"平均损失: {avg_loss:.4f}"
    )
    
    return avg_loss

def evaluate(model, data_loader, device, config):
    """评估模型性能
    
    Args:
        model: 模型实例
        data_loader: 评估数据加载器
        device: 计算设备
        config: 配置字典
        
    Returns:
        dict: {
            'ner_f1': float,      # NER的F1分数
            'relation_f1': float,  # 关系抽取的F1分数
            'loss': float         # 评估损失
        }
    """
    model.eval()
    total_loss = 0
    
    # 获取原始数据集
    # 处理 Subset 和其他可能的数据集代理类型
    if hasattr(data_loader.dataset, 'dataset'):
        # 对于 Subset 类型的数据集
        dataset = data_loader.dataset.dataset
    else:
        dataset = data_loader.dataset
    
    # 预测结果和真实标签
    pred_spo_list = []
    gold_spo_list = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            # 将需要的tensor数据移动到设备上
            model_inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device),
                'relation_labels': batch['relation_labels'].to(device),
                'spo_list': batch['spo_list']  # 保持原始Python列表
            }
            
            # 前向传播
            outputs = model(**model_inputs)
            loss = outputs['loss']
            total_loss += loss.item()
            
            # 收集SPO列表
            pred_spo_list.append(batch['spo_list'])
            gold_spo_list.append(batch['spo_list'])
    
    # 计算SPO指标
    from utils.metrics import compute_spo_metrics
    spo_metrics = compute_spo_metrics(pred_spo_list, gold_spo_list)
    
    # 返回符合文档要求的格式
    return {
        'ner_f1': spo_metrics['f1'],  # 使用总体F1分数
        'relation_f1': spo_metrics['f1'],
        'loss': total_loss / len(data_loader)
    }

def train_model(model, train_loader, eval_loader, optimizer, device, config):
    """训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        eval_loader: 评估数据加载器
        optimizer: 优化器
        device: 计算设备
        config: 配置字典
    """
    # 记录训练过程中的指标
    metrics_history = defaultdict(list)
    
    # 训练指定的epoch数
    try:
        num_epochs = config['training']['num_epochs']  # 必需参数，不使用默认值
    except KeyError:
        raise KeyError("配置文件中缺少必需的参数：training.num_epochs")
        
    for epoch in range(num_epochs):
        logger.info(f"\n开始 Epoch {epoch}/{num_epochs}")
        
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, None, device, epoch, config)
        metrics_history['train_loss'].append(train_loss)
        
        # 评估模型
        if eval_loader is not None:
            eval_metrics = evaluate(model, eval_loader, device, config)
            metrics_history['eval_loss'].append(eval_metrics['loss'])
            metrics_history['eval_f1'].append(eval_metrics['ner_f1'])
            
            logger.info(
                f"Epoch {epoch} 评估结果: "
                f"损失 = {eval_metrics['loss']:.4f}, "
                f"F1 = {eval_metrics['ner_f1']:.4f}"
            )
        
        # 记录当前的学习率
        if optimizer.param_groups:
            metrics_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 保存检查点
        if (epoch + 1) % config['output']['save_checkpoint_epochs'] == 0:
            checkpoint_dir = os.path.join(config['output']['output_dir'], 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'eval_metrics': eval_metrics if eval_loader is not None else None,
            }, checkpoint_path)
            logger.info(f"保存检查点到 {checkpoint_path}")
    
    # 绘制训练过程中的指标变化图
    if config['output']['plot_metrics']['enabled']:
        plot_training_metrics(metrics_history, config['output']['output_dir'])

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def calculate_class_weights(train_loader, num_spo_patterns, device):
    """
    计算每个 SPO 关系模式的权重
    Args:
        train_loader: 训练数据加载器
        num_spo_patterns: SPO 关系模式的数量
        device: 设备
    Returns:
        类别权重张量
    """
    logger.info(f"初始化关系类型权重计算，关系类型数量: {num_spo_patterns}")
    pattern_counts = torch.zeros(num_spo_patterns, dtype=torch.float32)
    total_patterns = 0
    
    # 统计每个模式的出现次数
    for batch in train_loader:
        spo_lists = batch['spo_list']  # 这是一个batch的SPO列表的列表
        for sample_spos in spo_lists:  # 遍历每个样本的SPO列表
            for spo in sample_spos:  # 遍历每个SPO三元组
                pattern_id = spo['spo_pattern']
                pattern_counts[pattern_id] += 1
                total_patterns += 1
    
    # 计算权重
    if total_patterns == 0:
        logger.warning("没有找到任何 SPO 关系模式！")
        return torch.ones(num_spo_patterns, device=device)
    
    # 使用逆频率作为权重
    weights = total_patterns / (pattern_counts + 1e-8)  # 添加小值避免除零
    
    # 将权重移动到指定设备
    weights = weights.to(device)
    
    # 输出每个模式的权重
    for pattern_id in range(num_spo_patterns):
        count = pattern_counts[pattern_id].item()
        weight = weights[pattern_id].item()
        logger.info(f"SPO模式 {pattern_id}: 出现次数 = {count}, 权重 = {weight:.4f}")
    
    return weights

def validate_config(config):
    """验证配置文件中的必需参数是否存在
    
    Args:
        config: 配置字典
        
    Raises:
        KeyError: 当缺少必需参数时抛出
    """
    required_params = {
        'training': [
            'num_epochs',
            'learning_rate',
            'gradient_accumulation_steps',
            'max_grad_norm',
            'use_weighted_loss',
            'batch_size',
            'eval_batch_size'
        ],
        'model': [
            'model_name_or_path'
        ],
        'data': [
            'train_file',
            'schema_file',
            'max_seq_length'
        ],
        'output': [
            'output_dir',
            'save_checkpoint_epochs'
        ]
    }
    
    missing_params = []
    for section, params in required_params.items():
        if section not in config:
            missing_params.append(f"缺少配置部分：{section}")
            continue
        for param in params:
            if param not in config[section]:
                missing_params.append(f"缺少参数：{section}.{param}")
    
    if missing_params:
        error_msg = "配置文件验证失败：\n" + "\n".join(missing_params)
        logger.error(error_msg)
        raise KeyError(error_msg)
    
    logger.info("配置文件验证通过")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 验证配置文件
    validate_config(config)
    
    # 设置随机种子
    set_seed(config['training'].get('random_seed', 42))
    
    # 设置设备
    device = set_device()
    
    # 初始化训练
    model, train_loader, eval_loader, optimizer = initialize_training(config, device)
    
    # 开始训练
    train_model(model, train_loader, eval_loader, optimizer, device, config)

if __name__ == "__main__":
    main()