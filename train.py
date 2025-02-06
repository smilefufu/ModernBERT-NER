"""
ModernBERT 医学实体关系提取模型训练脚本
"""
import os
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.cmeie import CMeIEDataset, collate_fn
from modern_model_re import ModernBertForRelationExtraction
from utils.metrics import compute_spo_metrics

import yaml
import argparse
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config):
    """模型训练主函数"""
    # 设置随机种子
    np.random.seed(config['training']['random_seed'])
    torch.manual_seed(config['training']['random_seed'])
    
    # 设备配置：优先级为 CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("使用 CUDA 设备训练")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("使用 MPS 设备训练")
    else:
        device = torch.device('cpu')
        logger.warning("使用 CPU 训练，训练速度可能较慢")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name_or_path'])
    
    # 数据集加载
    full_dataset = CMeIEDataset(
        data_file=config['data']['train_file'],
        schema_file=config['data']['schema_file'],
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )
    
    # 划分训练集和验证集
    train_size = int(len(full_dataset) * (1 - config['training']['eval_ratio']))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(config['training']['random_seed'])
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 模型初始化
    model = ModernBertForRelationExtraction(
        config_path=config['model']['model_name_or_path'],
        num_spo_patterns=full_dataset.num_relation_types
    ).to(device)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config['training']['warmup_ratio'])),
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_f1 = 0
    for epoch in range(config['training']['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0

        # 使用进度条展示训练进度
        train_progress_bar = tqdm(
            train_loader, 
            desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]}', 
            unit='batch'
        )
        for batch in train_progress_bar:
            # 准备数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            spo_list = batch['spo_list']

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播和损失计算
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                spo_list=spo_list
            )
            loss = outputs['loss']

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # 更新进度条
            train_progress_bar.set_postfix({'loss': loss.item()})
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_predictions = []
        all_ground_truth = []
        
        # 使用进度条展示验证进度
        val_progress_bar = tqdm(
            val_loader, 
            desc=f'Validation', 
            unit='batch'
        )
        
        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                spo_list = batch['spo_list']
                
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    spo_list=spo_list
                )
                val_loss += outputs['loss'].item()
                
                # 预测 SPO 三元组
                predictions = model.predict_spo(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                
                all_predictions.extend(predictions)
                all_ground_truth.extend(spo_list)
        
        # 计算指标
        metrics = compute_spo_metrics(
            predictions=all_predictions, 
            ground_truth=all_ground_truth,
            id2spo=full_dataset.id2spo
        )
        
        # 打印训练日志
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
        logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        
        # 模型保存
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            os.makedirs(config['output']['output_dir'], exist_ok=True)
            save_path = os.path.join(config['output']['output_dir'], 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_f1
            }, save_path)
            logger.info(f"Model saved with best F1: {best_f1:.4f}")

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="ModernBERT SPO Extraction Training")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)

if __name__ == '__main__':
    main()