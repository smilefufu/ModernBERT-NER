import json
import torch
import yaml
import logging
import os
import argparse
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
from src.bert_layers.modern_model_re import ModernBertForRelationExtraction
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import random

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="训练实体关系抽取模型")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    return parser.parse_args()

# 加载配置
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 设置设备
def set_device(config):
    if config['device']['use_cuda'] and torch.cuda.is_available():
        return torch.device("cuda")
    elif config['device']['use_mps'] and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class CMeIEDataset(Dataset):
    def __init__(self, data_file, tokenizer, schema_file, max_length=512):
        self.data = self.load_data(data_file)
        print(f"Loaded {len(self.data)} samples from {data_file}")  # 添加调试信息
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.schema = self.load_schema(schema_file)
        self.relation2id = {item: idx for idx, item in enumerate(self.schema)}
    
    def __len__(self):
        return len(self.data)
        
    def load_data(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                valid_samples = []
                
                for sample in data:
                    if not sample.get('spo_list'):
                        continue
                        
                    text = sample['text']
                    text_len = len(text)
                    valid_spos = []
                    
                    for spo in sample['spo_list']:
                        try:
                            subj_start = int(spo['subject_start_idx'])
                            subj_end = int(spo['subject_end_idx'])
                            obj_start = int(spo['object_start_idx'])
                            obj_end = int(spo['object_end_idx'])
                            
                            # 验证位置范围
                            # end_idx 是实体最后一个字符的下标
                            if not (0 <= subj_start <= subj_end < text_len and 
                                  0 <= obj_start <= obj_end < text_len):
                                logger.warning(f"实体位置超出范围: subject[{subj_start}:{subj_end}], object[{obj_start}:{obj_end}], text_len={text_len}")
                                continue
                                
                            # 验证实体文本
                            subject = text[subj_start:subj_end + 1]  # +1 是因为切片是左闭右开
                            object_ = text[obj_start:obj_end + 1]    # +1 是因为切片是左闭右开
                            
                            if not (subject.strip() and object_.strip()):
                                logger.warning(f"实体文本为空: subject='{subject}', object='{object_}'")
                                continue
                                
                            # 标准化object_type
                            if isinstance(spo['object_type'], dict):
                                spo['object_type'] = spo['object_type'].get('@value', '')
                                
                            valid_spos.append(spo)
                            
                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(f"实体位置无效: {str(e)}")
                            continue
                    
                    if valid_spos:
                        sample['spo_list'] = valid_spos
                        valid_samples.append(sample)
                
                logger.info(f"加载数据: {len(valid_samples)}/{len(data)} 个有效样本")
                return valid_samples
                
        except Exception as e:
            logger.error(f"加载数据出错 {filename}: {str(e)}")
            return []
    
    def load_schema(self, filename):
        schemas = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    schema = json.loads(line)
                    schemas.append(f"{schema['subject_type']}_{schema['predicate']}_{schema['object_type']}")
        return schemas
    
    def create_bio_labels(self, text, entities, max_length):
        # 初始化BIO标签
        bio_labels = ['O'] * len(text)
        
        for ent in entities:
            start_idx = ent['start_idx']
            end_idx = ent['end_idx']
            bio_labels[start_idx] = 'B'
            for i in range(start_idx + 1, end_idx):
                bio_labels[i] = 'I'
                
        # 转换为数字标签
        label_map = {'O': 0, 'B': 1, 'I': 2}
        labels = [label_map[label] for label in bio_labels]
        
        # 填充到最大长度
        if len(labels) < max_length:
            labels = labels + [0] * (max_length - len(labels))
        else:
            labels = labels[:max_length]
            
        return labels
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']
        text_length = len(text)
        
        # 确保每个样本至少有一个 SPO
        if not sample['spo_list']:
            logger.warning(f"样本 {idx} 没有 SPO")
            return None
            
        # 随机选择一个 SPO
        spo = random.choice(sample['spo_list'])
        
        # 获取实体位置
        subj_start = int(spo['subject_start_idx'])
        subj_end = int(spo['subject_end_idx'])
        obj_start = int(spo['object_start_idx'])
        obj_end = int(spo['object_end_idx'])
        
        # 构建关系类型
        relation_type = f"{spo['subject_type']}_{spo['predicate']}_{spo['object_type']}"
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # 准备BIO标签，使用相同的最大长度
        bio_labels = self.create_bio_labels(text, sample['entities'], self.max_length)
        
        # 准备关系标签
        relations = []
        entity_spans = []
        
        # 记录序列实际长度（不包括padding）
        seq_length = encoding['attention_mask'][0].sum().item()
        logger.info(f"\n处理样本 {idx}:")
        logger.info(f"文本长度: {len(text)}, Token长度: {seq_length}")
        
        for spo in sample['spo_list']:
            # 获取实体位置的token索引
            subj_start = spo.get('subject_start_idx')
            subj_end = spo.get('subject_end_idx')
            obj_start = spo.get('object_start_idx')
            obj_end = spo.get('object_end_idx')
            
            # 检查原始字符位置是否有效
            if any(x is None for x in [subj_start, subj_end, obj_start, obj_end]):
                logger.warning(f"跳过无效的实体位置: subject[{subj_start}:{subj_end}], object[{obj_start}:{obj_end}]")
                continue
                
            # 记录原始实体信息
            logger.info(f"\n关系: {spo.get('predicate')}")
            # if idx < 5:  # 只打印前5个样本的详细信息
            logger.info(f"主体实体: {text[subj_start:subj_end + 1]} [{subj_start}:{subj_end + 1}]")
            logger.info(f"客体实体: {text[obj_start:obj_end + 1]} [{obj_start}:{obj_end + 1}]")
            
            # 将字符级别的位置转换为token级别的位置
            subj_token_start = None
            subj_token_end = None
            obj_token_start = None
            obj_token_end = None
            
            offset_mapping = encoding['offset_mapping'][0].numpy()
            for i, (start, end) in enumerate(offset_mapping):
                if start == subj_start:
                    subj_token_start = i
                if end == subj_end:
                    subj_token_end = i + 1
                if start == obj_start:
                    obj_token_start = i
                if end == obj_end:
                    obj_token_end = i + 1
            
            # 检查token位置是否有效
            if any(x is None for x in [subj_token_start, subj_token_end, obj_token_start, obj_token_end]):
                logger.warning(f"无法找到对应的token位置: subject[{subj_token_start}:{subj_token_end}], object[{obj_token_start}:{obj_token_end}]")
                continue
                
            # 检查token位置是否在序列长度范围内
            if any(x >= seq_length for x in [subj_token_start, subj_token_end, obj_token_start, obj_token_end]):
                logger.warning(f"token位置超出序列长度 {seq_length}: subject[{subj_token_start}:{subj_token_end}], object[{obj_token_start}:{obj_token_end}]")
                continue
            
            # 记录最终的token位置
            logger.info(f"Token位置: subject[{subj_token_start}:{subj_token_end}], object[{obj_token_start}:{obj_token_end}]")
            
            entity_spans.append([subj_token_start, subj_token_end, obj_token_start, obj_token_end])
            relation_type = f"{spo['subject_type']}_{spo['predicate']}_{spo['object_type']}"
            relations.append(self.relation2id[relation_type])
        
        # 如果没有有效的关系，添加一个空的关系和位置
        if not relations:
            logger.info("该样本没有有效的关系，使用空标记")
            relations = [0]  # 使用0作为空关系的标记
            entity_spans = [[0, 0, 0, 0]]  # 使用0作为空位置的标记
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'bio_labels': torch.tensor(bio_labels),
            'relations': torch.tensor(relations),
            'entity_spans': torch.tensor(entity_spans),
            'text': text,
        }

def collate_fn(batch):
    # 基本输入
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    bio_labels = torch.stack([item['bio_labels'] for item in batch])
    
    # 处理实体位置和关系
    max_relations = max(len(item['relations']) for item in batch)
    batch_size = len(batch)
    
    # 初始化张量
    relations = torch.zeros((batch_size, max_relations), dtype=torch.long)
    entity_spans = torch.zeros((batch_size, max_relations, 4), dtype=torch.long)  # 改名为 entity_spans
    
    # 填充数据
    for i, item in enumerate(batch):
        if len(item['relations']) > 0:
            relations[i, :len(item['relations'])] = item['relations']
            entity_spans[i, :len(item['entity_spans'])] = item['entity_spans']  # 改名为 entity_spans
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': bio_labels,
        'relations': relations,
        'entity_spans': entity_spans,  # 改名为 entity_spans
    }

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            # 详细记录实体范围的原始值
            logger.info(f"\nBatch {batch_idx} - 详细数据:")
            for i in range(len(batch['entity_spans'])):
                # 记录当前样本的序列长度
                seq_len = batch['attention_mask'][i].sum().item()
                logger.info(f"\n样本 {i} - 序列长度: {seq_len}")
                logger.info("实体范围:")
                for j, spans in enumerate(batch['entity_spans'][i]):
                    if not (spans == 0).all():
                        logger.info(f"  关系 {j}: {spans.tolist()} - 标签: {batch['relations'][i][j].item()}")
            
            # 记录基本形状信息
            logger.info(f"\nBatch {batch_idx} - 输入形状:")
            logger.info(f"input_ids shape: {batch['input_ids'].shape}")
            logger.info(f"attention_mask shape: {batch['attention_mask'].shape}")
            logger.info(f"labels shape: {batch['labels'].shape}")
            logger.info(f"relations shape: {batch['relations'].shape}")
            logger.info(f"entity_spans shape: {batch['entity_spans'].shape}")
            
            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            relations = batch['relations'].to(device)
            entity_spans = batch['entity_spans'].to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                relations=relations,
                entity_spans=entity_spans,
            )
            
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            logger.info(f"Batch {batch_idx} - Loss: {loss.item()}")
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch} - Batch {batch_idx}/{num_batches} - Loss: {loss.item():.4f}")
                
        except Exception as e:
            logger.error(f"处理批次 {batch_idx} 时出错: {str(e)}")
            logger.error(f"错误详情:", exc_info=True)
            raise e
    
    return total_loss / num_batches

def evaluate(model, data_loader, device):
    model.eval()
    all_bio_preds = []
    all_bio_labels = []
    all_relation_preds = []
    all_relation_labels = []
    total_eval_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                # 记录每个批次的张量形状
                logger.info(f"Eval Batch {batch_idx} - 输入形状:")
                logger.info(f"input_ids shape: {batch['input_ids'].shape}")
                logger.info(f"attention_mask shape: {batch['attention_mask'].shape}")
                logger.info(f"labels shape: {batch['labels'].shape}")
                logger.info(f"relations shape: {batch['relations'].shape}")
                logger.info(f"entity_spans shape: {batch['entity_spans'].shape}")
                
                # 将数据移到设备
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                relations = batch['relations'].to(device)
                entity_spans = batch['entity_spans'].to(device)
                
                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    relations=relations,
                    entity_spans=entity_spans,
                )
                
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                ner_logits = outputs['ner_logits'] if isinstance(outputs, dict) else outputs[1]
                relation_logits = outputs['relation_logits'] if isinstance(outputs, dict) else outputs[2]
                
                # 计算预测
                ner_preds = torch.argmax(ner_logits, dim=-1)
                
                # 收集预测和标签
                active_mask = attention_mask.view(-1) == 1
                all_bio_preds.extend(ner_preds.view(-1)[active_mask].cpu().numpy())
                all_bio_labels.extend(labels.view(-1)[active_mask].cpu().numpy())
                
                # 计算关系预测
                for i in range(len(relation_logits)):
                    if len(relation_logits[i]) > 0:
                        rel_preds = torch.argmax(relation_logits[i], dim=-1)
                        all_relation_preds.extend(rel_preds.cpu().numpy())
                        all_relation_labels.extend(relations[i][:len(rel_preds)].cpu().numpy())
                
                total_eval_loss += loss.item()
                
                logger.info(f"Eval Batch {batch_idx} - Loss: {loss.item()}")
                
            except Exception as e:
                logger.error(f"评估批次 {batch_idx} 时出错: {str(e)}")
                logger.error(f"错误详情:", exc_info=True)
                raise e
    
    # 计算指标
    bio_metrics = precision_recall_fscore_support(
        all_bio_labels, all_bio_preds, average='macro'
    )
    
    relation_metrics = (0, 0, 0) if not all_relation_labels else \
        precision_recall_fscore_support(
            all_relation_labels, all_relation_preds, average='macro'
        )
    
    return {
        'eval_loss': total_eval_loss / len(data_loader),
        'bio_precision': bio_metrics[0],
        'bio_recall': bio_metrics[1],
        'bio_f1': bio_metrics[2],
        'relation_precision': relation_metrics[0],
        'relation_recall': relation_metrics[1],
        'relation_f1': relation_metrics[2]
    }

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = set_device(config)
    logger.info(f"使用设备: {device}")
    
    # 加载分词器和模型配置
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model'])
    model_config = AutoConfig.from_pretrained(
        config['model']['pretrained_model'],
        num_labels=3,  # BIO标注
        num_relations=53,  # 关系数量
        trust_remote_code=True,
    )
    
    # 初始化模型
    model = ModernBertForRelationExtraction.from_pretrained(
        config['model']['pretrained_model'],
        config=model_config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    
    model = model.to(device)
    
    # 数据集加载
    train_dataset = CMeIEDataset(
        data_file=config['data']['train_file'], 
        tokenizer=tokenizer, 
        schema_file=config['data']['schema_file'], 
        max_length=config['data']['max_seq_length']
    )
    
    eval_dataset = CMeIEDataset(
        data_file=config['data']['eval_file'], 
        tokenizer=tokenizer, 
        schema_file=config['data']['schema_file'], 
        max_length=config['data']['max_seq_length']
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config['training']['eval_batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * config['training']['num_train_epochs']
    warmup_steps = int(total_steps * config['training']['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # 训练参数
    num_epochs = config['training']['num_train_epochs']
    max_grad_norm = config['training']['max_grad_norm']
    
    # 输出目录
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练循环
    logger.info("开始训练...")
    best_f1 = 0
    total_steps = len(train_loader) * num_epochs
    progress_bar = tqdm(total=total_steps, desc="训练进度")
    
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            try:
                for batch_idx, batch in enumerate(train_loader):
                    try:
                        # 记录每个批次的张量形状
                        logger.info(f"Batch {batch_idx} - 输入形状:")
                        logger.info(f"input_ids shape: {batch['input_ids'].shape}")
                        logger.info(f"attention_mask shape: {batch['attention_mask'].shape}")
                        logger.info(f"labels shape: {batch['labels'].shape}")
                        logger.info(f"relations shape: {batch['relations'].shape}")
                        logger.info(f"entity_spans shape: {batch['entity_spans'].shape}")
                        
                        # 将数据移到设备
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        relations = batch['relations'].to(device)
                        entity_spans = batch['entity_spans'].to(device)
                        
                        # 清除梯度
                        optimizer.zero_grad()
                        
                        # 前向传播
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            relations=relations,
                            entity_spans=entity_spans,
                        )
                        
                        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                        logger.info(f"Batch {batch_idx} - Loss: {loss.item()}")
                        
                        # 反向传播
                        loss.backward()
                        
                        # 梯度裁剪
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        # 更新参数
                        optimizer.step()
                        scheduler.step()
                        
                        total_loss += loss.item()
                        
                        # 更新进度条
                        progress_bar.update(1)
                        if (batch_idx + 1) % config['training']['logging_steps'] == 0:
                            avg_loss = total_loss / (batch_idx + 1)
                            progress_bar.set_postfix({
                                'epoch': f'{epoch+1}/{num_epochs}',
                                'loss': f'{avg_loss:.4f}'
                            })
                    except Exception as batch_error:
                        logger.error(f"处理批次 {batch_idx} 时出错: {batch_error}")
                        continue
                
                # 每个epoch结束后的平均loss
                avg_epoch_loss = total_loss / len(train_loader)
                logger.info(f'Epoch {epoch+1}/{num_epochs} - 平均训练损失: {avg_epoch_loss:.4f}')
                
                # 评估
                eval_metrics = evaluate(model, eval_loader, device)
                logger.info(f'评估结果:')
                logger.info(f'验证损失: {eval_metrics["eval_loss"]:.4f}')
                logger.info(f'BIO标签 - 准确率: {eval_metrics["bio_precision"]:.4f}, '
                           f'召回率: {eval_metrics["bio_recall"]:.4f}, '
                           f'F1: {eval_metrics["bio_f1"]:.4f}')
                logger.info(f'关系分类 - 准确率: {eval_metrics["relation_precision"]:.4f}, '
                           f'召回率: {eval_metrics["relation_recall"]:.4f}, '
                           f'F1: {eval_metrics["relation_f1"]:.4f}')
                
                # 保存最佳模型
                if eval_metrics['bio_f1'] > best_f1:
                    best_f1 = eval_metrics['bio_f1']
                    model_path = os.path.join(output_dir, 'best_model.pth')
                    torch.save(model.state_dict(), model_path)
                    logger.info(f'保存最佳模型，F1分数: {best_f1:.4f}')
                    
                    # 限制保存的模型数量
                    saved_models = sorted(
                        [f for f in os.listdir(output_dir) if f.startswith('best_model')], 
                        key=lambda x: os.path.getctime(os.path.join(output_dir, x))
                    )
                    while len(saved_models) > config['output']['save_total_limit']:
                        os.remove(os.path.join(output_dir, saved_models.pop(0)))
            
            except RuntimeError as epoch_error:
                logger.error(f"训练轮次 {epoch+1} 出错: {epoch_error}")
                break
        
        progress_bar.close()
        logger.info("训练完成！")
    
    except Exception as main_error:
        logger.error(f"训练过程出现严重错误: {main_error}")
        raise

if __name__ == "__main__":
    main()