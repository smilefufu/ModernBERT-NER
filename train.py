import json
import torch
import yaml
import logging
import os
import argparse
from modelscope import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.model_re import FlexBertForRelationExtraction
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

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
    
    def load_data(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 添加数据验证
                valid_samples = [
                    sample for sample in data
                    if len(sample.get('spo_list', [])) > 0 and 'entities' in sample
                ]
                print(f"Found {len(valid_samples)} valid samples out of {len(data)}")
                return valid_samples
        except Exception as e:
            print(f"Error loading data from {filename}: {str(e)}")
            return []
    
    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)  # 确保 self.data 是一个列表并且不为 None
        
    def load_schema(self, filename):
        schemas = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    schema = json.loads(line)
                    schemas.append(f"{schema['subject_type']}_{schema['predicate']}_{schema['object_type']}")
        return schemas
    
    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return [
                sample for sample in json.load(f)
                if len(sample['spo_list']) > 0
            ]
    
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
        item = self.data[idx]
        text = item['text']
        
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
        bio_labels = self.create_bio_labels(text, item['entities'], self.max_length)
        
        # 准备关系标签
        relations = []
        entity_types = []
        entity_positions = []
        
        for spo in item['spo_list']:
            # 获取实体位置的token索引
            subj_start = spo['subject_start_idx']
            subj_end = spo['subject_end_idx']
            obj_start = spo['object_start_idx']
            obj_end = spo['object_end_idx']
            
            # 将字符级别的位置转换为token级别的位置
            subj_token_start = None
            subj_token_end = None
            obj_token_start = None
            obj_token_end = None
            
            offset_mapping = encoding['offset_mapping'][0].numpy()
            for idx, (start, end) in enumerate(offset_mapping):
                if start == subj_start:
                    subj_token_start = idx
                if end == subj_end:
                    subj_token_end = idx
                if start == obj_start:
                    obj_token_start = idx
                if end == obj_end:
                    obj_token_end = idx
            
            if all(x is not None for x in [subj_token_start, subj_token_end, obj_token_start, obj_token_end]):
                entity_positions.append((subj_token_start, obj_token_start))
                entity_types.append((spo['subject_type'], spo['object_type']['@value']))
                relation_type = f"{spo['subject_type']}_{spo['predicate']}_{spo['object_type']['@value']}"
                relations.append(self.relation2id[relation_type])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'bio_labels': torch.tensor(bio_labels),  # 现在长度已经固定
            'relations': torch.tensor(relations) if relations else torch.tensor([]),
            'entity_positions': torch.tensor(entity_positions) if entity_positions else torch.tensor([]),
            'entity_types': entity_types,
            'text': text,
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    bio_labels = torch.stack([item['bio_labels'] for item in batch])
    
    relations = []
    entity_positions = []
    entity_type_ids = []
    
    all_entity_types = set()
    for item in batch:
        if 'entity_types' in item and item['entity_types']:
            for subj_type, obj_type in item['entity_types']:
                all_entity_types.add(subj_type)
                all_entity_types.add(obj_type)
    entity_type_to_id = {etype: idx for idx, etype in enumerate(sorted(all_entity_types))}
    
    max_entities_per_sample = max(
        len(item['entity_types']) if 'entity_types' in item and item['entity_types'] else 0 
        for item in batch
    )
    
    for item in batch:
        if len(item['relations']) > 0:
            relations.append(item['relations'])
            entity_positions.append(item['entity_positions'])
            
            # 处理实体类型标签
            if 'entity_types' in item and item['entity_types']:
                current_types = []
                for subj_type, obj_type in item['entity_types']:
                    subj_type_id = entity_type_to_id[subj_type]
                    obj_type_id = entity_type_to_id[obj_type]
                    current_types.extend([subj_type_id, obj_type_id])
                
                # 填充到最大实体数
                if len(current_types) < max_entities_per_sample * 2:
                    padding = [-100] * (max_entities_per_sample * 2 - len(current_types))
                    current_types.extend(padding)
                
                entity_type_ids.append(torch.tensor(current_types))
    
    # 将收集到的数据转换为张量
    if relations:
        relations = torch.cat(relations)
        entity_positions = torch.cat(entity_positions)
        entity_type_ids = torch.stack(entity_type_ids)
    else:
        relations = torch.tensor([])
        entity_positions = torch.tensor([])
        entity_type_ids = torch.tensor([])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'bio_labels': bio_labels,
        'relations': relations,
        'entity_positions': entity_positions,
        'entity_type_ids': entity_type_ids,
    }

def evaluate(model, data_loader, device):
    model.eval()
    all_bio_preds = []
    all_bio_labels = []
    all_relation_preds = []
    all_relation_labels = []
    total_eval_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bio_labels = batch['bio_labels'].to(device)
            relations = batch['relations'].to(device) if len(batch['relations']) > 0 else None
            entity_positions = batch['entity_positions'].to(device) if len(batch['entity_positions']) > 0 else None
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_positions=entity_positions,
                labels=(bio_labels, None, relations)
            )
            
            loss = outputs[0]
            total_eval_loss += loss.item()
            
            bio_logits = outputs[1]
            # 确保bio_logits的batch_size与bio_labels匹配
            if bio_logits.size(0) != bio_labels.size(0):
                bio_logits = bio_logits.expand(bio_labels.size(0), -1, -1)
            
            bio_preds = torch.argmax(bio_logits, dim=-1)
            
            # 使用布尔索引前先展平张量
            bio_labels_flat = bio_labels.reshape(-1)
            bio_preds_flat = bio_preds.reshape(-1)
            mask = bio_labels_flat != -100
            
            all_bio_preds.extend(bio_preds_flat[mask].cpu().numpy())
            all_bio_labels.extend(bio_labels_flat[mask].cpu().numpy())
            
            if relations is not None and len(outputs) > 3:
                relation_logits = outputs[3]
                
                # 确保relation_logits的batch_size与relations匹配
                if relation_logits.size(0) != relations.size(0):
                    relation_logits = relation_logits.expand(relations.size(0), -1)
                
                relation_preds = torch.argmax(relation_logits, dim=-1)
                
                # 使用布尔索引前先展平张量
                relations_flat = relations.reshape(-1)
                relation_preds_flat = relation_preds.reshape(-1)
                mask = relations_flat != -100
                
                all_relation_preds.extend(relation_preds_flat[mask].cpu().numpy())
                all_relation_labels.extend(relations_flat[mask].cpu().numpy())
    
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
    
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model'])
    
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
    
    # 模型配置
    model_config = FlexBertConfig.from_pretrained(config['model']['pretrained_model'])
    for key in [
        'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size',
        'hidden_activation', 'max_position_embeddings', 'norm_eps', 'norm_bias',
        'global_rope_theta', 'attention_bias', 'attention_dropout',
        'global_attn_every_n_layers', 'local_attention', 'local_rope_theta',
        'embedding_dropout', 'mlp_bias', 'mlp_dropout', 'classifier_pooling',
        'classifier_dropout', 'hidden_dropout_prob', 'attention_probs_dropout_prob'
    ]:
        if key in config['model']:
            setattr(model_config, key, config['model'][key])
    
    # 初始化模型
    model = FlexBertForRelationExtraction.from_pretrained(
        config['model']['pretrained_model'],
        config=model_config
    )
    model = model.to(device)
    
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
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        bio_labels = batch['bio_labels'].to(device)
                        relations = batch['relations'].to(device) if len(batch['relations']) > 0 else None
                        entity_positions = batch['entity_positions'].to(device) if len(batch['entity_positions']) > 0 else None
                        
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            entity_positions=entity_positions,
                            labels=(bio_labels, None, relations)
                        )
                        
                        loss = outputs[0]
                        total_loss += loss.item()
                        
                        loss.backward()
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
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