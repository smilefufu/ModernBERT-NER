import json
import torch
from modelscope import AutoTokenizer
from transformers import TrainingArguments
from torch.utils.data import Dataset, DataLoader
from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.model_re import FlexBertForRelationExtraction
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import logging
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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
    config = FlexBertConfig(
        vocab_size=50368,
        hidden_size=1024,   
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="silu",  # ModernBERT 默认使用 silu
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=512,
        bert_layer="prenorm",
        attention_layer="base",
        embedding_layer="absolute_pos",
        encoder_layer="base",
        padding="padded",  
        use_fa2=False,  
        compile_model=False,
        normalization="layernorm",
        norm_kwargs={"eps": 1e-6},
        embed_norm=True,
        embed_dropout_prob=0.1,
        mlp_layer="mlp",
        mlp_dropout_prob=0.1,
        attn_qkv_bias=True,
        attn_out_bias=True,
        num_relations=53,
        entity_types=["疾病", "症状", "检查", "手术", "药物", "其他治疗", "部位", "社会学", "流行病学", "预后", "其他"],
    )
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    model = FlexBertForRelationExtraction(config)
    
    train_dataset = CMeIEDataset(
        "/Users/fufu/codes/playgruond/test-modernbert/workplace/data/CMeIE_train.json",
        tokenizer,
        "/Users/fufu/codes/playgruond/test-modernbert/workplace/data/53_schemas.jsonl"
    )
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=8e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    eval_dataset = CMeIEDataset(
        "/Users/fufu/codes/playgruond/test-modernbert/workplace/data/CMeIE_dev.json",
        tokenizer,
        "/Users/fufu/codes/playgruond/test-modernbert/workplace/data/53_schemas.jsonl"
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    
    num_training_steps = len(train_loader) * training_args.num_train_epochs
    num_warmup_steps = int(num_training_steps * training_args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    model = model.to(device)
    num_epochs = int(training_args.num_train_epochs)
    best_f1 = 0
    max_grad_norm = 1.0  
    
    logger.info("开始训练...")
    total_steps = len(train_loader) * num_epochs
    progress_bar = tqdm(total=total_steps, desc="训练进度")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
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
            if (batch_idx + 1) % 10 == 0:  # 每10个batch显示一次当前loss
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'epoch': f'{epoch+1}/{num_epochs}',
                    'loss': f'{avg_loss:.4f}'
                })
        
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
            torch.save(model.state_dict(), os.path.join(training_args.output_dir, 'best_model.pth'))
            logger.info(f'保存最佳模型，F1分数: {best_f1:.4f}')
    
    progress_bar.close()
    logger.info("训练完成！")

if __name__ == "__main__":
    main()