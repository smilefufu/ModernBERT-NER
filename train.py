import json
import torch
import yaml
import logging
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from src.bert_layers.modern_model_re import ModernBertForRelationExtraction
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

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

def set_device(config):
    """设置设备"""
    if torch.backends.mps.is_available() and config['device']['use_mps']:
        device = torch.device('mps')
    elif torch.cuda.is_available() and config['device']['use_cuda']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f'使用设备: {device}')
    return device

def get_optimizer(model, config):
    """获取优化器"""
    # 设置不同的学习率
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config['training']['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=float(config['training']['learning_rate']),
        eps=1e-8
    )

class CMeIEDataset(Dataset):
    """CMeIE数据集"""
    def __init__(self, data_file, tokenizer, schema_file, max_length):
        # 支持传入文件路径或直接传入数据列表
        if isinstance(data_file, str):
            raw_data = load_json_or_jsonl(data_file)
        else:
            raw_data = data_file
        
        self.data = [sample for sample in raw_data if self.validate_sample(sample)]
        logger.info(f"数据集初始化 - 加载了 {len(self.data)}/{len(raw_data)} 个有效样本")
        
        # 加载schema
        self.schema = self.load_schema(schema_file)
        self.relation2id = {item: idx for idx, item in enumerate(self.schema)}
        
        # 初始化tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                model_max_length=max_length,
                use_fast=True,
                do_lower_case=False,
                strip_accents=False,
                tokenize_chinese_chars=True,
                encoding='utf-8'
            )
        else:
            self.tokenizer = tokenizer
            
        self.max_length = max_length
        
    def load_schema(self, filename):
        """加载schema文件，返回所有predicate列表"""
        raw_schema = load_json_or_jsonl(filename)
        if isinstance(raw_schema, list):
            # 如果直接是predicate列表
            if isinstance(raw_schema[0], str):
                return raw_schema
            # 如果是字典列表，提取predicate字段
            return [item['predicate'] for item in raw_schema]
        # 如果是字典，提取predicate字段
        return [item['predicate'] for item in raw_schema.values()]

    def __len__(self):
        return len(self.data)
        
    def validate_sample(self, sample):
        """验证样本格式"""
        required_fields = ['text', 'spo_list']
        if not all(field in sample for field in required_fields):
            return False
            
        for spo in sample['spo_list']:
            if not all(field in spo for field in ['subject', 'predicate', 'object']):
                return False
                
        return True
        
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.data[idx]
        text = sample['text']
        spo_list = sample['spo_list']
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,  # 添加offset mapping以便定位token
            return_tensors='pt'
        )
        
        # 去除batch维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        offset_mapping = encoding['offset_mapping']
        
        # 初始化标签
        seq_len = len(encoding['input_ids'])
        labels = torch.zeros(seq_len, dtype=torch.long)  # 用于NER标注
        relations = torch.zeros(4, dtype=torch.long)  # 假设每个样本最多4个关系
        entity_spans = torch.zeros((4, 4), dtype=torch.long)  # [关系数, (start1, end1, start2, end2)]
        
        # 处理每个SPO
        valid_spo_count = 0
        for spo in spo_list[:4]:  # 最多处理4个关系
            # 检查必要的字段
            if not all(k in spo for k in ['subject', 'predicate', 'object', 'subject_start_idx', 'subject_end_idx', 'object_start_idx', 'object_end_idx']):
                continue
                
            # 检查索引是否在文本范围内
            text_len = len(text)
            if any(idx >= text_len for idx in [spo['subject_start_idx'], spo['subject_end_idx'], spo['object_start_idx'], spo['object_end_idx']]):
                continue
            
            # 找到subject的token范围
            subj_start_token = None
            subj_end_token = None
            obj_start_token = None
            obj_end_token = None
            
            # 遍历offset mapping找到对应的token位置
            for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                if char_start == char_end:  # 跳过特殊token
                    continue
                # 主体
                if char_start <= spo['subject_start_idx'] <= char_end:
                    subj_start_token = token_idx
                if char_start <= spo['subject_end_idx'] <= char_end:
                    subj_end_token = token_idx
                    
                # 客体
                if char_start <= spo['object_start_idx'] <= char_end:
                    obj_start_token = token_idx
                if char_start <= spo['object_end_idx'] <= char_end:
                    obj_end_token = token_idx
            
            # 如果找到了所有位置且在有效范围内
            if all(x is not None for x in [subj_start_token, subj_end_token, obj_start_token, obj_end_token]):
                # 确保所有token索引都在序列长度范围内
                if max(subj_start_token, subj_end_token, obj_start_token, obj_end_token) >= seq_len:
                    continue
                
                # 设置spans和relation
                entity_spans[valid_spo_count] = torch.tensor([
                    subj_start_token, subj_end_token,
                    obj_start_token, obj_end_token
                ])
                relations[valid_spo_count] = self.relation2id[spo['predicate']]
                
                # 设置NER标签
                labels[subj_start_token:subj_end_token+1] = 1  # 主体
                labels[obj_start_token:obj_end_token+1] = 2  # 客体
                
                valid_spo_count += 1
                if valid_spo_count >= 4:
                    break
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels,
            'relations': relations,
            'entity_spans': entity_spans,
        }

def collate_fn(batch):
    """处理batch数据"""
    # 过滤掉 None 值
    batch = [item for item in batch if item is not None]
    
    # 如果整个 batch 都是 None，返回 None
    if not batch:
        return None
        
    # 基本输入
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # 处理实体位置和关系
    max_relations = max(len(item['relations']) for item in batch)
    batch_size = len(batch)
    
    # 初始化张量
    relations = torch.zeros((batch_size, max_relations), dtype=torch.long)
    entity_spans = torch.zeros((batch_size, max_relations, 4), dtype=torch.long)
    labels = torch.zeros((batch_size, input_ids.shape[1]), dtype=torch.long)
    
    # 填充数据
    for i, item in enumerate(batch):
        if len(item['relations']) > 0:
            # 处理relations
            relations[i, :len(item['relations'])] = item['relations']
            
            # 处理entity_spans
            spans = item['entity_spans']
            entity_spans[i, :len(item['relations'])] = spans
            
            # 处理labels
            labels[i] = item['labels']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'relations': relations,
        'entity_spans': entity_spans,
        'labels': labels,
    }

def load_data(config, tokenizer):
    """加载训练和验证数据"""
    # 加载训练数据
    train_file = config['data']['train_file']
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 划分训练集和验证集
    train_data, eval_data = train_test_split(
        train_data, 
        test_size=config['training'].get('eval_ratio', 0.1),  # 默认10%作为验证集
        random_state=config['training'].get('random_seed', 42)
    )
    
    logger.info(f"数据划分: 训练集 {len(train_data)} 样本, 验证集 {len(eval_data)} 样本")
    
    # 创建数据集
    train_dataset = CMeIEDataset(
        data_file=train_data, 
        tokenizer=tokenizer, 
        schema_file=config['data']['schema_file'], 
        max_length=config['data']['max_seq_length']
    )
    
    eval_dataset = CMeIEDataset(
        data_file=eval_data, 
        tokenizer=tokenizer, 
        schema_file=config['data']['schema_file'], 
        max_length=config['data']['max_seq_length']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config['training']['eval_batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, eval_loader

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    # 添加进度条
    train_iterator = tqdm(
        train_loader,
        desc=f"训练 Epoch {epoch}",
        total=len(train_loader),
        ncols=100
    )
    
    for batch_idx, batch in enumerate(train_iterator):
        if batch is None:
            continue
            
        # 将数据移动到设备上
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        outputs = model(**batch)
        loss = outputs['loss']
        
        if loss is None:
            continue
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        train_iterator.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    if len(train_loader) == 0:
        raise RuntimeError("没有有效的batch进行训练！")
        
    return total_loss / len(train_loader)

def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
                
            # 将数据移动到设备上
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs['loss']
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    config = load_config(args.config)
    
    # 设置设备
    device = set_device(config)
    
    # 设置tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['model_name_or_path'],
        model_max_length=config['data']['max_seq_length'],
        use_fast=True,
        do_lower_case=False,
        strip_accents=False,
        tokenize_chinese_chars=True,
        encoding='utf-8'
    )
    
    # 加载模型
    model_config = AutoConfig.from_pretrained(
        config['model']['model_name_or_path'],
        num_labels=config['model']['num_labels'],
        num_relations=config['model']['num_relations'],
        entity_types=config['model']['entity_types']
    )
    model = ModernBertForRelationExtraction.from_pretrained(
        config['model']['model_name_or_path'],
        config=model_config
    )
    model.to(device)
    
    # 加载数据
    train_loader, eval_loader = load_data(config, tokenizer)
    
    # 准备优化器和学习率调度器
    optimizer = get_optimizer(model, config)
    
    total_steps = len(train_loader) * config['training']['num_train_epochs']
    warmup_steps = int(total_steps * config['training']['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 开始训练
    logger.info("开始训练...")
    best_loss = float('inf')
    
    # 创建输出目录
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # 训练循环
    epochs = trange(config['training']['num_train_epochs'], desc="训练进度")
    for epoch in epochs:
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        logger.info(f'Epoch {epoch+1}/{config["training"]["num_train_epochs"]} - Train Loss: {train_loss:.4f}')
        
        # 评估
        dev_loss = evaluate(model, eval_loader, device)
        logger.info(f'Epoch {epoch+1}/{config["training"]["num_train_epochs"]} - Dev Loss: {dev_loss:.4f}')
        
        # 保存最佳模型
        if dev_loss < best_loss:
            best_loss = dev_loss
            # 保存模型和tokenizer
            model_path = os.path.join(config['output']['output_dir'], 'best_model')
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info(f'保存最佳模型到 {model_path}')
            
            # 限制保存的模型数量
            saved_models = sorted(
                [d for d in os.listdir(config['output']['output_dir']) if os.path.isdir(os.path.join(config['output']['output_dir'], d))],
                key=lambda x: os.path.getctime(os.path.join(config['output']['output_dir'], x))
            )
            while len(saved_models) > config['output']['save_total_limit']:
                oldest_model = saved_models.pop(0)
                oldest_path = os.path.join(config['output']['output_dir'], oldest_model)
                if os.path.exists(oldest_path):
                    import shutil
                    shutil.rmtree(oldest_path)
                    logger.info(f'删除旧模型: {oldest_path}')

if __name__ == "__main__":
    main()