import json
import torch
import yaml
import logging
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from modern_model_re import ModernBertForRelationExtraction
from tqdm import tqdm, trange
from utils.metrics import calculate_ner_metrics, calculate_re_metrics, format_metrics
from data.cmeie import CMeIEDataset, collate_fn
import math
import safetensors.torch

# 配置日志
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)

# 创建文件处理器
log_file = os.path.join(os.path.dirname(__file__), 'debug.txt')
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)

# 创建格式器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加处理器到日志记录器
debug_logger.addHandler(file_handler)

# 设置调试日志
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.WARNING)
debug_handler = logging.FileHandler('debug.txt', mode='w', encoding='utf-8')
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
debug_logger.addHandler(debug_handler)

def log_tensor_info(name, tensor, step=None):
    return
    """记录tensor的详细信息"""
    if tensor is None:
        debug_logger.debug(f"{name} is None")
        return
        
    info = {
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': tensor.device,
        'min': tensor.min().item() if tensor.numel() > 0 else None,
        'max': tensor.max().item() if tensor.numel() > 0 else None,
        'mean': tensor.float().mean().item() if tensor.numel() > 0 else None,
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item()
    }
    
    step_info = f" (step {step})" if step is not None else ""
    debug_logger.debug(f"{name}{step_info}:")
    for k, v in info.items():
        debug_logger.debug(f"  {k}: {v}")

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
    base_lr = float(config['training']['learning_rate'])
    
    # 为不同组件设置不同的学习率
    optimizer_grouped_parameters = [
        # ModernBERT embedding层
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'embeddings' in n and not any(nd in n for nd in no_decay)],
            'weight_decay': config['training']['weight_decay'],
            'lr': base_lr * 0.5  # embedding层使用较小的学习率
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'embeddings' in n and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': base_lr * 0.5
        },
        # ModernBERT encoder浅层(0-3)
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'encoder.layer' in n and int(n.split('.')[3]) < 4 
                      and not any(nd in n for nd in no_decay)],
            'weight_decay': config['training']['weight_decay'],
            'lr': base_lr * 0.75  # 浅层使用较小的学习率
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'encoder.layer' in n and int(n.split('.')[3]) < 4 
                      and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': base_lr * 0.75
        },
        # ModernBERT encoder深层(4+)
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'encoder.layer' in n and int(n.split('.')[3]) >= 4 
                      and not any(nd in n for nd in no_decay)],
            'weight_decay': config['training']['weight_decay'],
            'lr': base_lr  # 深层使用基准学习率
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'encoder.layer' in n and int(n.split('.')[3]) >= 4 
                      and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': base_lr
        },
        # NER和实体类型分类头
        {
            'params': [p for n, p in model.named_parameters() 
                      if ('ner_head' in n or 'entity_type_head' in n) 
                      and not any(nd in n for nd in no_decay)],
            'weight_decay': config['training']['weight_decay'],
            'lr': base_lr * 2.0  # 分类头使用较大的学习率
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if ('ner_head' in n or 'entity_type_head' in n) 
                      and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': base_lr * 2.0
        },
        # 关系分类头
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'relation_head' in n and not any(nd in n for nd in no_decay)],
            'weight_decay': config['training']['weight_decay'],
            'lr': base_lr * 1.5  # 关系分类头使用适中的学习率
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'relation_head' in n and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': base_lr * 1.5
        }
    ]
    
    # 记录每组参数的学习率
    debug_logger.info("\n优化器参数组配置:")
    for group in optimizer_grouped_parameters:
        param_names = [n for n, p in model.named_parameters() 
                      if any(p is param for param in group['params'])]
        debug_logger.info(f"学习率 {group['lr']:.2e}, 权重衰减 {group['weight_decay']:.2e}:")
        for name in param_names:
            debug_logger.info(f"  - {name}")
    
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=base_lr,  # 这个值会被每个参数组的lr覆盖
        eps=1e-8,
        betas=(0.9, 0.999)
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
    # 初始化数据集
    dataset = CMeIEDataset(
        data_file=config['data']['train_file'],
        tokenizer=tokenizer,
        schema_file=config['data']['schema_file'],
        max_length=config['data']['max_seq_length']
    )
    
    # 划分训练集和验证集
    train_size = int((1 - config['training']['eval_ratio']) * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(eval_dataset)}")
    
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

def initialize_training(config, device):
    """初始化训练所需的模型、优化器和调度器"""
    # 加载预训练配置
    debug_logger.info(f"开始加载预训练模型配置，路径: {config['model']['model_name_or_path']}")
    model_config = AutoConfig.from_pretrained(
        config['model']['model_name_or_path'],
        num_labels=config['model']['num_labels'],
        num_relations=config['model']['num_relations'],
        entity_types=config['model']['entity_types']
    )
    debug_logger.info("预训练模型配置加载完成")
    debug_logger.info(f"配置信息: {model_config}")
    
    # 检查预训练模型文件是否存在
    model_path = config['model']['model_name_or_path']
    if not os.path.exists(model_path):
        debug_logger.error(f"预训练模型路径不存在: {model_path}")
        raise FileNotFoundError(f"预训练模型路径不存在: {model_path}")
    
    # 检查模型文件
    debug_logger.info("检查预训练模型文件:")
    for file in os.listdir(model_path):
        debug_logger.info(f"  - {file}")
    
    # 加载 safetensors 格式的模型文件
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        debug_logger.error(f"找不到 model.safetensors: {safetensors_path}")
        raise FileNotFoundError(f"找不到 model.safetensors: {safetensors_path}")
    
    # 从预训练模型初始化
    debug_logger.info("开始加载预训练模型权重")
    try:
        # 加载预训练权重
        state_dict = safetensors.torch.load_file(safetensors_path)
        debug_logger.info(f"成功加载权重文件，包含以下键:")
        for key in state_dict.keys():
            debug_logger.info(f"  - {key}")
        
        # 创建模型实例
        model = ModernBertForRelationExtraction(model_config)
        debug_logger.info("创建了新的ModernBertForRelationExtraction实例")
        
        # 检查模型的键和预训练权重的键是否匹配
        model_state = model.state_dict()
        debug_logger.info(f"模型需要的键:")
        for key in model_state.keys():
            debug_logger.info(f"  - {key}")
        
        # 检查键的匹配情况
        EXPECTED_MISSING_KEYS = {
            'ner_head.weight',
            'ner_head.bias',
            'entity_type_head.weight',
            'entity_type_head.bias',
            'relation_layer_norm.weight',
            'relation_layer_norm.bias',
            'relation_head.weight',
            'relation_head.bias'
        }
        EXPECTED_UNUSED_KEYS = {
            'head.dense.weight',
            'head.norm.weight',
            'decoder.bias'
        }
        missing_keys = [k for k in model_state.keys() if k not in state_dict]
        unexpected_keys = [k for k in state_dict.keys() if k not in model_state]
        
        if missing_keys:
            unexpected_missing = set(missing_keys) - EXPECTED_MISSING_KEYS
            if unexpected_missing:
                debug_logger.warning(f"发现意外缺少的键:")
                for k in sorted(unexpected_missing):
                    debug_logger.warning(f"  - {k}")
            if debug_logger.isEnabledFor(logging.DEBUG):
                debug_logger.debug(f"预期缺少的键:")
                for k in sorted(set(missing_keys) & EXPECTED_MISSING_KEYS):
                    debug_logger.debug(f"  - {k}")
        
        if unexpected_keys:
            unexpected_unused = set(unexpected_keys) - EXPECTED_UNUSED_KEYS
            if unexpected_unused:
                debug_logger.warning(f"发现意外未使用的键:")
                for k in sorted(unexpected_unused):
                    debug_logger.warning(f"  - {k}")
            if debug_logger.isEnabledFor(logging.DEBUG):
                debug_logger.debug(f"预期未使用的键:")
                for k in sorted(set(unexpected_keys) & EXPECTED_UNUSED_KEYS):
                    debug_logger.debug(f"  - {k}")
        
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        debug_logger.info("预训练模型权重加载完成")
        
        # 检查关键层的权重状态
        debug_logger.info("检查关键层的权重状态:")
        key_layers = [
            'model.embeddings.word_embeddings',
            'model.encoder.layer.0',
            'ner_head',
            'entity_type_head',
            'relation_head'
        ]
        for name, param in model.named_parameters():
            if any(key in name for key in key_layers):
                stats = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'has_nan': torch.isnan(param.data).any().item(),
                    'has_inf': torch.isinf(param.data).any().item()
                }
                debug_logger.info(f"{name} 统计信息:")
                for k, v in stats.items():
                    debug_logger.info(f"  {k}: {v}")
        
    except Exception as e:
        debug_logger.error(f"加载预训练模型时出错: {str(e)}")
        debug_logger.error(f"错误类型: {type(e)}")
        debug_logger.error("错误堆栈:", exc_info=True)
        raise
    
    # 移动到指定设备
    model.to(device)
    debug_logger.info(f"模型已移动到设备: {device}")
    
    # 设置tokenizer
    debug_logger.info("开始加载tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['model_name_or_path'],
        model_max_length=config['data']['max_seq_length'],
        use_fast=True,
        do_lower_case=False,
        strip_accents=False,
        tokenize_chinese_chars=True,
        encoding='utf-8'
    )
    debug_logger.info("tokenizer加载完成")
    
    # 冻结预训练模型的参数（可选）
    if config['training'].get('freeze_backbone', False):
        debug_logger.info("冻结预训练模型参数")
        for param in model.model.parameters():
            param.requires_grad = False
    
    # 准备优化器
    optimizer = get_optimizer(model, config)
    
    # 学习率调度器
    total_steps = len(load_data(config, tokenizer)[0]) * config['training']['num_train_epochs']
    warmup_steps = int(total_steps * config['training']['warmup_ratio'])
    
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # 余弦衰减阶段
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    debug_logger.info("\n学习率调度器配置:")
    debug_logger.info(f"总训练步数: {total_steps}")
    debug_logger.info(f"预热步数: {warmup_steps}")
    debug_logger.info(f"预热比例: {config['training']['warmup_ratio']}")
    
    # 记录学习率曲线
    steps = list(range(total_steps))
    lrs = [lr_lambda(step) * config['training']['learning_rate'] for step in steps]
    debug_logger.info("\n学习率曲线采样:")
    for i, step in enumerate(range(0, total_steps, total_steps//10)):
        debug_logger.info(f"Step {step}: {lrs[step]:.2e}")
    
    return model, tokenizer, optimizer, scheduler

def calculate_class_weights(data_loader, num_classes, device):
    """计算每个类别的权重"""
    label_counts = torch.zeros(num_classes)
    
    for batch in data_loader:
        if batch is None or 'labels' not in batch:
            continue
        labels = batch['labels']
        for i in range(num_classes):
            label_counts[i] += (labels == i).sum().item()
    
    # 使用逆频率作为权重，并进行归一化
    total_samples = label_counts.sum()
    weights = total_samples / (label_counts + 1e-8)  # 添加小值避免除零
    weights = weights / weights.sum() * num_classes  # 归一化，使权重和等于类别数
    
    logger.info("\n类别权重:")
    for i in range(num_classes):
        logger.info(f"  类别 {i}: {weights[i]:.4f}")
    
    return weights.to(device)

def weighted_cross_entropy_loss(logits, labels, class_weights):
    """计算加权交叉熵损失"""
    # 将logits转换为log概率
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # 创建one-hot标签
    num_classes = logits.size(-1)
    one_hot = torch.zeros_like(log_probs).scatter_(-1, labels.unsqueeze(-1), 1)
    
    # 计算每个位置的损失
    per_position_loss = -(one_hot * log_probs)  # [batch_size, seq_len, num_classes]
    
    # 应用类别权重
    weighted_loss = per_position_loss * class_weights.view(1, 1, -1)  # 广播权重到所有位置
    
    # 只计算非填充位置的损失
    mask = (labels != -100).float()
    loss = (weighted_loss.sum(-1) * mask).sum() / (mask.sum() + 1e-8)
    
    return loss

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    # 如果使用加权损失，计算类别权重
    class_weights = None
    if config['training'].get('use_weighted_loss', False) and epoch == 0:
        class_weights = calculate_class_weights(train_loader, model.config.num_labels, device)
        # 确保权重和为num_classes
        class_weights = class_weights * (model.config.num_labels / class_weights.sum())
    
    # 获取梯度裁剪阈值
    max_grad_norm = config['training'].get('max_grad_norm', 10.0)
    
    # 创建进度条
    progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        debug_logger.debug(f"\n处理第{epoch}轮第{batch_idx}个batch")
        
        # 检查输入数据
        debug_logger.debug("输入数据统计:")
        debug_logger.debug(f"  input_ids形状: {batch['input_ids'].shape}")
        debug_logger.debug(f"  attention_mask形状: {batch['attention_mask'].shape}")
        
        # 将数据移动到设备上
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        
        if loss is not None:
            # 反向传播
            loss.backward()
            
            # 计算梯度范数
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
            # debug_logger.warning(f"梯度范数过大: {total_norm.item()}")
            
            # # 记录每个参数的梯度范数
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         param_norm = param.grad.data.norm(2).item()
            #         if param_norm > 5.0:  # 只记录较大的梯度
            #             debug_logger.warning(f"  {name} 的梯度范数: {param_norm:.4f}")
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            # 更新损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    return total_loss / len(train_loader)

def evaluate(model, data_loader, device, config):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    # 收集预测和标签
    all_ner_preds = []
    all_ner_labels = []
    all_re_preds = []
    all_re_labels = []
    
    # 统计标签分布
    label_counts = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue
                
            # 将数据移动到设备上
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            # 记录第一个batch的输出键和形状
            if batch_idx == 0:
                logger.info("模型输出包含以下属性: loss, ner_logits, relation_logits, entity_type_logits, hidden_states, attentions")
                if hasattr(outputs, 'ner_logits'):
                    logger.info(f"输出 ner_logits 的形状: {outputs.ner_logits.shape}")
                if hasattr(outputs, 'relation_logits'):
                    logger.info(f"输出 relation_logits 的形状: {outputs.relation_logits.shape}")
                logger.info(f"输入batch包含以下键: {list(batch.keys())}")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"输入 {k} 的形状: {v.shape}")
            
            # 收集NER预测和标签
            if hasattr(outputs, 'ner_logits'):
                ner_logits = outputs.ner_logits
                if 'labels' in batch:
                    ner_preds = ner_logits.argmax(dim=-1)
                    labels = batch['labels']
                    
                    # 统计标签分布
                    for i in range(config['model']['num_labels']):
                        count = (labels == i).sum().item()
                        label_counts[i] = label_counts.get(i, 0) + count
                    
                    # 确保维度匹配
                    if len(ner_preds.shape) == len(labels.shape):
                        all_ner_preds.append(ner_preds.cpu().numpy())
                        all_ner_labels.append(labels.cpu().numpy())
                        if batch_idx == 0:
                            logger.info(f"NER预测形状: {ner_preds.shape}")
                            logger.info(f"NER标签形状: {labels.shape}")
                            logger.info(f"NER预测值范围: [{ner_preds.min().item()}, {ner_preds.max().item()}]")
                            logger.info(f"NER标签值范围: [{labels.min().item()}, {labels.max().item()}]")
            
            # 收集RE预测和标签
            if hasattr(outputs, 'relation_logits'):
                re_logits = outputs.relation_logits
                if 'relations' in batch:
                    re_preds = re_logits.argmax(dim=-1)
                    # 确保维度匹配
                    if len(re_preds.shape) == len(batch['relations'].shape):
                        all_re_preds.append(re_preds.cpu().numpy())
                        all_re_labels.append(batch['relations'].cpu().numpy())
                        if batch_idx == 0:
                            logger.info(f"RE预测形状: {re_preds.shape}")
                            logger.info(f"RE标签形状: {batch['relations'].shape}")
            elif batch_idx == 0:
                if not hasattr(outputs, 'relation_logits'):
                    logger.info("模型没有输出relation_logits")
                if 'relations' not in batch:
                    logger.info("输入batch中没有relations")
    
    # 输出标签分布
    logger.info("\n标签分布:")
    entity_types = config['model']['entity_types']
    for i in range(config['model']['num_labels']):
        type_name = entity_types[i] if i < len(entity_types) else f"未知类型-{i}"
        count = label_counts.get(i, 0)
        logger.info(f"  {type_name}: {count}")
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    
    # 初始化结果字典
    results = {'loss': avg_loss}
    
    # 计算NER指标（如果有）
    if all_ner_preds and all_ner_labels:
        try:
            ner_metrics = calculate_ner_metrics(
                all_ner_preds, 
                all_ner_labels,
                config['model']['entity_types']
            )
            results['ner'] = ner_metrics
        except Exception as e:
            logger.error(f"计算NER指标时出错: {str(e)}")
            logger.error(f"NER预测形状: {[p.shape for p in all_ner_preds]}")
            logger.error(f"NER标签形状: {[l.shape for l in all_ner_labels]}")
    
    # 计算RE指标（如果有）
    if all_re_preds and all_re_labels:
        try:
            # 从schema文件加载关系类型
            schema = load_json_or_jsonl(config['data']['schema_file'])
            relations = [item['predicate'] for item in schema]
            
            re_metrics = calculate_re_metrics(
                all_re_preds,
                all_re_labels,
                relations
            )
            results['re'] = re_metrics
            logger.info("成功计算RE指标")
        except Exception as e:
            logger.error(f"计算RE指标时出错: {str(e)}")
            logger.error(f"RE预测形状: {[p.shape for p in all_re_preds]}")
            logger.error(f"RE标签形状: {[l.shape for l in all_re_labels]}")
    else:
        logger.info("没有收集到RE预测和标签")
    
    return results

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    config = load_config(args.config)
    
    # 设置设备
    device = set_device(config)
    
    # 创建输出目录
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # 初始化训练
    model, tokenizer, optimizer, scheduler = initialize_training(config, device)
    
    # 加载数据
    train_loader, eval_loader = load_data(config, tokenizer)
    
    # 开始训练
    logger.info("开始训练...")
    best_loss = float('inf')
    best_f1 = 0.0
    
    # 训练循环
    epochs = trange(config['training']['num_train_epochs'], desc="训练进度")
    for epoch in epochs:
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, config)
        logger.info(f'Epoch {epoch+1}/{config["training"]["num_train_epochs"]} - Train Loss: {train_loss:.4f}')
        
        # 评估
        eval_metrics = evaluate(model, eval_loader, device, config)
        
        # 记录评估结果
        logger.info(f'Epoch {epoch+1}/{config["training"]["num_train_epochs"]} - 评估结果:')
        logger.info(f'  Loss: {eval_metrics["loss"]:.4f}')
        
        # 如果有NER指标
        if 'ner' in eval_metrics:
            logger.info("\nNER评估指标:")
            logger.info(format_metrics(eval_metrics['ner']))
        
        # 如果有RE指标
        if 're' in eval_metrics:
            logger.info("\nRE评估指标:")
            logger.info(format_metrics(eval_metrics['re']))
        
        # 计算总体F1分数（NER和RE的平均）
        current_f1 = 0.0
        num_tasks = 0
        
        if 'ner' in eval_metrics:
            current_f1 += eval_metrics['ner']['f1']
            num_tasks += 1
            
        if 're' in eval_metrics:
            current_f1 += eval_metrics['re']['f1']
            num_tasks += 1
            
        # 只在有任务时计算平均值
        if num_tasks > 0:
            current_f1 = current_f1 / num_tasks
        
        # 保存最佳模型（基于F1）
        if current_f1 > best_f1:
            best_f1 = current_f1
            output_dir = os.path.join(config['output']['output_dir'], f"model_epoch_{epoch+1}_f1_{best_f1:.4f}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存模型配置
            model.config.save_pretrained(output_dir)
            
            # 保存模型权重
            state_dict = model.state_dict()
            safetensors.torch.save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            
            # 保存分词器
            tokenizer.save_pretrained(output_dir)
            logger.info(f"保存最佳模型到: {output_dir}")
        
        # 定期保存检查点
        if (epoch + 1) % config['output'].get('save_checkpoint_epochs', 5) == 0:
            checkpoint_path = os.path.join(
                config['output']['output_dir'],
                f'checkpoint_epoch_{epoch+1}_f1_{current_f1:.4f}'
            )
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # 保存模型配置
            model.config.save_pretrained(checkpoint_path)
            
            # 保存模型权重
            state_dict = model.state_dict()
            safetensors.torch.save_file(state_dict, os.path.join(checkpoint_path, "model.safetensors"))
            
            # 保存分词器
            tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"保存检查点到: {checkpoint_path}")
        
        # 控制检查点总数
        checkpoints = sorted(
            [d for d in os.listdir(config['output']['output_dir']) if d.startswith('checkpoint_')],
            key=lambda x: int(x.split('_')[2])
        )
        while len(checkpoints) > config['output'].get('save_total_limit', 3):
            oldest_checkpoint = os.path.join(config['output']['output_dir'], checkpoints[0])
            import shutil
            shutil.rmtree(oldest_checkpoint)
            checkpoints.pop(0)

if __name__ == "__main__":
    main()