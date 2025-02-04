import unittest
import torch
import os
import json
import tempfile
import yaml
from train import (
    load_json_or_jsonl,
    load_config,
    set_device,
    get_optimizer,
    CMeIEDataset,
    collate_fn,
    load_data,
    load_schema,
    initialize_training,
    weighted_cross_entropy_loss,
    calculate_metrics
)
from transformers import AutoTokenizer, AutoConfig, BertTokenizer, BertTokenizerFast
from modern_model_re import ModernBertForRelationExtraction

class TestTrainUtils(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        # 加载配置
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        # 设置测试参数
        self.config['max_length'] = 128
        self.config['batch_size'] = 32

        # 创建测试文件
        os.makedirs('tests', exist_ok=True)
        self.data_file = 'tests/test_data.jsonl'
        self.schema_file = 'tests/test_schema.jsonl'
        
        # 从实际数据中读取前5行作为测试数据
        with open(self.config['data']['train_file'], 'r', encoding='utf-8') as f_src, \
             open(self.data_file, 'w', encoding='utf-8') as f_dst:
            for i, line in enumerate(f_src):
                if i >= 5:  # 只取前5行
                    break
                f_dst.write(line)
        
        # 复制 schema 文件
        with open(self.config['data']['schema_file'], 'r', encoding='utf-8') as f_src, \
             open(self.schema_file, 'w', encoding='utf-8') as f_dst:
            for line in f_src:
                f_dst.write(line)

        # 创建分词器和数据集
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['model_name_or_path'])
        self.dataset = CMeIEDataset(
            data_file=self.data_file,
            schema_file=self.schema_file,
            tokenizer=self.tokenizer,
            config=self.config,
            max_length=self.config['max_length']
        )

    def test_load_json_or_jsonl(self):
        """测试JSON/JSONL文件加载"""
        # 测试JSONL格式
        data = load_json_or_jsonl(self.data_file)
        self.assertEqual(len(data), 5)
        self.assertIsInstance(data[0], dict)

        # 测试JSON格式
        json_file = os.path.join(tempfile.mkdtemp(), "test.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([{"text": "患者发热，建议服用布洛芬。"}], f, ensure_ascii=False)
        data = load_json_or_jsonl(json_file)
        self.assertEqual(len(data), 1)

    def test_load_schema(self):
        """测试schema加载"""
        schema = load_schema(self.schema_file)
        self.assertGreater(len(schema), 0)
        self.assertIsInstance(schema, list)

    def test_collate_fn(self):
        """测试 collate_fn 函数"""
        # 创建一个简单的 batch
        batch = [
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [0, 1, 2],
                'relation_labels': [0, 1],
                'entity_spans': [(0, 1, 1, 2), (1, 2, 2, 3)],
                'entity_types': [[0, 1], [1, 0]]  # 每个关系对应一对实体类型
            },
            {
                'input_ids': [4, 5, 6],
                'attention_mask': [1, 1, 0],
                'labels': [2, 1, 0],
                'relation_labels': [1, 0],
                'entity_spans': [(0, 1, 1, 2)],
                'entity_types': [[1, 0]]
            }
        ]
        
        # 调用 collate_fn
        collated_batch = collate_fn(batch)
        
        # 验证 collate_fn 的输出
        self.assertIn('input_ids', collated_batch)
        self.assertIn('attention_mask', collated_batch)
        self.assertIn('labels', collated_batch)
        self.assertIn('relation_labels', collated_batch)
        self.assertIn('entity_spans', collated_batch)
        self.assertIn('entity_types', collated_batch)
        
        # 验证张量形状
        self.assertEqual(collated_batch['input_ids'].shape[0], 2)
        self.assertEqual(collated_batch['attention_mask'].shape[0], 2)
        self.assertEqual(collated_batch['labels'].shape[0], 2)
        self.assertEqual(collated_batch['relation_labels'].shape[0], 2)
        self.assertEqual(collated_batch['entity_spans'].shape[0], 2)
        self.assertEqual(collated_batch['entity_types'].shape[0], 2)

    def test_dataset(self):
        """测试数据集加载"""
        # 检查数据集大小
        self.assertGreater(len(self.dataset), 0)

        # 检查schema是否正确加载
        self.assertTrue(hasattr(self.dataset, 'schema'), "数据集应该有schema属性")
        self.assertTrue(hasattr(self.dataset, 'relation2id'), "数据集应该有relation2id属性")
        self.assertTrue(hasattr(self.dataset, 'entity_types'), "数据集应该有entity_types属性")
        self.assertGreater(len(self.dataset.schema), 0, "schema不应该为空")
        self.assertGreater(len(self.dataset.entity_types), 0, "entity_types不应该为空")

        # 获取一个样本
        sample = self.dataset[0]

        # 检查数据格式
        required_fields = {
            'input_ids', 'attention_mask', 'labels', 'relation_labels',
            'entity_spans', 'entity_types'
        }
        self.assertTrue(
            all(field in sample for field in required_fields),
            f"缺少必要字段，当前字段：{list(sample.keys())}"
        )

        # 检查数据类型
        self.assertIsInstance(sample['input_ids'], list)
        self.assertIsInstance(sample['attention_mask'], list)
        self.assertIsInstance(sample['labels'], list)
        self.assertIsInstance(sample['relation_labels'], list)
        self.assertIsInstance(sample['entity_spans'], list)
        self.assertIsInstance(sample['entity_types'], list)
        
        # 检查每个实体类型列表的形状
        self.assertEqual(len(sample['entity_types']), len(sample['relation_labels']), 
                        "实体类型对的数量应该等于关系数量")
        for entity_type_pair in sample['entity_types']:
            self.assertEqual(len(entity_type_pair), 2, 
                           "每个实体类型对应该包含两个类型ID")

        # 检查关系有效性
        for i, relation in enumerate(sample['relation_labels']):
            # 关系ID应该在schema中定义
            self.assertLess(relation, len(self.dataset.schema), 
                          f"关系ID {relation} 超出schema范围")
            
            # 获取实体span
            span = sample['entity_spans'][i]
            entity_type1 = sample['entity_types'][i][0]  # 第i个关系的头实体类型
            entity_type2 = sample['entity_types'][i][1]  # 第i个关系的尾实体类型
            
            # 检查实体类型是否有效
            self.assertLess(entity_type1, len(self.dataset.entity_types), 
                          f"实体类型1 {entity_type1} 超出范围，最大值应为 {len(self.dataset.entity_types)-1}")
            self.assertLess(entity_type2, len(self.dataset.entity_types), 
                          f"实体类型2 {entity_type2} 超出范围，最大值应为 {len(self.dataset.entity_types)-1}")
            
            # 检查实体span是否有效
            start1, end1, start2, end2 = span
            self.assertTrue(0 <= start1 < end1 <= len(sample['input_ids']), 
                          f"实体1的span {(start1, end1)} 超出范围")
            self.assertTrue(0 <= start2 < end2 <= len(sample['input_ids']), 
                          f"实体2的span {(start2, end2)} 超出范围")

    def test_metrics(self):
        """测试评估指标计算"""
        # 创建模拟的logits和labels
        logits = torch.randn(2, 10, 3)  # (batch_size, seq_len, num_classes)
        labels = torch.randint(0, 3, (2, 10))  # (batch_size, seq_len)
        
        metrics = calculate_metrics(logits, labels)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)

    def test_loss(self):
        """测试损失函数"""
        logits = torch.randn(2, 10, 3)  # (batch_size, seq_len, num_classes)
        labels = torch.randint(0, 3, (2, 10))  # (batch_size, seq_len)
        weights = torch.ones(3)  # (num_classes,)
        
        loss = weighted_cross_entropy_loss(logits, labels, weights)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # 标量

    def test_optimizer(self):
        """测试优化器配置"""
        # 创建模型配置
        model_config = AutoConfig.from_pretrained(self.config['model']['model_name_or_path'])
        model_config.num_labels = 5  # 实体类型数量 * 2 + 1 (O标签)
        model_config.num_relations = len(self.dataset.schema)  # 关系类型数量
    
        # 添加 ModernBERT 所需的配置
        model_config.norm_eps = 1e-6
        model_config.norm_bias = True
        model_config.use_cache = self.config['model'].get('use_cache', True)
        model_config.use_rope = True
        model_config.use_parallel_attention = True
        model_config.use_flash_attention = False
        model_config.use_fused_attention = False
        model_config.use_fused_mlp = False
        model_config.use_fused_softmax = False
        model_config.use_fused_dropout = False
        model_config.embedding_dropout = 0.1
        model_config.hidden_dropout = 0.1
        model_config.attention_dropout = 0.1
        model_config.classifier_dropout = 0.1
        model_config.hidden_size = 768
        model_config.intermediate_size = 3072
        model_config.num_attention_heads = 12
        model_config.num_hidden_layers = 12
        model_config.deterministic_flash_attn = True
        model_config.max_position_embeddings = 512
        model_config.type_vocab_size = 2
        model_config.layer_norm_eps = 1e-12
        model_config.attention_bias = True
        model_config.global_attn_every_n_layers = 1
        model_config.global_rope_theta = 10000
        model_config.mlp_bias = True
        model_config.hidden_activation = "gelu"
        model_config.mlp_dropout = 0.1
        model_config.initializer_cutoff_factor = 1.0

        # 创建模型
        model = ModernBertForRelationExtraction(model_config)

        # 创建优化器
        learning_rate = float(self.config['training']['learning_rate'])
        optimizer = get_optimizer(model, learning_rate)

        # 检查优化器类型
        self.assertIsInstance(optimizer, torch.optim.AdamW)

        # 检查学习率
        self.assertEqual(optimizer.param_groups[0]['lr'], learning_rate)

    def tearDown(self):
        """清理测试环境"""
        # 删除测试文件
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.schema_file):
            os.remove(self.schema_file)

if __name__ == "__main__":
    unittest.main()
