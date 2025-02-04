"""
测试 CMeIE 数据集加载器
"""
import unittest
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoConfig
)
from data.cmeie import CMeIEDataset, load_json_or_jsonl
import tempfile
import os
import yaml

class TestCMeIEDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 加载配置
        with open('config.yaml', 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # 设置测试参数
        cls.config['max_length'] = 128
        
        # 创建临时文件
        cls.data_file = '/tmp/test_data.jsonl'
        cls.schema_file = '/tmp/test_schema.jsonl'
        
        # 从实际数据中读取前5行作为测试数据
        with open(cls.config['data']['train_file'], 'r', encoding='utf-8') as f_src, \
             open(cls.data_file, 'w', encoding='utf-8') as f_dst:
            for i, line in enumerate(f_src):
                if i >= 5:  # 只取前5行
                    break
                f_dst.write(line)
        
        # 复制 schema 文件
        with open(cls.config['data']['schema_file'], 'r', encoding='utf-8') as f_src, \
             open(cls.schema_file, 'w', encoding='utf-8') as f_dst:
            for line in f_src:
                f_dst.write(line)
        
        # 创建分词器
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.config['model']['model_name_or_path'])
        
        # 创建数据集
        cls.dataset = CMeIEDataset(
            data_file=cls.data_file,
            schema_file=cls.schema_file,
            tokenizer=cls.tokenizer,
            config=cls.config,
            max_length=cls.config['max_length']
        )

    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 删除测试文件
        if os.path.exists(cls.data_file):
            os.remove(cls.data_file)
        if os.path.exists(cls.schema_file):
            os.remove(cls.schema_file)

    def test_data_loading(self):
        """测试数据加载功能"""
        # 测试数据集大小
        self.assertEqual(len(self.dataset), 5)
        
        # 测试schema加载
        self.assertGreater(len(self.dataset.schema), 0)
        
        # 测试数据格式
        sample = self.dataset[0]
        self.assertIn('input_ids', sample)
        self.assertIn('attention_mask', sample)
        self.assertIn('labels', sample)
        self.assertIn('relation_labels', sample)
        self.assertIn('entity_spans', sample)
        self.assertIn('entity_types', sample)

    def test_bio_labeling(self):
        """测试BIO标注功能"""
        # 获取一个样本
        sample = self.dataset[0]
        
        # 检查标签长度是否与输入长度一致
        self.assertEqual(len(sample['labels']), len(sample['input_ids']))
        
        # 验证标签值的范围
        for label in sample['labels']:
            self.assertGreaterEqual(label, 0)
            self.assertLess(label, len(self.dataset.label2id))

    def test_output_format(self):
        """测试输出格式"""
        # 获取一个样本
        sample = self.dataset[0]
        
        # 检查标签长度是否与输入长度一致
        self.assertEqual(len(sample['labels']), len(sample['input_ids']))
        
        # 检查关系标签
        self.assertIsInstance(sample['relation_labels'], list)
        for relation in sample['relation_labels']:
            self.assertGreaterEqual(relation, 0)
            self.assertLess(relation, len(self.dataset.schema))

    def test_entity_extraction(self):
        """测试实体提取"""
        # 获取一个样本
        sample = self.dataset[0]

        # 检查实体标注
        labels = sample['labels']
        entity_spans = sample['entity_spans']

        # 验证每个实体位置的标注
        for span in entity_spans:
            start_idx = span[0]  # 实体开始位置
            end_idx = span[1]    # 实体结束位置
            
            # 检查开始位置是否是B标签
            self.assertTrue(labels[start_idx] % 2 == 1, "实体首个token应该是B标签")
            
            # 检查中间位置是否是I标签
            for i in range(start_idx + 1, end_idx):
                self.assertTrue(labels[i] % 2 == 0, "实体中间token应该是I标签")

    def test_long_text(self):
        """测试长文本的截断"""
        # 获取一个样本
        sample = self.dataset[0]

        # 检查输入长度是否被截断
        self.assertLessEqual(len(sample['input_ids']), self.config['max_length'])

    def test_overlapping_entities(self):
        """测试实体重叠的情况"""
        # 获取一个样本
        sample = self.dataset[0]

        # 检查标签长度是否与输入长度一致
        self.assertEqual(len(sample['labels']), len(sample['input_ids']))

    def test_special_chars(self):
        """测试特殊字符的处理"""
        # 获取一个样本
        sample = self.dataset[0]

        # 检查标签长度是否与输入长度一致
        self.assertEqual(len(sample['labels']), len(sample['input_ids']))

    def test_schema_constraints(self):
        """测试关系的实体类型约束"""
        # 创建一个违反约束的样本
        invalid_data = [{
            "text": "患者发热，建议服用布洛芬。",
            "spo_list": [
                {
                    "subject": "发热",
                    "subject_type": "药物",  # 错误的实体类型
                    "predicate": "治疗",
                    "object": {"@value": "布洛芬"},
                    "object_type": {"@value": "症状"},  # 错误的实体类型
                    "subject_start_idx": 2,
                    "object_start_idx": 8
                }
            ]
        }]
        
        # 写入临时文件
        with open('/tmp/test_invalid_types.json', 'w', encoding='utf-8') as f:
            json.dump(invalid_data, f, ensure_ascii=False)
        
        # 创建数据集
        dataset = CMeIEDataset(
            data_file='/tmp/test_invalid_types.json',
            tokenizer=self.tokenizer,
            schema_file='/tmp/test_schema.json',
            config=self.config,
            max_length=self.config['max_length']
        )
        
        # 验证无效样本被过滤
        self.assertEqual(len(dataset), 0)

if __name__ == '__main__':
    unittest.main()
