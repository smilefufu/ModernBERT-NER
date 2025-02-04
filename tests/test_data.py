import sys
import os
import unittest
import torch
import yaml
import json
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.cmeie import CMeIEDataset

class TestCMeIEDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建测试数据
        cls.test_data = [{
            "text": "患者出现发热症状，建议服用布洛芬。",
            "spo_list": [
                {
                    "subject": "发热",
                    "subject_type": "症状",
                    "predicate": "治疗",
                    "object": {"@value": "布洛芬"},
                    "object_type": {"@value": "药物"},
                    "subject_start_idx": 4,
                    "object_start_idx": 13
                }
            ]
        }]

        # 创建配置
        cls.config = {
            'data_file': '/tmp/test.json',
            'schema_file': '/tmp/test_schema.json',
            'max_length': 128,
            'model_name_or_path': 'bert-base-chinese'
        }

        # 写入测试数据
        with open('/tmp/test.json', 'w', encoding='utf-8') as f:
            json.dump(cls.test_data, f, ensure_ascii=False)

        # 写入模式文件
        schema = [
            {
                "predicate": "治疗",
                "subject_type": ["症状", "疾病"],
                "object_type": {"@value": ["药物", "手术治疗"]}
            }
        ]
        with open('/tmp/test_schema.json', 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False)

        # 初始化分词器和数据集
        cls.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        cls.dataset = CMeIEDataset(
            data_file='/tmp/test.json',
            tokenizer=cls.tokenizer,
            schema_file='/tmp/test_schema.json',
            config=cls.config,
            max_length=cls.config['max_length']
        )

    def test_data_format(self):
        """测试数据格式"""
        item = self.dataset[0]
        
        # 检查返回的字典包含所有必要的键
        required_keys = {'input_ids', 'attention_mask', 
                        'labels', 'relation_labels', 'entity_spans', 'entity_types'}
        self.assertTrue(all(key in item for key in required_keys))
        
        # 检查张量的形状
        self.assertEqual(len(item['input_ids']), self.dataset.max_length)
        self.assertEqual(len(item['attention_mask']), self.dataset.max_length)
        self.assertEqual(len(item['labels']), self.dataset.max_length)
        
        # 检查关系相关列表的形状
        num_relations = len(item['relation_labels'])
        self.assertEqual(len(item['entity_spans']), num_relations)
        self.assertEqual(len(item['entity_types']), num_relations)

    def test_label_values(self):
        """测试标签值的范围"""
        item = self.dataset[0]
        labels = item['labels']
        
        # 计算应该有的标签数量（每个实体类型有B和I标签）
        num_entity_types = len(self.dataset.entity_types)
        max_label_value = num_entity_types * 2 + 1  # BIO标注
        
        # 验证标签值在合理范围内
        self.assertTrue(all(0 <= label <= max_label_value for label in labels))
        
        # 验证B标签和I标签的对应关系
        for i in range(len(labels) - 1):
            if labels[i] > 0 and labels[i] % 2 == 1:  # B标签
                # 如果下一个标签不是对应的I标签或O标签，报错
                next_label = labels[i + 1]
                self.assertTrue(
                    next_label == 0 or  # O标签
                    next_label == labels[i] + 1  # 对应的I标签
                )

    def test_entity_extraction(self):
        """测试实体提取"""
        item = self.dataset[0]
        
        # 检查实体spans的格式
        for span in item['entity_spans']:
            # 每个span应该是一个包含4个整数的元组或列表
            self.assertEqual(len(span), 4)
            # 起始位置应该小于等于结束位置
            self.assertTrue(span[0] <= span[1] and span[2] <= span[3])
            # 位置应该在序列长度范围内
            self.assertTrue(all(0 <= pos < self.dataset.max_length for pos in span))

if __name__ == '__main__':
    unittest.main()
