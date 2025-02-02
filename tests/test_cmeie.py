"""
测试 CMeIE 数据集加载器
"""
import unittest
import json
import torch
from transformers import BertTokenizerFast
from data.cmeie import CMeIEDataset, load_json_or_jsonl

class TestCMeIEDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 创建一个简单的测试数据文件
        cls.test_data = [
            {
                "text": "患者出现发热症状，建议服用布洛芬退烧。",
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
            }
        ]
        
        # 创建一个简单的schema文件
        cls.test_schema = [
            {"predicate": "治疗"},
            {"predicate": "预防"},
            {"predicate": "检查"}
        ]
        
        # 写入测试文件
        with open('/tmp/test_data.json', 'w', encoding='utf-8') as f:
            json.dump(cls.test_data, f, ensure_ascii=False)
            
        with open('/tmp/test_schema.json', 'w', encoding='utf-8') as f:
            json.dump(cls.test_schema, f, ensure_ascii=False)
            
        # 初始化tokenizer和数据集
        cls.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        cls.dataset = CMeIEDataset(
            data_file='/tmp/test_data.json',
            tokenizer=cls.tokenizer,
            schema_file='/tmp/test_schema.json',
            max_length=128
        )

    def test_data_loading(self):
        """测试数据加载功能"""
        # 测试数据集大小
        self.assertEqual(len(self.dataset), 1)
        
        # 测试schema加载
        self.assertEqual(len(self.dataset.schema), 3)
        self.assertIn('治疗', self.dataset.schema)
        
        # 测试关系到ID的映射存在性
        self.assertIn('治疗', self.dataset.relation2id)
        self.assertIsInstance(self.dataset.relation2id['治疗'], int)

    def test_entity_extraction(self):
        """测试实体提取功能"""
        text = "患者发热，建议服用布洛芬。"
        spo_list = [
            {
                "subject": "发热",
                "subject_type": "症状",
                "predicate": "治疗",
                "object": {"@value": "布洛芬"},
                "object_type": {"@value": "药物"},
                "subject_start_idx": text.index("发热"),
                "object_start_idx": text.index("布洛芬")
            }
        ]
        
        # 提取实体
        entities = self.dataset.extract_entities_from_spo(text, spo_list)
        
        # 由于布洛芬和发热长度相同，按照起始位置排序
        self.assertEqual(entities[0][0], '发热')
        self.assertEqual(entities[0][3], '症状')
        self.assertEqual(entities[1][0], '布洛芬')
        self.assertEqual(entities[1][3], '药物')

    def test_bio_labeling(self):
        """测试BIO标注功能"""
        sample = self.dataset[0]
        labels = sample['labels']
        
        # 确保标签是一个tensor
        self.assertIsInstance(labels, torch.Tensor)
        
        # 获取分词结果以便调试
        tokens = self.tokenizer.convert_ids_to_tokens(sample['input_ids'])
        
        # 找到实体对应的token位置
        发热_idx = tokens.index('发')
        布洛芬_idx = tokens.index('布')
        
        # 验证"发热"的标注（症状）
        症状_B = self.dataset.entity_type2id['症状'] * 2 + 1
        症状_I = 症状_B + 1
        self.assertEqual(labels[发热_idx].item(), 症状_B)  # B-症状
        self.assertEqual(labels[发热_idx + 1].item(), 症状_I)  # I-症状
        
        # 验证"布洛芬"的标注（药物）
        药物_B = self.dataset.entity_type2id['药物'] * 2 + 1
        药物_I = 药物_B + 1
        self.assertEqual(labels[布洛芬_idx].item(), 药物_B)  # B-药物
        self.assertEqual(labels[布洛芬_idx + 1].item(), 药物_I)  # I-药物

    def test_relation_mapping(self):
        """测试关系映射功能"""
        sample = self.dataset[0]
        
        # 获取分词结果以便调试
        tokens = self.tokenizer.convert_ids_to_tokens(sample['input_ids'])
        print("\n分词结果:", tokens)
        
        # 验证关系ID
        relation_id = sample['relations'][0].item()
        self.assertEqual(relation_id, self.dataset.relation2id['治疗'])
        
        # 验证实体span
        spans = sample['spans'][0]
        subject_start = spans[0].item()
        subject_end = spans[1].item()
        object_start = spans[2].item()
        object_end = spans[3].item()
        
        print(f"\n主实体span: [{subject_start}, {subject_end}]")
        print(f"主实体tokens: {tokens[subject_start:subject_end+1]}")
        print(f"\n客实体span: [{object_start}, {object_end}]")
        print(f"客实体tokens: {tokens[object_start:object_end+1]}")
        
        # 验证主实体span（发热）- 确保span内包含发热的字符
        subject_text = ''.join(tokens[subject_start:subject_end+1]).replace('#', '')
        self.assertTrue('发' in subject_text)
        self.assertTrue('热' in subject_text)
        
        # 验证客实体span（布洛芬）- 确保span内包含布洛芬的字符
        object_text = ''.join(tokens[object_start:object_end+1]).replace('#', '')
        self.assertTrue('布' in object_text)
        self.assertTrue('芬' in object_text)

    def test_invalid_data(self):
        """测试无效数据的处理"""
        invalid_data = [
            {},  # 空字典
            {"text": ""},  # 空文本
            {"text": "测试", "spo_list": None},  # 无效的spo_list
            {"text": "测试", "spo_list": [{}]},  # 无效的spo
        ]
        
        for data in invalid_data:
            self.assertFalse(self.dataset.validate_sample(data))

    def test_long_text(self):
        """测试长文本的截断"""
        # 创建一个超长的文本，但确保实体位置正确
        base_text = "这是一个测试"  # 长度为5
        long_text = base_text * 100
        long_data = [{
            "text": long_text,
            "spo_list": [
                {
                    "subject": base_text,
                    "subject_type": "其他",
                    "predicate": "治疗",
                    "object": {"@value": base_text},
                    "object_type": {"@value": "其他治疗"},
                    "subject_start_idx": 0,  # 第一个实例的位置
                    "object_start_idx": len(base_text)  # 第二个实例的位置
                }
            ]
        }]
        
        # 写入临时文件
        with open('/tmp/test_long.json', 'w', encoding='utf-8') as f:
            json.dump(long_data, f, ensure_ascii=False)
        
        # 创建数据集
        dataset = CMeIEDataset(
            data_file='/tmp/test_long.json',
            tokenizer=self.tokenizer,
            schema_file='/tmp/test_schema.json',
            max_length=128
        )
        
        # 获取第一个样本
        sample = dataset[0]
        
        # 验证输入长度被截断
        self.assertEqual(len(sample['input_ids']), 128)
        self.assertEqual(len(sample['attention_mask']), 128)
        self.assertEqual(len(sample['labels']), 128)

    def test_special_chars(self):
        """测试特殊字符的处理"""
        text = "患者发热，建议服用布洛芬。"
        special_data = [{
            "text": text,
            "spo_list": [
                {
                    "subject": "发热",
                    "subject_type": "症状",
                    "predicate": "治疗",
                    "object": {"@value": "布洛芬"},
                    "object_type": {"@value": "药物"},
                    "subject_start_idx": text.index("发热"),  # 动态计算位置
                    "object_start_idx": text.index("布洛芬")  # 动态计算位置
                }
            ]
        }]
        
        # 写入临时文件
        with open('/tmp/test_special.json', 'w', encoding='utf-8') as f:
            json.dump(special_data, f, ensure_ascii=False)
        
        # 创建数据集
        dataset = CMeIEDataset(
            data_file='/tmp/test_special.json',
            tokenizer=self.tokenizer,
            schema_file='/tmp/test_schema.json',
            max_length=128
        )
        
        # 获取第一个样本
        sample = dataset[0]
        
        # 验证特殊字符被正确处理
        tokens = self.tokenizer.convert_ids_to_tokens(sample['input_ids'])
        text = ''.join(tokens).replace('#', '')
        self.assertIn('发热', text)
        self.assertIn('布洛芬', text)

    def test_overlapping_entities(self):
        """测试实体重叠的情况"""
        text = "患者重度发热，需要退烧药。"
        overlap_data = [
            {
                "text": text,
                "spo_list": [
                    {
                        "subject": "重度发热",
                        "subject_type": "症状",
                        "predicate": "治疗",
                        "object": {"@value": "退烧药"},
                        "object_type": {"@value": "药物"},
                        "subject_start_idx": text.index("重度发热"),  # 动态计算位置
                        "object_start_idx": text.index("退烧药")  # 动态计算位置
                    },
                    {
                        "subject": "发热",
                        "subject_type": "症状",
                        "predicate": "治疗",
                        "object": {"@value": "退烧药"},
                        "object_type": {"@value": "药物"},
                        "subject_start_idx": text.index("重度发热") + 2,  # 动态计算位置，跳过"重度"
                        "object_start_idx": text.index("退烧药")  # 动态计算位置
                    }
                ]
            }
        ]
        
        # 写入临时文件
        with open('/tmp/test_overlap.json', 'w', encoding='utf-8') as f:
            json.dump(overlap_data, f, ensure_ascii=False)
        
        # 创建数据集
        dataset = CMeIEDataset(
            data_file='/tmp/test_overlap.json',
            tokenizer=self.tokenizer,
            schema_file='/tmp/test_schema.json',
            max_length=128
        )
        
        # 获取第一个样本
        sample = dataset[0]
        
        # 验证重叠实体都被正确识别
        entities = dataset.extract_entities_from_spo(overlap_data[0]['text'], overlap_data[0]['spo_list'])
        
        # 打印实体信息以便调试
        print("\n提取的实体:")
        for e in entities:
            print(f"实体: {e[0]}, 类型: {e[3]}, 位置: [{e[1]}, {e[2]}]")
        
        # 验证提取的实体数量（应该有3个实体：重度发热、发热、退烧药）
        self.assertEqual(len(entities), 3)
        
        # 验证标签序列
        tokens = self.tokenizer.convert_ids_to_tokens(sample['input_ids'])
        labels = sample['labels']
        
        print("\n分词结果:", tokens)
        
        # 找到"重度发热"的位置
        for i, token in enumerate(tokens):
            if token == '重':
                重度发热_start = i
                break
        
        # 验证"重度发热"的标注
        症状_B = self.dataset.entity_type2id['症状'] * 2 + 1
        症状_I = 症状_B + 1
        
        print(f"\n标签序列: {labels}")
        print(f"'重度发热' 开始位置: {重度发热_start}")
        print(f"症状标签: B={症状_B}, I={症状_I}")
        
        # 验证标注
        self.assertEqual(labels[重度发热_start].item(), 症状_B)  # B-症状
        self.assertEqual(labels[重度发热_start + 1].item(), 症状_I)  # I-症状
        self.assertEqual(labels[重度发热_start + 2].item(), 症状_I)  # I-症状

if __name__ == '__main__':
    unittest.main()
