"""
测试 ModernBertForRelationExtraction 模型
"""
import unittest
import torch
from transformers import AutoConfig, ModernBertModel
from modern_model_re import ModernBertForRelationExtraction
import numpy as np

class TestModernBertForRelationExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建配置
        cls.config = AutoConfig.from_pretrained("/Users/fufu/Downloads/modern_bert_multilingual")
        
        # 设置任务相关的配置
        cls.config.num_labels = 5  # 2个实体类型：B-Type1, I-Type1, B-Type2, I-Type2, O
        cls.config.num_relations = 53  # CMeIE数据集的53种关系
        cls.config.max_relations = 5  # 每个样本最多5个关系
        cls.config.classifier_dropout = 0.1
        cls.config.norm_eps = 1e-5
        cls.config.norm_bias = True
        
        # 初始化模型
        cls.model = ModernBertForRelationExtraction(cls.config)
        
        # 设置为评估模式以避免dropout等随机性
        cls.model.eval()
        
        # 准备输入数据
        cls.batch_size = 2
        cls.seq_len = 32
        cls.input_ids = torch.randint(0, cls.config.vocab_size, (cls.batch_size, cls.seq_len))
        cls.attention_mask = torch.ones(cls.batch_size, cls.seq_len)
        
        # 生成标签，部分位置使用 -100 表示忽略
        cls.labels = torch.randint(0, cls.config.num_labels, (cls.batch_size, cls.seq_len))
        mask = torch.rand(cls.batch_size, cls.seq_len) < 0.1  # 10% 的位置设为忽略
        cls.labels[mask] = -100
        
        # 生成关系标签，部分位置使用 -100 表示忽略
        cls.relations = torch.randint(0, cls.config.num_relations, (cls.batch_size, cls.config.max_relations))
        mask = torch.rand(cls.batch_size, cls.config.max_relations) < 0.1  # 10% 的位置设为忽略
        cls.relations[mask] = -100
        
        # 生成实体spans，包括一些无效的span（start > end）
        cls.entity_spans = torch.randint(0, cls.seq_len, (cls.batch_size, cls.config.max_relations, 4))
        # 设置一些无效的span
        invalid_mask = torch.rand(cls.batch_size, cls.config.max_relations) < 0.2  # 20% 的实体对设为无效
        for i, j in zip(*invalid_mask.nonzero(as_tuple=True)):
            cls.entity_spans[i, j] = torch.tensor([-1, -1, -1, -1])
        
    def test_model_config(self):
        """测试模型配置"""
        self.assertEqual(self.model.num_labels, self.config.num_labels)
        self.assertEqual(self.model.num_relations, self.config.num_relations)
        self.assertEqual(self.model.num_entity_types, (self.config.num_labels - 1) // 2)
        
    def test_model_components(self):
        """测试模型组件"""
        # 检查 ModernBERT 基础模型
        self.assertIsInstance(self.model.model, ModernBertModel)
        
        # 检查分类器
        self.assertIsInstance(self.model.ner_head, torch.nn.Linear)
        self.assertEqual(self.model.ner_head.out_features, self.config.num_labels)
        
        # 检查关系分类器
        self.assertIsInstance(self.model.relation_head, torch.nn.Linear)
        self.assertEqual(self.model.relation_head.out_features, self.config.num_relations)
        
        # 检查 dropout 和归一化层
        self.assertIsInstance(self.model.ner_dropout, torch.nn.Dropout)
        self.assertEqual(self.model.ner_dropout.p, self.config.classifier_dropout)
        self.assertIsInstance(self.model.relation_dropout, torch.nn.Dropout)
        self.assertEqual(self.model.relation_dropout.p, self.config.classifier_dropout)
        
        # 检查归一化层
        self.assertIsInstance(self.model.entity_norm, torch.nn.LayerNorm)
        self.assertEqual(self.model.entity_norm.eps, self.config.norm_eps)
        self.assertEqual(self.model.entity_norm.bias is not None, self.config.norm_bias)
        self.assertIsInstance(self.model.relation_norm, torch.nn.LayerNorm)
        self.assertEqual(self.model.relation_norm.eps, self.config.norm_eps)
        self.assertEqual(self.model.relation_norm.bias is not None, self.config.norm_bias)

    def test_get_entity_embeddings(self):
        """测试实体表示提取"""
        # 创建一个模拟的序列输出
        sequence_output = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        
        # 测试正常情况
        entity_embeds = self.model.get_entity_embeddings(sequence_output, self.entity_spans)
        expected_shape = (self.batch_size, self.config.max_relations, self.config.hidden_size * 2)
        self.assertEqual(entity_embeds.shape, expected_shape)
        self.assertTrue(torch.isfinite(entity_embeds).all())
        
        # 测试全部是无效span的情况
        invalid_spans = torch.full((self.batch_size, self.config.max_relations, 4), -1)
        entity_embeds = self.model.get_entity_embeddings(sequence_output, invalid_spans)
        self.assertEqual(entity_embeds.shape, expected_shape)
        self.assertTrue(torch.isfinite(entity_embeds).all())
        
        # 测试start > end的情况
        invalid_spans = torch.tensor([[[5, 3, 7, 4], [2, 1, 8, 6]]])
        entity_embeds = self.model.get_entity_embeddings(sequence_output[:1], invalid_spans)
        self.assertEqual(entity_embeds.shape, (1, 2, self.config.hidden_size * 2))
        self.assertTrue(torch.isfinite(entity_embeds).all())

    def test_forward_pass(self):
        """测试模型前向传播"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            relations=self.relations,
            entity_spans=self.entity_spans,
            return_dict=True
        )
        
        # 检查输出
        self.assertIsInstance(outputs, dict)
        self.assertIn('loss', outputs)
        self.assertIn('labels_logits', outputs)
        self.assertIn('relation_logits', outputs)
        
        # 检查输出形状
        self.assertEqual(
            outputs['labels_logits'].shape,
            (self.batch_size, self.seq_len, self.config.num_labels)
        )
        self.assertEqual(
            outputs['relation_logits'].shape,
            (self.batch_size, self.config.max_relations, self.config.num_relations)
        )
        
        # 检查损失值
        self.assertTrue(torch.isfinite(outputs['loss']))
        self.assertTrue(outputs['loss'] >= 0)

    def test_forward_without_labels(self):
        """测试无标签时的前向传播"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            entity_spans=self.entity_spans,
            return_dict=True
        )
        
        # 检查输出
        self.assertIsInstance(outputs, dict)
        self.assertIsNone(outputs['loss'])
        self.assertIn('labels_logits', outputs)
        self.assertIn('relation_logits', outputs)
        
        # 检查logits是否正确
        self.assertTrue(torch.isfinite(outputs['labels_logits']).all())
        self.assertTrue(torch.isfinite(outputs['relation_logits']).all())

    def test_only_ner_loss(self):
        """测试只有NER损失的情况"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            return_dict=True
        )
        
        # 检查输出
        self.assertIsInstance(outputs, dict)
        self.assertIn('loss', outputs)
        self.assertTrue(outputs['loss'] >= 0)
        self.assertIsNone(outputs['relation_logits'])

    def test_only_relation_loss(self):
        """测试只有关系损失的情况"""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            relations=self.relations,
            entity_spans=self.entity_spans,
            return_dict=True
        )
        
        # 检查输出
        self.assertIsInstance(outputs, dict)
        self.assertIn('loss', outputs)
        self.assertTrue(outputs['loss'] >= 0)
        self.assertIn('labels_logits', outputs)
        self.assertIn('relation_logits', outputs)

    def test_invalid_inputs(self):
        """测试无效输入的情况"""
        # 测试空的attention_mask
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=torch.zeros_like(self.attention_mask),
            labels=self.labels,
            relations=self.relations,
            entity_spans=self.entity_spans,
            return_dict=True
        )
        self.assertTrue(outputs['loss'] >= 0)
        
        # 测试全部是-100的labels
        invalid_labels = torch.full_like(self.labels, -100)
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=invalid_labels,
            relations=self.relations,
            entity_spans=self.entity_spans,
            return_dict=True
        )
        self.assertTrue(outputs['loss'] >= 0)
        
        # 测试全部是-100的relations
        invalid_relations = torch.full_like(self.relations, -100)
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            relations=invalid_relations,
            entity_spans=self.entity_spans,
            return_dict=True
        )
        self.assertTrue(outputs['loss'] >= 0)

if __name__ == '__main__':
    unittest.main()
