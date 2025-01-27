import torch
import json
import yaml
from transformers import AutoTokenizer, ModernBertConfig
from src.bert_layers.modern_model_re import ModernBertForRelationExtraction

class RelationExtractor:
    def __init__(self, model_path, config_path="config.yaml", device='cuda'):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 使用配置初始化模型
        model_config = ModernBertConfig.from_pretrained(model_path)
        for key in [
            'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size',
            'hidden_activation', 'max_position_embeddings', 'norm_eps', 'norm_bias',
            'global_rope_theta', 'attention_bias', 'attention_dropout',
            'global_attn_every_n_layers', 'local_attention', 'local_rope_theta',
            'embedding_dropout', 'mlp_bias', 'mlp_dropout', 'classifier_pooling',
            'classifier_dropout', 'hidden_dropout_prob', 'attention_probs_dropout_prob'
        ]:
            if key in self.config['model']:
                setattr(model_config, key, self.config['model'][key])
        
        # 设置实体类型和关系数量
        model_config.entity_types = ["疾病", "症状", "检查", "手术", "药物", "其他治疗", "部位", "社会学", "流行病学", "预后", "其他"]
        model_config.num_relations = 53
        
        self.model = ModernBertForRelationExtraction.from_pretrained(
            model_path,
            config=model_config
        )
        self.model.to(self.device)
        self.model.eval()
        
        # 加载 schema
        self.schema = self.load_schema(self.config['data']['schema_file'])
        self.id2relation = {idx: rel for rel, idx in self.relation2id.items()}
        
    def load_schema(self, schema_file="data/53_schemas.jsonl"):
        schemas = []
        with open(schema_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    schema = json.loads(line)
                    schemas.append(f"{schema['subject_type']}_{schema['predicate']}_{schema['object_type']}")
        return schemas
    
    def extract_entities(self, text, bio_labels):
        entities = []
        current_entity = None
        
        for i, label in enumerate(bio_labels):
            if label == 1:  # B
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {'start': i, 'end': i + 1, 'text': text[i]}
            elif label == 2:  # I
                if current_entity is not None:
                    current_entity['end'] = i + 1
                    current_entity['text'] += text[i]
            else:  # O
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                    
        if current_entity is not None:
            entities.append(current_entity)
            
        return entities
    
    def predict(self, text):
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # 将输入移到设备上
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 获取实体识别结果
            bio_logits = outputs[1]
            bio_preds = torch.argmax(bio_logits, dim=-1)[0]
            
            # 提取实体
            entities = self.extract_entities(text, bio_preds.cpu().numpy())
            
            # 如果找到实体，预测关系
            results = []
            if len(entities) >= 2:
                for i in range(len(entities)):
                    for j in range(len(entities)):
                        if i != j:
                            # 准备实体对的位置信息
                            entity_positions = torch.tensor([[
                                entities[i]['start'],
                                entities[j]['start']
                            ]]).to(self.device)
                            
                            # 预测关系
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                entity_positions=entity_positions
                            )
                            
                            if len(outputs) > 3:
                                relation_logits = outputs[3]
                                relation_pred = torch.argmax(relation_logits, dim=-1).item()
                                
                                # 添加到结果中
                                results.append({
                                    'subject': entities[i]['text'],
                                    'object': entities[j]['text'],
                                    'relation': self.id2relation[relation_pred]
                                })
            
            return results

def main():
    # 使用示例
    extractor = RelationExtractor("./results/best_model")
    text = "患者有高血压病史10年，长期服用降压药。"
    results = extractor.predict(text)
    print(f"输入文本：{text}")
    print("提取结果：")
    for r in results:
        print(f"主体：{r['subject']}")
        print(f"关系：{r['relation']}")
        print(f"客体：{r['object']}")
        print("---")

if __name__ == "__main__":
    main()