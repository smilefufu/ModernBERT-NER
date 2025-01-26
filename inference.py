import torch
import json
from transformers import AutoTokenizer
from src.bert_layers.model_re import FlexBertForRelationExtraction

class RelationExtractor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = FlexBertForRelationExtraction.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载 schema
        self.schema = self.load_schema()
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