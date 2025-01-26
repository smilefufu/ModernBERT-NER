# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: CHIP/CBLUE 医学实体关系抽取，数据来源 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def load_name(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        samples = json.load(f)
        for sample in samples:
            spo_list = sample['spo_list']
            if not len(spo_list):
                continue
        
            D.append({
                "text": sample["text"],
                "spo_list":[(spo["subject"], 
                             spo["predicate"], 
                             spo["object"]["@value"], 
                             spo["subject_type"], 
                             spo["object_type"]["@value"],
                             spo["subject_start_idx"],
                             spo["subject_end_idx"],
                             spo["object_start_idx"],
                             spo["object_end_idx"]
                             )
                            for spo in sample["spo_list"]],
                "entities": sample['entities']
            })
        return D


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """
    Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, entity_schema):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema # spo
        self.entity_schema = entity_schema
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        # 不需要将文本转换为空格分隔的形式
        encoder_text = self.tokenizer(
            text,
            padding="max_length",  # 使用固定长度填充
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,  # 确保返回 attention_mask
        )
        
        input_ids = encoder_text["input_ids"]
        attention_mask = encoder_text["attention_mask"]
        # ModernBERT 不需要 token_type_ids
        token_type_ids = [0] * len(input_ids)  # 创建一个全零的列表
        
        spoes = set()
        for s_w, p, o_w, s_t, o_t, s_start_idx, s_end_idx, o_start_idx, o_end_idx in item["spo_list"]:
            s_w = ' '.join(list(s_w))
            o_w = ' '.join(list(o_w))
            sh = s_start_idx + 1
            oh = o_start_idx + 1
            st = s_end_idx + 1
            ot = o_end_idx + 1
            p = '有关系'
            p = self.schema[p]            
            if sh != -1 and oh != -1:
                spoes.add((sh, st, p, oh, ot))
        
        entity_labels = [set() for i in range(len(self.entity_schema))]
        for entity in item['entities']:
            start_idx = entity['start_idx']
            end_idx = entity['end_idx']
            et_type = entity['type']
            et_type_id = self.entity_schema[et_type]
            entity_labels[et_type_id].add((start_idx + 1, end_idx + 1))

        entity_labels_2 = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            entity_labels_2[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels_2[1].add((oh, ot))
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
    
        for label in entity_labels + head_labels + tail_labels:
            if not label:
                label.add((0,0))
                
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()

        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels


