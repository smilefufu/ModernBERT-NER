import json
import sys
from typing import Optional, List, Dict, Any

def find_all_positions(text: str, target: str) -> List[int]:
    """查找目标字符串在文本中的所有出现位置"""
    positions = []
    start = 0
    while True:
        pos = text.find(target, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions

def process_line(line: dict) -> Optional[dict]:
    """处理单行数据，如果实体有多个匹配位置则返回None"""
    text = line['text']
    new_spo_list = []
    
    for spo in line['spo_list']:
        subject = spo['subject']
        # 对于object，需要处理可能存在的 @value 字段
        object_value = spo['object'].get('@value', spo['object']) if isinstance(spo['object'], dict) else spo['object']
        
        # 查找subject的所有位置
        subject_positions = find_all_positions(text, subject)
        # 查找object的所有位置
        object_positions = find_all_positions(text, object_value)
        
        # 如果任一实体有多个匹配位置，返回None
        if len(subject_positions) != 1 or len(object_positions) != 1:
            return None
            
        # 创建新的spo，包含start_idx
        new_spo = spo.copy()
        new_spo['subject_start_idx'] = subject_positions[0]
        new_spo['object_start_idx'] = object_positions[0]
        new_spo_list.append(new_spo)
    
    # 创建新的行数据
    new_line = line.copy()
    new_line['spo_list'] = new_spo_list
    return new_line

def main():
    input_file = sys.argv[1]
    output_file = input_file.rsplit('.', 1)[0] + '_with_idx.jsonl'
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line.strip())
            processed_data = process_line(data)
            
            if processed_data is not None:
                f_out.write(json.dumps(processed_data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
