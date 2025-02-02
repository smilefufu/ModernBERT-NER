import logging
import os
import sys
import json
import numpy as np
import torch
from typing import Any, Dict, Optional, Union
from pathlib import Path

class DebugLogger:
    """调试日志工具类，用于统一管理调试日志"""
    
    def __init__(self, name: str = "debug"):
        self.name = name
        self._setup_logger()
        
    def _setup_logger(self):
        """设置日志记录器"""
        # 获取项目根目录
        project_root = Path(__file__).parent.parent
        debug_dir = project_root / "debug_logs"
        debug_dir.mkdir(exist_ok=True)
        
        # 设置日志文件路径
        self.log_file = debug_dir / "debug.txt"
        
        # 创建日志记录器
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # 清理已有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器（只输出警告及以上级别）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # 设置格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_tensor(self, name: str, tensor: torch.Tensor, step: Optional[int] = None):
        """记录张量的详细信息"""
        if tensor is None:
            self.logger.debug(f"{name} is None")
            return
            
        info = {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'min': tensor.min().item() if tensor.numel() > 0 else None,
            'max': tensor.max().item() if tensor.numel() > 0 else None,
            'mean': tensor.float().mean().item() if tensor.numel() > 0 else None,
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item()
        }
        
        step_info = f" (step {step})" if step is not None else ""
        self.logger.debug(f"{name}{step_info}:")
        for k, v in info.items():
            self.logger.debug(f"  {k}: {v}")
    
    def log_distribution(self, name: str, values: Union[np.ndarray, torch.Tensor], 
                        labels: Optional[list] = None):
        """记录数值分布情况"""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        unique_vals, counts = np.unique(values, return_counts=True)
        
        self.logger.debug(f"\n{name} 分布:")
        for i, (val, count) in enumerate(zip(unique_vals, counts)):
            label = labels[int(val)] if labels and int(val) < len(labels) else str(val)
            self.logger.debug(f"  {label}: {count} 个")
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """记录评估指标"""
        self.logger.debug(f"\n{prefix}指标:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.debug(f"  {metric_name}: {value:.4f}")
            elif isinstance(value, np.ndarray):
                self.logger.debug(f"  {metric_name}:\n{value}")
            elif isinstance(value, dict):
                self.logger.debug(f"  {metric_name}:")
                for k, v in value.items():
                    if isinstance(v, dict):
                        self.logger.debug(f"    {k}:")
                        for sub_k, sub_v in v.items():
                            self.logger.debug(f"      {sub_k}: {sub_v:.4f}")
                    else:
                        self.logger.debug(f"    {k}: {v}")
    
    def log_data_sample(self, sample: Dict[str, Any], idx: int):
        """记录数据样本的详细信息"""
        self.logger.debug(f"\n样本 {idx} 详细信息:")
        for key, value in sample.items():
            if isinstance(value, (str, int, float)):
                self.logger.debug(f"  {key}: {value}")
            elif isinstance(value, (list, dict)):
                self.logger.debug(f"  {key}: {json.dumps(value, ensure_ascii=False, indent=2)}")
            elif isinstance(value, torch.Tensor):
                self.log_tensor(f"  {key}", value)
    
    def debug(self, msg: str):
        """输出调试信息"""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """输出信息"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """输出警告"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """输出错误"""
        self.logger.error(msg)

# 创建全局调试日志实例
debug_logger = DebugLogger()
