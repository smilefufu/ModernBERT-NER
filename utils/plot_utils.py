"""绘图相关的工具函数"""
import os
import matplotlib.pyplot as plt

def plot_training_metrics(metrics_history, output_dir):
    """绘制训练过程中的指标变化图
    
    Args:
        metrics_history: 包含训练过程中各项指标的历史记录
        output_dir: 输出目录，用于保存图表
    """
    # 创建图表目录
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 的中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history['train_loss'], label='训练损失')
    if 'eval_loss' in metrics_history:
        plt.plot(metrics_history['eval_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练过程中的损失变化')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'loss.png'))
    plt.close()
    
    # 绘制F1分数曲线（如果有的话）
    if 'eval_f1' in metrics_history:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_history['eval_f1'], label='F1分数')
        plt.xlabel('Epoch')
        plt.ylabel('F1分数')
        plt.title('验证集上的F1分数变化')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'f1_score.png'))
        plt.close()
    
    # 绘制学习率变化曲线（如果有的话）
    if 'learning_rate' in metrics_history:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_history['learning_rate'], label='学习率')
        plt.xlabel('Epoch')
        plt.ylabel('学习率')
        plt.title('学习率变化')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # 使用对数坐标
        plt.savefig(os.path.join(plots_dir, 'learning_rate.png'))
        plt.close()
