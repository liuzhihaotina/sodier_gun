import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
def set_chinese():
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from modules.chinese_font import chinese_font
    plt.rcParams['font.family'] = [chinese_font.get_name(), 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
sns.set_palette("husl")

def generate_test_data(n_samples: int = 10000, distribution_type: str = 'normal') -> np.ndarray:
    """
    生成测试数据
    
    参数:
        n_samples: 样本数量
        distribution_type: 分布类型 ('normal', 'uniform', 'exponential', 'log_normal')
    
    返回:
        np.ndarray: 生成的数据
    """
    np.random.seed(42)  # 设置随机种子以便复现
    
    if distribution_type == 'normal':
        # 正态分布 N(μ=100, σ=15)
        data = np.random.normal(loc=100, scale=15, size=n_samples)
    elif distribution_type == 'uniform':
        # 均匀分布 U[20, 180]
        data = np.random.uniform(low=20, high=180, size=n_samples)
    elif distribution_type == 'exponential':
        # 指数分布 λ=1/50
        data = np.random.exponential(scale=50, size=n_samples)
    elif distribution_type == 'log_normal':
        # 对数正态分布
        data = np.random.lognormal(mean=4, sigma=0.5, size=n_samples)
    else:
        raise ValueError(f"不支持的分布类型: {distribution_type}")
    
    return data

def standardize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    对数据进行标准化: z = (x - μ) / σ
    
    参数:
        data: 原始数据
    
    返回:
        tuple: (标准化后的数据, 原始均值, 原始标准差)
    """
    mean = np.mean(data)
    std = np.std(data, ddof=0)  # 使用总体标准差
    
    # 标准化
    standardized = (data - mean) / std
    
    return standardized, mean, std

def analyze_distribution(original_data: np.ndarray, 
                        standardized_data: np.ndarray,
                        title_prefix: str = "") -> None:
    """
    分析并可视化数据分布
    
    参数:
        original_data: 原始数据
        standardized_data: 标准化后的数据
        title_prefix: 图表标题前缀
    """
    # 计算统计量
    orig_mean = np.mean(original_data)
    orig_std = np.std(original_data, ddof=0)
    orig_skew = stats.skew(original_data)
    orig_kurtosis = stats.kurtosis(original_data)
    
    std_mean = np.mean(standardized_data)
    std_std = np.std(standardized_data, ddof=0)
    std_skew = stats.skew(standardized_data)
    std_kurtosis = stats.kurtosis(standardized_data)
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{title_prefix}数据分布验证 - 标准化变换", fontsize=16, fontweight='bold')
    
    # 1. 原始数据和标准化数据的直方图对比
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(original_data, bins=50, alpha=0.6, color='blue', density=True, 
             label=f'原始数据\nμ={orig_mean:.2f}, σ={orig_std:.2f}')
    ax1.set_xlabel('数值')
    ax1.set_ylabel('概率密度')
    ax1.set_title('原始数据分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(standardized_data, bins=50, alpha=0.6, color='red', density=True,
             label=f'标准化数据\nμ={std_mean:.4f}, σ={std_std:.4f}')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, label='均值线')
    ax2.set_xlabel('标准化值')
    ax2.set_ylabel('概率密度')
    ax2.set_title('标准化后数据分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2. 叠加理论正态分布
    ax3 = plt.subplot(2, 3, 3)
    # 绘制标准化数据的直方图
    ax3.hist(standardized_data, bins=50, alpha=0.6, color='red', density=True,
             label='标准化数据')
    
    # 绘制理论标准正态分布
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    ax3.plot(x, y, 'k-', linewidth=2, label='理论N(0,1)')
    
    ax3.set_xlabel('标准化值')
    ax3.set_ylabel('概率密度')
    ax3.set_title('与理论正态分布比较')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. QQ图（分位数-分位数图）
    ax4 = plt.subplot(2, 3, 4)
    stats.probplot(standardized_data, dist="norm", plot=ax4)
    ax4.set_title('QQ图 - 检验正态性')
    ax4.grid(True, alpha=0.3)
    
    # 4. 箱线图对比
    ax5 = plt.subplot(2, 3, 5)
    data_to_plot = [original_data, standardized_data]
    labels = ['原始数据', '标准化数据']
    bp = ax5.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # 设置箱线图颜色
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax5.set_ylabel('数值')
    ax5.set_title('箱线图对比')
    ax5.grid(True, alpha=0.3)
    
    # 添加均值线
    ax5.axhline(y=orig_mean, color='blue', linestyle=':', alpha=0.5)
    ax5.axhline(y=0, color='red', linestyle=':', alpha=0.5)
    
    # 5. 统计摘要表格
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # 创建表格数据
    table_data = [
        ['统计量', '原始数据', '标准化数据', '理论值'],
        ['均值(μ)', f'{orig_mean:.4f}', f'{std_mean:.6f}', '0.0000'],
        ['标准差(σ)', f'{orig_std:.4f}', f'{std_std:.6f}', '1.0000'],
        ['偏度', f'{orig_skew:.4f}', f'{std_skew:.4f}', '0.0000'],
        ['峰度', f'{orig_kurtosis:.4f}', f'{std_kurtosis:.4f}', '0.0000'],
        ['最小值', f'{original_data.min():.4f}', f'{standardized_data.min():.4f}', '-∞'],
        ['最大值', f'{original_data.max():.4f}', f'{standardized_data.max():.4f}', '+∞'],
        ['样本数', f'{len(original_data)}', f'{len(standardized_data)}', 'N/A']
    ]
    
    # 创建表格
    table = ax6.table(cellText=table_data, cellLoc='center', 
                      loc='center', colWidths=[0.15, 0.2, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置特定单元格颜色
    for i in range(1, len(table_data)):
        if i <= 4:  # 统计量行
            # 高亮接近理论值的单元格
            if i == 1:  # 均值行
                if abs(std_mean) < 0.01:
                    table[(i, 2)].set_facecolor('#90EE90')  # 浅绿色
            elif i == 2:  # 标准差行
                if abs(std_std - 1) < 0.01:
                    table[(i, 2)].set_facecolor('#90EE90')
    
    ax6.set_title('统计摘要', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细统计信息
    print("=" * 60)
    print(f"{title_prefix}分布验证结果:")
    print("=" * 60)
    print(f"原始数据: μ = {orig_mean:.6f}, σ = {orig_std:.6f}")
    print(f"标准化后: μ = {std_mean:.6f}, σ = {std_std:.6f}")
    print(f"理论标准正态: μ = 0.000000, σ = 1.000000")
    print("-" * 60)
    print(f"均值偏差: {abs(std_mean):.8f} (目标: 接近0)")
    print(f"标准差偏差: {abs(std_std - 1):.8f} (目标: 接近0)")
    print("-" * 60)
    
    # Shapiro-Wilk正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(standardized_data)
    print(f"Shapiro-Wilk正态性检验:")
    print(f"  统计量 W = {shapiro_stat:.6f}")
    print(f"  p值 = {shapiro_p:.6f}")
    
    if shapiro_p > 0.05:
        print("  结论: 不能拒绝正态性假设 (p > 0.05)")
    else:
        print("  结论: 拒绝正态性假设 (p ≤ 0.05)")
    
    print("=" * 60)
    
    return fig

def run_comprehensive_validation():
    """运行全面的验证"""
    distributions = [
        ('正态分布 N(100, 15²)', 'normal'),
        ('均匀分布 U[20, 180]', 'uniform'),
        ('指数分布 Exp(50)', 'exponential'),
        ('对数正态分布', 'log_normal')
    ]
    
    all_results = []
    
    for dist_name, dist_type in distributions:
        print(f"\n{'='*80}")
        print(f"验证 {dist_name}")
        print(f"{'='*80}")
        
        # 生成数据
        data = generate_test_data(10000, dist_type)
        
        # 标准化
        standardized_data, orig_mean, orig_std = standardize_data(data)
        
        # 分析并可视化
        fig = analyze_distribution(data, standardized_data, dist_name)
        
        # 保存结果
        all_results.append({
            'distribution': dist_name,
            'original_mean': orig_mean,
            'original_std': orig_std,
            'standardized_mean': np.mean(standardized_data),
            'standardized_std': np.std(standardized_data),
            'data': standardized_data
        })
        
        # 保存图表
        fig.savefig(f'{save_dir}/{dist_name}_标准化验证.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # 汇总比较
    print(f"\n{'='*80}")
    print("所有分布标准化结果汇总")
    print(f"{'='*80}")
    
    summary_data = []
    for result in all_results:
        summary_data.append([
            result['distribution'],
            f"{result['standardized_mean']:.8f}",
            f"{result['standardized_std']:.8f}",
            f"{abs(result['standardized_mean']):.8f}",
            f"{abs(result['standardized_std'] - 1):.8f}"
        ])
    
    # 创建汇总表格
    fig_summary, ax_summary = plt.subplots(figsize=(12, 6))
    ax_summary.axis('tight')
    ax_summary.axis('off')
    
    summary_table = ax_summary.table(
        cellText=summary_data,
        colLabels=['分布类型', '标准化均值', '标准化标准差', '|均值偏差|', '|标准差偏差-1|'],
        cellLoc='center',
        loc='center'
    )
    
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(11)
    summary_table.scale(1, 2)
    
    # 高亮最优结果
    mean_deviations = [float(row[3]) for row in summary_data]
    std_deviations = [float(row[4]) for row in summary_data]
    
    min_mean_idx = np.argmin(mean_deviations)
    min_std_idx = np.argmin(std_deviations)
    
    # 均值偏差最小的行
    summary_table[(min_mean_idx + 1, 2)].set_facecolor('#90EE90')
    # 标准差偏差最小的行
    summary_table[(min_std_idx + 1, 3)].set_facecolor('#ADD8E6')
    
    plt.title('标准化变换效果汇总', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    fig_summary.savefig(f'{save_dir}/标准化验证汇总.png', dpi=150, bbox_inches='tight')
    
    return all_results

def interactive_demo():
    """交互式演示"""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    if 'ipykernel' not in sys.modules:
        print("交互式演示需要Jupyter Notebook环境")
        return
    
    # 创建控件
    dist_selector = widgets.Dropdown(
        options=[
            ('正态分布', 'normal'),
            ('均匀分布', 'uniform'),
            ('指数分布', 'exponential'),
            ('对数正态分布', 'log_normal')
        ],
        value='normal',
        description='分布类型:'
    )
    
    sample_slider = widgets.IntSlider(
        value=1000,
        min=100,
        max=100000,
        step=100,
        description='样本数:'
    )
    
    run_button = widgets.Button(description="运行验证")
    output = widgets.Output()
    
    def on_button_click(b):
        with output:
            clear_output()
            data = generate_test_data(sample_slider.value, dist_selector.value)
            standardized_data, _, _ = standardize_data(data)
            dist_names = {
                'normal': '正态分布',
                'uniform': '均匀分布',
                'exponential': '指数分布',
                'log_normal': '对数正态分布'
            }
            analyze_distribution(data, standardized_data, dist_names[dist_selector.value])
    
    run_button.on_click(on_button_click)
    
    # 显示控件
    display(widgets.VBox([dist_selector, sample_slider, run_button, output]))

def mean_std_vis(input_data, save_back):
    """归一化验证"""
    
    # 标准化
    standardized_data, orig_mean, orig_std = standardize_data(input_data)
    
    # 分析并可视化
    fig = analyze_distribution(input_data, standardized_data, '')
    
    # 保存结果
    all_results={
        'original_mean': orig_mean,
        'original_std': orig_std,
        'standardized_mean': np.mean(standardized_data),
        'standardized_std': np.std(standardized_data),
        'data': standardized_data
    }
    
    # 保存图表
    fig.savefig(f'tmp/mean_std_IMG/标准化验证_{save_back}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return all_results

set_chinese() # 全局设置中文
if __name__ == "__main__":
    save_dir = 'tmp/mean_std_IMG'
    print("标准化变换验证程序")
    print("验证公式: z = (x - μ) / σ")
    print("目标: 标准化后数据应满足 μ ≈ 0, σ ≈ 1")
    print("=" * 60)
    
    # 运行全面验证
    results = run_comprehensive_validation()
    
    print("\n" + "="*60)
    print("验证结论:")
    print("="*60)
    print("1. (x-μ)/σ 变换确实能将数据的均值变为0，标准差变为1")
    print("2. 变换后的分布形状与原始分布相同（只是平移和缩放）")
    print("3. 只有原始分布是正态分布时，标准化后才是标准正态分布")
    print("4. 对于非正态分布，标准化只改变位置和尺度，不改变分布形状")
    print("="*60)