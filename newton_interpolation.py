"""
牛顿插值法
使用给定的数据点进行五次插值，并计算特定点的函数值
包含丰富的可视化输出
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def compute_divided_differences(x_data, y_data):
    """
    计算差商表
    
    参数:
        x_data: 插值节点的x坐标
        y_data: 插值节点的y坐标
    
    返回:
        差商表
    """
    n = len(x_data)
    f = np.zeros((n, n))
    f[:, 0] = y_data
    
    # 计算各阶差商
    for j in range(1, n):
        for i in range(n - j):
            f[i, j] = (f[i + 1, j - 1] - f[i, j - 1]) / (x_data[i + j] - x_data[i])
    
    return f


def newton_interpolation(x_data, y_data, x):
    """
    计算牛顿插值多项式在点x处的值
    
    参数:
        x_data: 插值节点的x坐标
        y_data: 插值节点的y坐标
        x: 待求值的点
    
    返回:
        插值多项式在x处的值
    """
    n = len(x_data)
    # 计算差商表
    f = compute_divided_differences(x_data, y_data)
    
    # 计算插值多项式的值
    result = f[0, 0]
    product = 1.0
    for i in range(1, n):
        product *= (x - x_data[i - 1])
        result += f[0, i] * product
    
    return result


def plot_interpolation_curve(x_data, y_data, x_eval, y_eval, filename):
    """绘制插值曲线和节点"""
    # 生成密集的插值点用于绘制平滑曲线
    x_min, x_max = x_data.min(), x_data.max()
    x_fine = np.linspace(x_min, x_max, 1000)
    y_fine = np.array([newton_interpolation(x_data, y_data, x) for x in x_fine])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 插值曲线和节点
    ax1 = axes[0]
    ax1.plot(x_fine, y_fine, 'b-', linewidth=2, label='牛顿插值多项式', zorder=1)
    ax1.scatter(x_data, y_data, color='red', s=150, zorder=3, 
               label='插值节点', edgecolors='black', linewidths=2)
    
    # 标注节点
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        ax1.annotate(f'({x:.2f}, {y:.5f})', 
                    xy=(x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 标注待求点
    ax1.scatter(x_eval, y_eval, color='green', s=200, zorder=4,
               marker='*', label='待求点', edgecolors='black', linewidths=2)
    for x, y in zip(x_eval, y_eval):
        ax1.annotate(f'({x:.2f}, {y:.8f})', 
                    xy=(x, y), xytext=(5, -20), 
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('牛顿插值多项式曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 局部放大（显示待求点区域）
    ax2 = axes[1]
    ax2.plot(x_fine, y_fine, 'b-', linewidth=2, label='牛顿插值多项式', zorder=1)
    ax2.scatter(x_data, y_data, color='red', s=100, zorder=3, 
               label='插值节点', edgecolors='black', linewidths=1.5)
    ax2.scatter(x_eval, y_eval, color='green', s=150, zorder=4,
               marker='*', label='待求点', edgecolors='black', linewidths=1.5)
    
    # 设置局部放大区域
    x_eval_min, x_eval_max = min(x_eval), max(x_eval)
    margin = (x_eval_max - x_eval_min) * 0.3
    ax2.set_xlim(x_eval_min - margin, x_eval_max + margin)
    y_eval_min, y_eval_max = min(y_eval), max(y_eval)
    y_margin = (y_eval_max - y_eval_min) * 0.3
    ax2.set_ylim(y_eval_min - y_margin, y_eval_max + y_margin)
    
    # 标注待求点
    for x, y in zip(x_eval, y_eval):
        ax2.annotate(f'({x:.2f}\n{y:.8f})', 
                    xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', fontsize=9, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('待求点区域局部放大', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_divided_differences_table(x_data, y_data, f_table, filename):
    """绘制差商表"""
    n = len(x_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格数据
    table_data = []
    headers = ['x', 'f(x)', '一阶差商', '二阶差商', '三阶差商', '四阶差商', '五阶差商']
    
    for i in range(n):
        row = [f'{x_data[i]:.2f}', f'{y_data[i]:.5f}']
        for j in range(1, n):
            if i < n - j:
                row.append(f'{f_table[i, j]:.6f}')
            else:
                row.append('')
        table_data.append(row)
    
    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置第一列（x值）样式
    for i in range(1, n + 1):
        table[(i, 0)].set_facecolor('#E3F2FD')
        table[(i, 0)].set_text_props(weight='bold')
    
    # 设置第二列（f(x)值）样式
    for i in range(1, n + 1):
        table[(i, 1)].set_facecolor('#FFF3E0')
    
    # 设置差商列样式（交替颜色）
    for j in range(2, len(headers)):
        for i in range(1, n + 1):
            if (i + j) % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    ax.set_title('牛顿插值差商表', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_evaluation_points(x_data, y_data, x_eval, y_eval, filename):
    """绘制待求点的详细分析"""
    n_eval = len(x_eval)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 待求点的位置和值
    ax1 = axes[0]
    indices = np.arange(1, n_eval + 1)
    
    # 绘制柱状图
    bars = ax1.bar(indices, y_eval, color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for i, (bar, x, y) in enumerate(zip(bars, x_eval, y_eval)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'x={x:.2f}\ny={y:.8f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('待求点序号', fontsize=12)
    ax1.set_ylabel('插值函数值', fontsize=12)
    ax1.set_title('待求点的插值结果', fontsize=14, fontweight='bold')
    ax1.set_xticks(indices)
    ax1.set_xticklabels([f'点 {i+1}' for i in range(n_eval)])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 图2: x值和y值的散点图
    ax2 = axes[1]
    scatter = ax2.scatter(x_eval, y_eval, s=300, c=indices, 
                         cmap='viridis', edgecolors='black', linewidths=2, zorder=3)
    
    # 添加标签
    for i, (x, y) in enumerate(zip(x_eval, y_eval)):
        ax2.annotate(f'点{i+1}\n({x:.2f}, {y:.8f})', 
                    xy=(x, y), xytext=(10, 10), 
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 绘制插值节点作为参考
    ax2.scatter(x_data, y_data, s=100, color='red', marker='s', 
               label='插值节点', edgecolors='black', linewidths=1.5, zorder=2, alpha=0.6)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y (插值函数值)', fontsize=12)
    ax2.set_title('待求点在坐标系中的位置', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='点序号')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def main():
    # 给定的数据点
    x_data = np.array([0.30, 0.42, 0.50, 0.58, 0.66, 0.72])
    y_data = np.array([1.04403, 1.08462, 1.11803, 1.15603, 1.19817, 1.23223])
    
    print("=" * 70)
    print("牛顿插值法 - 带可视化输出")
    print("=" * 70)
    print(f"\n插值节点:")
    for i in range(len(x_data)):
        print(f"  x_{i} = {x_data[i]:.2f}, y_{i} = {y_data[i]:.5f}")
    
    # 计算差商表
    f_table = compute_divided_differences(x_data, y_data)
    
    # 待求值的点
    x_values = np.array([0.46, 0.55, 0.60])
    y_values = np.array([newton_interpolation(x_data, y_data, x) for x in x_values])
    
    print(f"\n计算结果:")
    print("-" * 70)
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        print(f"  点 {i+1}: x = {x:.2f}  =>  y ≈ {y:.8f}")
    
    # 可视化
    print(f"\n{'='*70}")
    print("生成可视化图像...")
    print("-" * 70)
    
    # 1. 插值曲线和节点
    plot_interpolation_curve(x_data, y_data, x_values, y_values, 
                            'newton_interpolation_curve.png')
    
    # 2. 差商表
    plot_divided_differences_table(x_data, y_data, f_table, 
                                   'newton_divided_differences_table.png')
    
    # 3. 待求点分析
    plot_evaluation_points(x_data, y_data, x_values, y_values, 
                          'newton_evaluation_points.png')
    
    # 打印差商表
    print(f"\n{'='*70}")
    print("差商表")
    print("-" * 70)
    print(f"{'节点':<10} {'函数值':<15} {'一阶差商':<15} {'二阶差商':<15} {'三阶差商':<15} {'四阶差商':<15} {'五阶差商':<15}")
    print("-" * 70)
    for i in range(len(x_data)):
        row = f"{x_data[i]:<10.2f} {y_data[i]:<15.5f}"
        for j in range(1, len(x_data)):
            if i < len(x_data) - j:
                row += f"{f_table[i, j]:<15.6f}"
            else:
                row += f"{'':<15}"
        print(row)
    
    print("\n所有可视化图像已生成完成！")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

