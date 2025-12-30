"""
方程求根的迭代法
使用不动点迭代法和牛顿迭代法求 f(x) = x^3 - 3x - 1 = 0 的根
包含丰富的可视化输出
"""

import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def f(x):
    """原函数 f(x) = x^3 - 3x - 1"""
    return x**3 - 3*x - 1


def df(x):
    """导数 f'(x) = 3x^2 - 3"""
    return 3*x**2 - 3


def fixed_point_iteration(x0, tol=1e-4, max_iter=100):
    """
    不动点迭代法
    
    将 f(x) = x^3 - 3x - 1 = 0 改写为 x = g(x)
    选择 g(x) = (x^3 - 1) / 3 或 g(x) = (3x + 1)^(1/3)
    
    参数:
        x0: 初始值
        tol: 容差（四位有效数字，相对误差约 0.5e-4）
        max_iter: 最大迭代次数
    
    返回:
        (根, 迭代次数, 迭代历史)
    """
    # 使用 g(x) = (3x + 1)^(1/3)
    def g(x):
        return (3*x + 1)**(1/3)
    
    x = x0
    history = [x0]
    iterations = 0
    
    for i in range(max_iter):
        x_new = g(x)
        iterations += 1
        history.append(x_new)
        
        # 检查收敛性（相对误差）
        if abs(x_new - x) < tol * max(abs(x_new), 1.0):
            return x_new, iterations, history
        
        x = x_new
    
    return x, iterations, history


def newton_iteration(x0, tol=1e-4, max_iter=100):
    """
    牛顿迭代法
    
    参数:
        x0: 初始值
        tol: 容差（四位有效数字，相对误差约 0.5e-4）
        max_iter: 最大迭代次数
    
    返回:
        (根, 迭代次数, 迭代历史, 函数值历史)
    """
    x = x0
    history = [x0]
    f_history = [f(x0)]
    iterations = 0
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-10:
            print("警告: 导数接近零，可能无法收敛")
            break
        
        x_new = x - fx / dfx
        iterations += 1
        history.append(x_new)
        f_history.append(f(x_new))
        
        # 检查收敛性（相对误差）
        if abs(x_new - x) < tol * max(abs(x_new), 1.0):
            return x_new, iterations, history, f_history
        
        x = x_new
    
    return x, iterations, history, f_history


def plot_function_and_root(x_range, exact_root, filename):
    """绘制函数图像和根的位置"""
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = np.array([f(xi) for xi in x])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制函数曲线
    ax.plot(x, y, 'b-', linewidth=2, label='f(x) = x³ - 3x - 1', zorder=1)
    
    # 绘制x轴
    ax.axhline(y=0, color='k', linewidth=0.8, linestyle='--', zorder=2)
    
    # 标注精确根
    ax.scatter([exact_root], [0], color='red', s=200, zorder=4,
              marker='*', label=f'精确根 x* = {exact_root:.8f}', 
              edgecolors='black', linewidths=2)
    ax.axvline(x=exact_root, color='red', linewidth=1.5, linestyle=':', 
              alpha=0.7, zorder=2)
    
    # 标注函数零点区域
    ax.fill_between(x, 0, y, where=(y >= 0), alpha=0.2, color='green', 
                    label='f(x) ≥ 0')
    ax.fill_between(x, 0, y, where=(y < 0), alpha=0.2, color='red', 
                    label='f(x) < 0')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('函数图像和根的位置', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_fixed_point_iteration(x0, history, exact_root, filename):
    """绘制不动点迭代过程"""
    # 定义g(x)
    def g(x):
        return (3*x + 1)**(1/3)
    
    x_min, x_max = min(min(history), exact_root) - 0.5, max(max(history), exact_root) + 0.5
    x_fine = np.linspace(x_min, x_max, 1000)
    y_g = np.array([g(xi) for xi in x_fine])
    y_line = x_fine  # y = x
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 不动点迭代的几何解释
    ax1 = axes[0]
    ax1.plot(x_fine, y_g, 'b-', linewidth=2, label='g(x) = (3x+1)^(1/3)', zorder=1)
    ax1.plot(x_fine, y_line, 'k--', linewidth=1.5, label='y = x', zorder=1)
    
    # 绘制迭代过程（阶梯图）
    for i in range(len(history) - 1):
        x_curr = history[i]
        x_next = history[i + 1]
        g_curr = g(x_curr)
        
        # 垂直线：从(x_curr, x_curr)到(x_curr, g(x_curr))
        ax1.plot([x_curr, x_curr], [x_curr, g_curr], 'r-', linewidth=1.5, alpha=0.6, zorder=2)
        # 水平线：从(x_curr, g(x_curr))到(g(x_curr), g(x_curr))
        ax1.plot([x_curr, g_curr], [g_curr, g_curr], 'r-', linewidth=1.5, alpha=0.6, zorder=2)
        
        # 标注迭代点
        if i < 5:  # 只标注前几个点
            ax1.annotate(f'x{i}', xy=(x_curr, x_curr), xytext=(5, 5),
                        textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 标注不动点（根）
    ax1.scatter([exact_root], [exact_root], color='red', s=200, zorder=4,
               marker='*', label=f'不动点 (根) x* = {exact_root:.8f}',
               edgecolors='black', linewidths=2)
    
    # 标注初始点
    ax1.scatter([x0], [x0], color='green', s=150, zorder=4,
               marker='o', label=f'初始点 x₀ = {x0}',
               edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('不动点迭代法几何解释', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 图2: 迭代序列收敛过程
    ax2 = axes[1]
    iterations = np.arange(len(history))
    ax2.plot(iterations, history, 'o-', linewidth=2, markersize=8, 
            color='steelblue', label='迭代序列', zorder=2)
    ax2.axhline(y=exact_root, color='red', linewidth=2, linestyle='--',
               label=f'精确根 x* = {exact_root:.8f}', zorder=1)
    
    # 添加数值标注（前几个点）
    for i in range(min(5, len(history))):
        ax2.annotate(f'{history[i]:.4f}', 
                    xy=(i, history[i]), xytext=(0, 10),
                    textcoords='offset points', fontsize=8,
                    ha='center', bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('x值', fontsize=12)
    ax2.set_title('不动点迭代法收敛过程', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_newton_iteration(x0, history, f_history, exact_root, filename):
    """绘制牛顿迭代过程"""
    x_min, x_max = min(min(history), exact_root) - 0.3, max(max(history), exact_root) + 0.3
    x_fine = np.linspace(x_min, x_max, 1000)
    y_fine = np.array([f(xi) for xi in x_fine])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 牛顿迭代的几何解释（切线法）
    ax1 = axes[0]
    ax1.plot(x_fine, y_fine, 'b-', linewidth=2, label='f(x) = x³ - 3x - 1', zorder=1)
    ax1.axhline(y=0, color='k', linewidth=1, linestyle='--', zorder=2)
    
    # 绘制前几次迭代的切线
    for i in range(min(5, len(history) - 1)):
        x_curr = history[i]
        fx_curr = f_history[i]
        dfx_curr = df(x_curr)
        
        if abs(dfx_curr) > 1e-10:
            # 切线方程: y = f'(x_curr)(x - x_curr) + f(x_curr)
            x_tangent = np.linspace(x_curr - 0.5, x_curr + 0.5, 100)
            y_tangent = dfx_curr * (x_tangent - x_curr) + fx_curr
            
            ax1.plot(x_tangent, y_tangent, 'r--', linewidth=1.5, alpha=0.6, zorder=2)
            ax1.scatter([x_curr], [fx_curr], color='green', s=100, zorder=4,
                       edgecolors='black', linewidths=1.5)
            
            # 标注
            if i < 3:
                ax1.annotate(f'x{i}', xy=(x_curr, fx_curr), xytext=(5, 10),
                           textcoords='offset points', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 标注根
    ax1.scatter([exact_root], [0], color='red', s=200, zorder=4,
               marker='*', label=f'精确根 x* = {exact_root:.8f}',
               edgecolors='black', linewidths=2)
    
    # 标注初始点
    ax1.scatter([x0], [f(x0)], color='green', s=150, zorder=4,
               marker='o', label=f'初始点 x₀ = {x0}',
               edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('牛顿迭代法几何解释（切线法）', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 迭代序列收敛过程
    ax2 = axes[1]
    iterations = np.arange(len(history))
    ax2.plot(iterations, history, 's-', linewidth=2, markersize=8,
            color='coral', label='迭代序列', zorder=2)
    ax2.axhline(y=exact_root, color='red', linewidth=2, linestyle='--',
               label=f'精确根 x* = {exact_root:.8f}', zorder=1)
    
    # 添加数值标注
    for i in range(min(5, len(history))):
        ax2.annotate(f'{history[i]:.4f}',
                    xy=(i, history[i]), xytext=(0, 10),
                    textcoords='offset points', fontsize=8,
                    ha='center', bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='lightblue', alpha=0.7))
    
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('x值', fontsize=12)
    ax2.set_title('牛顿迭代法收敛过程', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_convergence_comparison(hist1, hist2, exact_root, filename):
    """绘制两种方法的收敛性对比"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 迭代序列对比
    ax1 = axes[0]
    iter1 = np.arange(len(hist1))
    iter2 = np.arange(len(hist2))
    
    ax1.plot(iter1, hist1, 'o-', linewidth=2, markersize=8,
            color='steelblue', label='不动点迭代法', zorder=2)
    ax1.plot(iter2, hist2, 's-', linewidth=2, markersize=8,
            color='coral', label='牛顿迭代法', zorder=2)
    ax1.axhline(y=exact_root, color='red', linewidth=2, linestyle='--',
               label=f'精确根 x* = {exact_root:.8f}', zorder=1)
    
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('x值', fontsize=12)
    ax1.set_title('两种方法的迭代序列对比', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 误差随迭代次数的变化
    ax2 = axes[1]
    errors1 = [abs(x - exact_root) for x in hist1]
    errors2 = [abs(x - exact_root) for x in hist2]
    
    ax2.semilogy(iter1, errors1, 'o-', linewidth=2, markersize=8,
                color='steelblue', label='不动点迭代法误差', zorder=2)
    ax2.semilogy(iter2, errors2, 's-', linewidth=2, markersize=8,
                color='coral', label='牛顿迭代法误差', zorder=2)
    
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('绝对误差 |x - x*|', fontsize=12)
    ax2.set_title('误差随迭代次数的变化（对数坐标）', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_function_values(hist1, hist2, filename):
    """绘制函数值随迭代次数的变化"""
    f_hist1 = [f(x) for x in hist1]
    f_hist2 = [f(x) for x in hist2]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iter1 = np.arange(len(hist1))
    iter2 = np.arange(len(hist2))
    
    ax.semilogy(iter1, np.abs(f_hist1), 'o-', linewidth=2, markersize=8,
               color='steelblue', label='不动点迭代法 |f(x)|', zorder=2)
    ax.semilogy(iter2, np.abs(f_hist2), 's-', linewidth=2, markersize=8,
               color='coral', label='牛顿迭代法 |f(x)|', zorder=2)
    ax.axhline(y=1e-10, color='gray', linewidth=1, linestyle=':', 
              label='机器精度', zorder=1)
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('|f(x)|', fontsize=12)
    ax.set_title('函数值随迭代次数的变化（对数坐标）', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def main():
    x0 = 2.0
    exact_root = 1.87938524
    tol = 0.5e-4  # 四位有效数字的容差
    
    print("=" * 70)
    print("方程求根的迭代法")
    print("=" * 70)
    print(f"\n方程: f(x) = x^3 - 3x - 1 = 0")
    print(f"初始值: x0 = {x0}")
    print(f"精确根: x* = {exact_root}")
    print(f"容差: {tol:.2e} (四位有效数字)")
    
    # 方法1: 不动点迭代法
    print(f"\n{'='*70}")
    print("方法1: 不动点迭代法")
    print(f"{'='*70}")
    root1, iter1, hist1 = fixed_point_iteration(x0, tol)
    error1 = abs(root1 - exact_root)
    
    print(f"\n迭代过程:")
    for i, x in enumerate(hist1[:min(10, len(hist1))]):
        print(f"  第 {i} 次: x = {x:.10f}, f(x) = {f(x):.10e}")
    if len(hist1) > 10:
        print(f"  ... (共 {len(hist1)} 次迭代)")
    
    print(f"\n结果:")
    print(f"  近似根: {root1:.10f}")
    print(f"  迭代次数: {iter1}")
    print(f"  绝对误差: {error1:.10e}")
    print(f"  函数值: f({root1:.10f}) = {f(root1):.10e}")
    
    # 方法2: 牛顿迭代法
    print(f"\n{'='*70}")
    print("方法2: 牛顿迭代法")
    print(f"{'='*70}")
    root2, iter2, hist2, f_hist2 = newton_iteration(x0, tol)
    error2 = abs(root2 - exact_root)
    
    print(f"\n迭代过程:")
    for i, x in enumerate(hist2[:min(10, len(hist2))]):
        print(f"  第 {i} 次: x = {x:.10f}, f(x) = {f(x):.10e}")
    if len(hist2) > 10:
        print(f"  ... (共 {len(hist2)} 次迭代)")
    
    print(f"\n结果:")
    print(f"  近似根: {root2:.10f}")
    print(f"  迭代次数: {iter2}")
    print(f"  绝对误差: {error2:.10e}")
    print(f"  函数值: f({root2:.10f}) = {f(root2):.10e}")
    
    # 计算量对比
    print(f"\n{'='*70}")
    print("计算量对比")
    print(f"{'='*70}")
    print(f"\n{'方法':<20} {'迭代次数':<15} {'每次迭代计算量':<25} {'总计算量':<20}")
    print("-" * 70)
    print(f"{'不动点迭代法':<20} {iter1:<15} {'1次函数求值':<25} {iter1:<20}")
    print(f"{'牛顿迭代法':<20} {iter2:<15} {'1次函数+1次导数求值':<25} {iter2 * 2:<20}")
    
    print(f"\n结论:")
    print(f"  - 牛顿迭代法收敛更快（迭代次数更少）")
    print(f"  - 但每次迭代需要计算函数值和导数值")
    print(f"  - 总体而言，牛顿迭代法通常更高效")
    
    # 可视化
    print(f"\n{'='*70}")
    print("生成可视化图像...")
    print("-" * 70)
    
    # 确定绘图范围
    x_range = [min(min(hist1), min(hist2), exact_root) - 0.5,
               max(max(hist1), max(hist2), exact_root) + 0.5]
    
    # 1. 函数图像和根的位置
    plot_function_and_root(x_range, exact_root, 'root_finding_function.png')
    
    # 2. 不动点迭代过程
    plot_fixed_point_iteration(x0, hist1, exact_root, 'root_finding_fixed_point.png')
    
    # 3. 牛顿迭代过程
    plot_newton_iteration(x0, hist2, f_hist2, exact_root, 'root_finding_newton.png')
    
    # 4. 两种方法的收敛性对比
    plot_convergence_comparison(hist1, hist2, exact_root, 'root_finding_convergence.png')
    
    # 5. 函数值随迭代次数的变化
    plot_function_values(hist1, hist2, 'root_finding_function_values.png')
    
    print("\n所有可视化图像已生成完成！")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

