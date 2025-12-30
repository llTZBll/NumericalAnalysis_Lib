"""
复化求积法
使用复化梯形公式和复化辛卜生公式计算 f(x) = sin(x)/x 的积分
"""

import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def f(x):
    """被积函数 f(x) = sin(x)/x"""
    if abs(x) < 1e-10:  # 处理 x=0 的情况
        return 1.0
    return math.sin(x) / x


def composite_trapezoidal(a, b, n):
    """
    复化梯形公式
    
    参数:
        a: 积分下限
        b: 积分上限
        n: 区间等分数
    
    返回:
        积分近似值, 节点坐标, 节点函数值
    """
    h = (b - a) / n
    x_nodes = np.linspace(a, b, n + 1)
    y_nodes = np.array([f(x) for x in x_nodes])
    
    result = (y_nodes[0] + y_nodes[-1]) / 2.0
    result += np.sum(y_nodes[1:-1])
    result *= h
    
    return result, x_nodes, y_nodes


def composite_simpson(a, b, n):
    """
    复化辛卜生公式
    
    参数:
        a: 积分下限
        b: 积分上限
        n: 区间等分数（必须是偶数）
    
    返回:
        积分近似值, 节点坐标, 节点函数值
    """
    if n % 2 != 0:
        n += 1  # 确保n是偶数
    
    h = (b - a) / n
    x_nodes = np.linspace(a, b, n + 1)
    y_nodes = np.array([f(x) for x in x_nodes])
    
    result = y_nodes[0] + y_nodes[-1]
    
    # 奇数节点（系数为4）
    result += 4 * np.sum(y_nodes[1:-1:2])
    
    # 偶数节点（系数为2）
    result += 2 * np.sum(y_nodes[2:-1:2])
    
    result *= h / 3.0
    return result, x_nodes, y_nodes


def plot_function_and_approximation(a, b, n_trap, n_simp):
    """绘制函数图像和积分近似区域"""
    # 创建精细的函数曲线
    x_fine = np.linspace(a, b, 1000)
    y_fine = np.array([f(x) for x in x_fine])
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 复化梯形公式
    ax1 = axes[0]
    ax1.plot(x_fine, y_fine, 'b-', linewidth=2, label='f(x) = sin(x)/x')
    
    # 计算梯形近似
    trap_val, x_trap, y_trap = composite_trapezoidal(a, b, n_trap)
    
    # 绘制梯形区域
    h_trap = (b - a) / n_trap
    for i in range(n_trap):
        x1 = x_trap[i]
        x2 = x_trap[i + 1]
        y1 = y_trap[i]
        y2 = y_trap[i + 1]
        
        # 绘制梯形
        trap_x = [x1, x2, x2, x1, x1]
        trap_y = [0, 0, y2, y1, 0]
        ax1.fill(trap_x, trap_y, alpha=0.3, color='green', edgecolor='green', linewidth=1)
    
    # 绘制节点
    ax1.plot(x_trap, y_trap, 'ro', markersize=6, label=f'节点 (n={n_trap})')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=a, color='k', linewidth=0.5, linestyle='--')
    ax1.axvline(x=b, color='k', linewidth=0.5, linestyle='--')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title(f'复化梯形公式 (n={n_trap}, 积分值≈{trap_val:.8f})', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 复化辛卜生公式
    ax2 = axes[1]
    ax2.plot(x_fine, y_fine, 'b-', linewidth=2, label='f(x) = sin(x)/x')
    
    # 计算辛卜生近似
    simp_val, x_simp, y_simp = composite_simpson(a, b, n_simp)
    
    # 绘制辛卜生抛物线区域
    h_simp = (b - a) / n_simp
    for i in range(0, n_simp, 2):
        x0 = x_simp[i]
        x1 = x_simp[i + 1]
        x2 = x_simp[i + 2]
        y0 = y_simp[i]
        y1 = y_simp[i + 1]
        y2 = y_simp[i + 2]
        
        # 计算抛物线系数 (通过三点确定抛物线)
        # P(x) = ax^2 + bx + c
        A = np.array([[x0**2, x0, 1],
                      [x1**2, x1, 1],
                      [x2**2, x2, 1]])
        b_vec = np.array([y0, y1, y2])
        coeffs = np.linalg.solve(A, b_vec)
        
        # 绘制抛物线区域
        x_parabola = np.linspace(x0, x2, 100)
        y_parabola = coeffs[0] * x_parabola**2 + coeffs[1] * x_parabola + coeffs[2]
        ax2.fill_between(x_parabola, 0, y_parabola, alpha=0.3, color='orange', edgecolor='orange')
    
    # 绘制节点
    ax2.plot(x_simp, y_simp, 'ro', markersize=6, label=f'节点 (n={n_simp})')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=a, color='k', linewidth=0.5, linestyle='--')
    ax2.axvline(x=b, color='k', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('f(x)', fontsize=12)
    ax2.set_title(f'复化辛卜生公式 (n={n_simp}, 积分值≈{simp_val:.8f})', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('composite_quadrature_approximation.png', dpi=300, bbox_inches='tight')
    print("\n✓ 已保存图像: composite_quadrature_approximation.png")
    plt.close()


def plot_convergence_analysis(a, b, n_values):
    """绘制收敛性分析图"""
    
    trap_values = []
    simp_values = []
    trap_errors = []
    simp_errors = []
    
    prev_trap = None
    prev_simp = None
    
    for n in n_values:
        trap_val, _, _ = composite_trapezoidal(a, b, n)
        simp_val, _, _ = composite_simpson(a, b, n)
        
        trap_values.append(trap_val)
        simp_values.append(simp_val)
        
        if prev_trap is not None:
            trap_errors.append(abs(trap_val - prev_trap))
        else:
            trap_errors.append(None)
        
        if prev_simp is not None:
            simp_errors.append(abs(simp_val - prev_simp))
        else:
            simp_errors.append(None)
        
        prev_trap = trap_val
        prev_simp = simp_val
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 图1: 积分值随n的变化
    ax1 = axes[0]
    ax1.plot(n_values, trap_values, 'o-', linewidth=2, markersize=8, label='复化梯形公式', color='green')
    ax1.plot(n_values, simp_values, 's-', linewidth=2, markersize=8, label='复化辛卜生公式', color='orange')
    ax1.set_xlabel('区间等分数 n', fontsize=12)
    ax1.set_ylabel('积分值', fontsize=12)
    ax1.set_title('积分值随区间等分数的变化', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # 图2: 误差随n的变化
    ax2 = axes[1]
    trap_err_plot = [e for e in trap_errors if e is not None]
    simp_err_plot = [e for e in simp_errors if e is not None]
    n_trap_err = n_values[1:]
    n_simp_err = n_values[1:]
    
    ax2.loglog(n_trap_err, trap_err_plot, 'o-', linewidth=2, markersize=8, label='复化梯形公式误差', color='green')
    ax2.loglog(n_simp_err, simp_err_plot, 's-', linewidth=2, markersize=8, label='复化辛卜生公式误差', color='orange')
    ax2.set_xlabel('区间等分数 n', fontsize=12)
    ax2.set_ylabel('误差估计', fontsize=12)
    ax2.set_title('误差随区间等分数的变化（对数坐标）', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('composite_quadrature_convergence.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存图像: composite_quadrature_convergence.png")
    plt.close()


def main():
    # 积分区间（需要根据题目要求设置，这里假设是 [0, 1]）
    # 注意：如果包含0，需要特殊处理
    a = 0.0
    b = 1.0
    
    print("=" * 70)
    print("复化求积法 - 带可视化输出")
    print("=" * 70)
    print(f"\n被积函数: f(x) = sin(x)/x")
    print(f"积分区间: [{a}, {b}]")
    
    # 使用不同的区间等分数进行测试
    n_values = [4, 8, 16, 32, 64]
    
    print(f"\n{'='*70}")
    print("复化梯形公式:")
    print("-" * 70)
    print(f"{'区间等分数':<12} {'积分值':<20} {'误差估计':<20}")
    print("-" * 70)
    
    trap_results = []
    prev_trap = None
    for n in n_values:
        trap, _, _ = composite_trapezoidal(a, b, n)
        trap_results.append(trap)
        if prev_trap is not None:
            error_est = abs(trap - prev_trap)
            print(f"{n:<12} {trap:<20.10f} {error_est:<20.10e}")
        else:
            print(f"{n:<12} {trap:<20.10f} {'-':<20}")
        prev_trap = trap
    
    print(f"\n{'='*70}")
    print("复化辛卜生公式:")
    print("-" * 70)
    print(f"{'区间等分数':<12} {'积分值':<20} {'误差估计':<20}")
    print("-" * 70)
    
    simp_results = []
    prev_simp = None
    for n in n_values:
        simp, _, _ = composite_simpson(a, b, n)
        simp_results.append(simp)
        if prev_simp is not None:
            error_est = abs(simp - prev_simp)
            print(f"{n:<12} {simp:<20.10f} {error_est:<20.10e}")
        else:
            print(f"{n:<12} {simp:<20.10f} {'-':<20}")
        prev_simp = simp
    
    # 精度比较
    print(f"\n{'='*70}")
    print("精度比较 (n = {})".format(n_values[-1]))
    print("-" * 70)
    final_trap = trap_results[-1]
    final_simp = simp_results[-1]
    print(f"复化梯形公式: {final_trap:.10f}")
    print(f"复化辛卜生公式: {final_simp:.10f}")
    print(f"差值: {abs(final_simp - final_trap):.10e}")
    print(f"\n结论: 复化辛卜生公式通常具有更高的精度（代数精度为3阶）")
    
    # 可视化
    print(f"\n{'='*70}")
    print("生成可视化图像...")
    print("-" * 70)
    

    # 绘制函数和近似区域
    plot_function_and_approximation(a, b, n_values[0], n_values[0])
    # 绘制收敛性分析
    plot_convergence_analysis(a, b, n_values)
    
    # 详细数据表格
    print(f"\n{'='*70}")
    print("详细数据对比表")
    print("-" * 70)
    print(f"{'n':<8} {'梯形公式':<20} {'辛卜生公式':<20} {'差值':<20}")
    print("-" * 70)
    for i, n in enumerate(n_values):
        diff = abs(trap_results[i] - simp_results[i])
        print(f"{n:<8} {trap_results[i]:<20.10f} {simp_results[i]:<20.10f} {diff:<20.10e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

