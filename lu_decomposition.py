"""
利用直接三角分解法（LU分解）求方程组的解 Ax=b
包含丰富的可视化输出
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def lu_decomposition(A):
    """
    LU分解（Doolittle分解）
    
    参数:
        A: 系数矩阵 (n x n)
    
    返回:
        L: 下三角矩阵
        U: 上三角矩阵
    """
    n = len(A)
    L = np.eye(n)  # 单位下三角矩阵
    U = np.zeros((n, n))
    
    for i in range(n):
        # 计算U的第i行
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        
        # 计算L的第i列
        for j in range(i + 1, n):
            if abs(U[i, i]) < 1e-10:
                raise ValueError("矩阵奇异，无法进行LU分解")
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
    
    return L, U


def forward_substitution(L, b):
    """
    前向替换：求解 Ly = b
    
    参数:
        L: 下三角矩阵
        b: 右端向量
    
    返回:
        y: 解向量
    """
    n = len(b)
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    return y


def backward_substitution(U, y):
    """
    后向替换：求解 Ux = y
    
    参数:
        U: 上三角矩阵
        y: 右端向量
    
    返回:
        x: 解向量
    """
    n = len(y)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-10:
            raise ValueError("矩阵奇异，无法求解")
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x


def solve_lu(A, b):
    """
    使用LU分解求解线性方程组 Ax = b
    
    参数:
        A: 系数矩阵
        b: 右端向量
    
    返回:
        x: 解向量, L: 下三角矩阵, U: 上三角矩阵, y: 中间向量
    """
    # LU分解
    L, U = lu_decomposition(A)
    
    # 前向替换：Ly = b
    y = forward_substitution(L, b)
    
    # 后向替换：Ux = y
    x = backward_substitution(U, y)
    
    return x, L, U, y


def plot_matrix_heatmap(matrices, titles, filename):
    """绘制矩阵热力图"""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    
    if n == 1:
        axes = [axes]
    
    for idx, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axes[idx]
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
        
        # 添加数值标注
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('列索引', fontsize=12)
        ax.set_ylabel('行索引', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_solution_comparison(x_computed, x_exact, filename):
    """绘制解的对比图"""
    n = len(x_computed)
    indices = np.arange(1, n + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 图1: 解的对比柱状图
    ax1 = axes[0]
    x_pos = np.arange(n)
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, x_computed, width, label='计算解', 
                    color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, x_exact, width, label='精确解', 
                    color='coral', alpha=0.8)
    
    # 添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{height1:.2f}', ha='center', va='bottom', fontsize=9)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{height2:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('变量索引', fontsize=12)
    ax1.set_ylabel('解的值', fontsize=12)
    ax1.set_title('计算解与精确解对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'x_{i+1}' for i in range(n)])
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    
    # 图2: 误差分析
    ax2 = axes[1]
    errors = np.abs(x_computed - x_exact)
    bars = ax2.bar(indices, errors, color='red', alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for i, (bar, err) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.2e}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('变量索引', fontsize=12)
    ax2.set_ylabel('绝对误差', fontsize=12)
    ax2.set_title('绝对误差分析', fontsize=14, fontweight='bold')
    ax2.set_xticks(indices)
    ax2.set_xticklabels([f'x_{i+1}' for i in range(n)])
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y', which='both')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_residual_analysis(A, x, b, filename):
    """绘制残差分析图"""
    residual = A @ x - b
    n = len(residual)
    indices = np.arange(1, n + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 图1: 残差向量
    ax1 = axes[0]
    colors = ['red' if r > 0 else 'blue' for r in residual]
    bars = ax1.bar(indices, residual, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for i, (bar, res) in enumerate(zip(bars, residual)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{res:.2e}', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=9)
    
    ax1.set_xlabel('方程索引', fontsize=12)
    ax1.set_ylabel('残差值 (Ax - b)', fontsize=12)
    ax1.set_title('残差向量分析', fontsize=14, fontweight='bold')
    ax1.set_xticks(indices)
    ax1.set_xticklabels([f'方程 {i+1}' for i in range(n)])
    ax1.axhline(y=0, color='k', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 图2: 残差范数（对数坐标）
    ax2 = axes[1]
    residual_norm = np.linalg.norm(residual)
    ax2.bar([1], [residual_norm], color='purple', alpha=0.7, edgecolor='black', width=0.5)
    ax2.text(1, residual_norm, f'{residual_norm:.2e}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.set_ylabel('残差范数 ||Ax - b||', fontsize=12)
    ax2.set_title('残差范数', fontsize=14, fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['||Ax - b||'])
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y', which='both')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_lu_decomposition_process(A, L, U, filename):
    """绘制LU分解过程的可视化"""
    n = A.shape[0]
    
    fig = plt.figure(figsize=(16, 6))
    
    # 创建子图
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.3], hspace=0.3)
    
    # 图1: 原始矩阵 A
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(A, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f'{A[i, j]:.1f}', ha="center", va="center", 
                    color="black", fontsize=11, fontweight='bold')
    ax1.set_title('原始矩阵 A', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('列索引', fontsize=11)
    ax1.set_ylabel('行索引', fontsize=11)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 图2: 下三角矩阵 L
    ax2 = fig.add_subplot(gs[1])
    # 创建掩码，只显示下三角部分
    L_mask = np.triu(np.ones_like(L, dtype=bool), k=1)
    L_display = np.ma.masked_array(L, L_mask)
    im2 = ax2.imshow(L_display, cmap='Greens', aspect='auto', interpolation='nearest', 
                     vmin=0, vmax=np.max(L))
    for i in range(n):
        for j in range(n):
            if j <= i:
                ax2.text(j, i, f'{L[i, j]:.3f}', ha="center", va="center", 
                        color="black", fontsize=10, fontweight='bold')
            else:
                ax2.text(j, i, '0', ha="center", va="center", 
                        color="gray", fontsize=10, alpha=0.5)
    ax2.set_title('下三角矩阵 L', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('列索引', fontsize=11)
    ax2.set_ylabel('行索引', fontsize=11)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 图3: 上三角矩阵 U
    ax3 = fig.add_subplot(gs[2])
    # 创建掩码，只显示上三角部分
    U_mask = np.tril(np.ones_like(U, dtype=bool), k=-1)
    U_display = np.ma.masked_array(U, U_mask)
    im3 = ax3.imshow(U_display, cmap='Oranges', aspect='auto', interpolation='nearest',
                     vmin=0, vmax=np.max(U))
    for i in range(n):
        for j in range(n):
            if j >= i:
                ax3.text(j, i, f'{U[i, j]:.2f}', ha="center", va="center", 
                        color="black", fontsize=10, fontweight='bold')
            else:
                ax3.text(j, i, '0', ha="center", va="center", 
                        color="gray", fontsize=10, alpha=0.5)
    ax3.set_title('上三角矩阵 U', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel('列索引', fontsize=11)
    ax3.set_ylabel('行索引', fontsize=11)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 图4: 验证 L @ U = A
    ax4 = fig.add_subplot(gs[3])
    LU_product = L @ U
    verification = np.allclose(LU_product, A, atol=1e-10)
    
    # 显示验证结果
    ax4.axis('off')
    result_text = "✓ 验证通过\nL @ U = A" if verification else "✗ 验证失败"
    color = 'green' if verification else 'red'
    ax4.text(0.5, 0.5, result_text, ha='center', va='center', 
            fontsize=14, fontweight='bold', color=color,
            transform=ax4.transAxes, bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def plot_solving_process(L, U, b, y, x, filename):
    """绘制求解过程的可视化"""
    n = len(b)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 前向替换过程 Ly = b
    ax1 = axes[0, 0]
    indices = np.arange(1, n + 1)
    ax1.bar(indices - 0.2, b, 0.4, label='右端向量 b', color='steelblue', alpha=0.8)
    ax1.bar(indices + 0.2, y, 0.4, label='中间向量 y', color='green', alpha=0.8)
    ax1.set_xlabel('索引', fontsize=11)
    ax1.set_ylabel('值', fontsize=11)
    ax1.set_title('前向替换: Ly = b', fontsize=12, fontweight='bold')
    ax1.set_xticks(indices)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bi, yi) in enumerate(zip(b, y)):
        ax1.text(i+1-0.2, bi, f'{bi:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i+1+0.2, yi, f'{yi:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 图2: 后向替换过程 Ux = y
    ax2 = axes[0, 1]
    ax2.bar(indices - 0.2, y, 0.4, label='中间向量 y', color='green', alpha=0.8)
    ax2.bar(indices + 0.2, x, 0.4, label='解向量 x', color='coral', alpha=0.8)
    ax2.set_xlabel('索引', fontsize=11)
    ax2.set_ylabel('值', fontsize=11)
    ax2.set_title('后向替换: Ux = y', fontsize=12, fontweight='bold')
    ax2.set_xticks(indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (yi, xi) in enumerate(zip(y, x)):
        ax2.text(i+1-0.2, yi, f'{yi:.2f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i+1+0.2, xi, f'{xi:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 图3: L矩阵结构
    ax3 = axes[1, 0]
    L_display = np.ma.masked_array(L, np.triu(np.ones_like(L, dtype=bool), k=1))
    im3 = ax3.imshow(L_display, cmap='Greens', aspect='auto', interpolation='nearest')
    for i in range(n):
        for j in range(n):
            if j <= i:
                ax3.text(j, i, f'{L[i, j]:.3f}', ha="center", va="center", 
                        color="black", fontsize=9)
    ax3.set_title('下三角矩阵 L', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 图4: U矩阵结构
    ax4 = axes[1, 1]
    U_display = np.ma.masked_array(U, np.tril(np.ones_like(U, dtype=bool), k=-1))
    im4 = ax4.imshow(U_display, cmap='Oranges', aspect='auto', interpolation='nearest')
    for i in range(n):
        for j in range(n):
            if j >= i:
                ax4.text(j, i, f'{U[i, j]:.2f}', ha="center", va="center", 
                        color="black", fontsize=9)
    ax4.set_title('上三角矩阵 U', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存图像: {filename}")
    plt.close()


def main():
    # 给定的系数矩阵和右端向量
    A = np.array([
        [10, 7, 8, 7],
        [7, 5, 6, 5],
        [8, 6, 10, 9],
        [7, 5, 9, 10]
    ], dtype=float)
    
    b = np.array([10, 8, 6, 7], dtype=float)
    
    # 精确解
    exact_solution = np.array([-60, 102, -27, 16], dtype=float)
    
    print("=" * 70)
    print("利用直接三角分解法（LU分解）求方程组的解")
    print("=" * 70)
    
    print(f"\n系数矩阵 A:")
    print(A)
    
    print(f"\n右端向量 b:")
    print(b)
    
    # 求解
    try:
        x, L, U, y = solve_lu(A, b)
        
        print(f"\n{'='*70}")
        print("LU分解结果")
        print(f"{'='*70}")
        
        print(f"\n下三角矩阵 L:")
        print(L)
        
        print(f"\n上三角矩阵 U:")
        print(U)
        
        print(f"\n验证: L @ U =")
        print(L @ U)
        
        print(f"\n中间向量 y (Ly = b 的解):")
        print(y)
        
        print(f"\n{'='*70}")
        print("求解结果")
        print(f"{'='*70}")
        
        print(f"\n计算得到的解 x:")
        for i, xi in enumerate(x):
            print(f"  x_{i+1} = {xi:.10f}")
        
        print(f"\n精确解:")
        for i, xi in enumerate(exact_solution):
            print(f"  x_{i+1} = {xi:.10f}")
        
        # 计算误差
        error = np.abs(x - exact_solution)
        print(f"\n绝对误差:")
        for i, err in enumerate(error):
            print(f"  |x_{i+1} - x*_{i+1}| = {err:.10e}")
        
        print(f"\n最大绝对误差: {np.max(error):.10e}")
        print(f"相对误差: {np.linalg.norm(x - exact_solution) / np.linalg.norm(exact_solution):.10e}")
        
        # 验证解
        residual = A @ x - b
        print(f"\n残差向量 (Ax - b):")
        print(residual)
        print(f"残差范数: {np.linalg.norm(residual):.10e}")
        
        # 可视化
        print(f"\n{'='*70}")
        print("生成可视化图像...")
        print("-" * 70)
        
        # 1. LU分解过程可视化
        plot_lu_decomposition_process(A, L, U, 'lu_decomposition_process.png')
        
        # 2. 矩阵热力图
        plot_matrix_heatmap([A, L, U], 
                           ['系数矩阵 A', '下三角矩阵 L', '上三角矩阵 U'],
                           'lu_matrices_heatmap.png')
        
        # 3. 求解过程可视化
        plot_solving_process(L, U, b, y, x, 'lu_solving_process.png')
        
        # 4. 解的对比
        plot_solution_comparison(x, exact_solution, 'lu_solution_comparison.png')
        
        # 5. 残差分析
        plot_residual_analysis(A, x, b, 'lu_residual_analysis.png')
        
        print("\n所有可视化图像已生成完成！")
        
    except ValueError as e:
        print(f"\n错误: {e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

