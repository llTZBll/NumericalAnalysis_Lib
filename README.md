# 《数值分析》实验

***注：出与整洁考虑，所有源代码统一放置在了实验报告的第三章。***

## 复化求积

### 题目描述

$$
f(x) = \frac{\sin x}{x}
$$

分别用复化梯形公式和复化辛卜生公式计算，并判断精度。

### 实验目的

1. 掌握复化梯形公式和复化辛普森（辛卜生）公式的基本原理与实现方法，理解数值积分中“复化”思想的核心——将积分区间等分后对每个子区间应用基本求积公式，以提升积分精度。
2. 针对被积函数 $f(x) = \frac{\sin x}{x}$（需特殊处理 $x=0$ 处的连续性，$\lim\limits_{x \to 0}\frac{\sin x}{x}=1$），实现两种复化求积算法，对比不同区间等分数下的积分结果。
3. 分析复化梯形公式和复化辛普森公式的收敛特性与精度差异，验证辛普森公式更高的代数精度优势。
4. 通过可视化手段直观展示两种求积方法的近似过程，加深对数值积分几何意义的理解。


### 算法描述

#### 复化梯形公式

##### 核心原理

将积分区间 $[a,b]$ 等分为 $n$ 个子区间，每个子区间宽度 $h = \frac{b-a}{n}$，节点为 $x_k = a + kh$（$k=0,1,...,n$）。对每个子区间 $[x_k, x_{k+1}]$ 应用梯形公式，累加所有子区间的积分近似值，得到整体公式：
$$
\int_a^b f(x)dx \approx h\left(\frac{f(x_0) + f(x_n)}{2} + \sum_{k=1}^{n-1}f(x_k)\right)
$$
复化梯形公式的代数精度为1阶，截断误差为 $O(h^2)$，即区间等分数翻倍时，误差约缩小为原来的 $\frac{1}{4}$。

##### 实现步骤

- 计算步长 $h$ 与所有节点坐标；
- 计算节点处的函数值，按公式加权求和（首尾节点权重为 $\frac{1}{2}$，中间节点权重为1）；
- 乘以步长 $h$ 得到积分近似值。

#### 复化辛普森公式

##### 核心原理

要求将积分区间 $[a,b]$ 等分为偶数个（$n=2m$）子区间，步长 $h = \frac{b-a}{n}$，将每两个相邻子区间 $[x_{2k}, x_{2k+2}]$ 作为一组，应用辛普森公式（抛物线求积）。整体公式为：
$$
\int_a^b f(x)dx \approx \frac{h}{3}\left[f(x_0) + f(x_n) + 4\sum_{k=0}^{m-1}f(x_{2k+1}) + 2\sum_{k=1}^{m-1}f(x_{2k})\right]
$$
复化辛普森公式的代数精度为3阶，截断误差为 $O(h^4)$，即区间等分数翻倍时，误差约缩小为原来的 $\frac{1}{16}$，收敛速度远快于复化梯形公式。

##### 实现步骤

- 确保区间等分数 $n$ 为偶数（若输入奇数则自动加1）；
- 计算步长 $h$ 与所有节点坐标；
- 按公式加权求和（首尾节点权重为1，奇数位置节点权重为4，偶数位置中间节点权重为2）；
- 乘以 $\frac{h}{3}$ 得到积分近似值。

#### 可视化与收敛性分析

- 绘制被积函数曲线，叠加复化梯形公式的梯形近似区域、复化辛普森公式的抛物线近似区域，直观展示两种方法的几何近似过程；
- 计算不同等分数（$n=4,8,16,32,64$）下的积分值，分析积分结果随 $n$ 的收敛趋势，通过相邻两次计算结果的差值估计误差，对比两种方法的收敛速度。


### 运行结果及可视化表示

#### 数值计算结果

| 区间等分数 $n$ | 复化梯形公式积分值 | 复化辛普森公式积分值 | 两者差值（绝对值） |
| -------------- | ------------------ | -------------------- | ------------------ |
| 4              | 0.9445135217       | 0.9460869340         | 1.5734122864e-03   |
| 8              | 0.9456908636       | 0.9460833109         | 3.9244730577e-04   |
| 16             | 0.9459850299       | 0.9460830854         | 9.8055450562e-05   |
| 32             | 0.9460585610       | 0.9460830713         | 2.4510342794e-05   |
| 64             | 0.9460769431       | 0.9460830704         | 6.1273657651e-06   |

#### 误差收敛特性

- 复化梯形公式：$n$ 从4增加到64（每次翻倍），误差依次为 $1.1773419173e-03$、$2.9416635168e-04$、$7.3531028382e-05$、$1.8382097295e-05$，误差约以 $\frac{1}{4}$ 倍缩小，符合 $O(h^2)$ 的收敛阶；
- 复化辛普森公式：$n$ 从4增加到64，误差依次为 $3.6230633219e-06$、$2.2550352397e-07$、$1.4079385657e-08$、$8.7973406337e-10$，误差约以 $\frac{1}{16}$ 倍缩小，符合 $O(h^4)$ 的收敛阶。

#### 可视化结果

##### 近似过程可视化

展示了 $n=4$ 时，将区间 $[0,1]$ 分为4个子区间，每个子区间用近似函数曲线下的面积，结果如下：

<img src="D:\Codes\Numerical_Analysis\composite_quadrature_approximation.png" alt="composite_quadrature_approximation" style="zoom: 10%;" />

##### 收敛性分析

- 积分值随 $n$ 变化曲线：复化辛普森公式的积分值更快收敛到稳定值，复化梯形公式收敛较慢；
- 误差对数坐标图：复化辛普森公式的误差曲线斜率更大，体现出更高的收敛速度。

<img src="D:\Codes\Numerical_Analysis\composite_quadrature_convergence.png" alt="composite_quadrature_convergence" style="zoom:12%;" />



### 对算法的理解与分析

#### 精度对比

- 复化梯形公式：代数精度为1阶，仅能精确积分一次多项式，对高次函数或非多项式函数（如 $\frac{\sin x}{x}$）需增大 $n$ 才能提升精度。当 $n=64$ 时，积分值为 $0.9460769431$，与高精度值仍有 $6.1273657651e-06$ 的偏差；
- 复化辛普森公式：代数精度为3阶，能精确积分三次多项式，对光滑函数（如 $\frac{\sin x}{x}$）具有极佳的近似效果。当 $n=64$ 时，积分值为 $0.9460830704$，已非常接近该积分的精确值（$\int_0^1 \frac{\sin x}{x}dx \approx 0.9460830703671831$），误差仅为 $8.7973406337e-10$。

#### 收敛特性

- 复化梯形公式的误差与 $h^2$ 成正比，即步长减半（$n$ 翻倍），误差约降低为原来的 $\frac{1}{4}$。实验中 $n$ 从4到8，误差从 $1.1773419173e-03$ 降至 $2.9416635168e-04$，约为 $\frac{1}{4}$；
- 复化辛普森公式的误差与 $h^4$ 成正比，步长减半，误差约降低为原来的 $\frac{1}{16}$。实验中 $n$ 从4到8，误差从 $3.6230633219e-06$ 降至 $2.2550352397e-07$，约为 $\frac{1}{16}$，收敛速度远快于梯形公式。

## 方程求根

### 实验目的

1. 掌握不动点迭代法和牛顿迭代法的基本原理与实现方法，理解方程求根迭代算法的核心思想——通过构造迭代格式逐步逼近方程的精确根。
2. 针对方程 $f(x) = x^3 - 3x - 1 = 0$，实现两种迭代算法，求解 $x_0 = 2$ 附近的根，要求结果准确到四位有效数字。
3. 对比不动点迭代法和牛顿迭代法的收敛速度、计算量，分析两种算法的优缺点及适用场景。
4. 通过可视化手段直观展示迭代过程的几何意义，加深对迭代法收敛特性的理解。

### 算法描述

#### 不动点迭代法

##### 核心原理

将方程 $f(x) = 0$ 等价变形为 $x = g(x)$ 的形式，构造迭代格式 $x_{k+1} = g(x_k)$，从初始值 $x_0$ 开始逐步迭代，若迭代序列 $\{x_k\}$ 收敛，则极限值即为方程的根（不动点）。

对于本实验方程 $x^3 - 3x - 1 = 0$，选择收敛性较好的变形形式：
$$
g(x) = (3x + 1)^{\frac{1}{3}}
$$
需满足不动点迭代收敛条件：在根的邻域内 $|g'(x)| < 1$，保证迭代序列收敛。

##### 实现步骤

1. 定义迭代函数 $g(x) = (3x + 1)^{\frac{1}{3}}$；
2. 从初始值 $x_0 = 2$ 开始，计算 $x_{k+1} = g(x_k)$；
3. 计算相邻迭代值的相对误差，若小于设定容差（四位有效数字对应容差 $0.5 \times 10^{-4}$），则停止迭代；
4. 记录迭代次数、迭代序列及最终近似根。

#### 牛顿迭代法

##### 核心原理

牛顿迭代法（切线法）利用函数在迭代点的切线近似代替函数曲线，通过切线与x轴的交点确定下一个迭代值，迭代格式为：
$$
x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}
$$
其中 $f'(x)$ 为 $f(x)$ 的一阶导数，本实验中 $f'(x) = 3x^2 - 3$。

牛顿迭代法具有二阶收敛性，收敛速度远快于不动点迭代法，但每次迭代需计算函数值和导数值，单次迭代计算量更大。

##### 实现步骤

1. 计算迭代点 $x_k$ 处的函数值 $f(x_k)$ 和导数值 $f'(x_k)$；
2. 代入牛顿迭代公式计算 $x_{k+1}$；
3. 检查收敛条件（相对误差小于容差），满足则停止迭代；
4. 记录迭代次数、迭代序列、函数值序列及最终近似根。

### 实验结果

#### 数值计算结果

| 迭代方法     | 初始值 | 迭代次数 | 近似根       | 绝对误差                | 函数值                  | 满足精度要求 |
| ------------ | ------ | -------- | ------------ | ----------------------- | ----------------------- | ------------ |
| 不动点迭代法 | 2.0    | 7        | 1.8794023973 | $1.7157 \times 10^{-5}$ | $1.3032 \times 10^{-4}$ | 是           |
| 牛顿迭代法   | 2.0    | 3        | 1.8793852448 | $4.8367 \times 10^{-9}$ | $2.4801 \times 10^{-8}$ | 是           |

#### 计算量对比

| 方法         | 迭代次数 | 每次迭代计算量            | 总计算量 |
| ------------ | -------- | ------------------------- | -------- |
| 不动点迭代法 | 7        | 1次函数求值               | 7        |
| 牛顿迭代法   | 3        | 1次函数求值 + 1次导数求值 | 6        |

#### 可视化结果

##### 函数图像与根的位置

<img src="数值分析.assets/root_finding_function.png" alt="root_finding_function" style="zoom:10%;" />

该图展示了函数 $f(x) = x^3 - 3x - 1$ 的曲线形态，清晰标注了精确根的位置（$x^* = 1.87938524$），直观呈现函数在根附近的符号变化，为迭代法初始值选择提供依据。

##### 不动点迭代法几何解释与收敛过程

![root_finding_fixed_point](数值分析.assets/root_finding_fixed_point.png)

- 上图：不动点迭代的几何意义——通过 $y = g(x)$ 与 $y = x$ 的交点确定不动点，迭代过程表现为“阶梯状”逼近交点，清晰展示了从初始值 $x_0=2$ 逐步向根收敛的过程。
- 下图：迭代序列随迭代次数的变化曲线，可见迭代值逐步趋近精确根，收敛过程平稳但速度较慢。

##### 牛顿迭代法几何解释与收敛过程

<img src="数值分析.assets/root_finding_newton.png" alt="root_finding_newton" style="zoom:10%;" />

- 上图：牛顿迭代法的几何意义（切线法）——每次迭代用函数在当前点的切线代替曲线，切线与x轴交点即为下一个迭代值，直观体现“以直代曲”的核心思想。
- 下图：牛顿迭代序列的收敛过程，迭代值快速逼近精确根，体现二阶收敛的优势。

##### 两种方法收敛性对比

<img src="数值分析.assets/root_finding_convergence.png" alt="root_finding_convergence" style="zoom:10%;" />

- 上图：两种迭代法的迭代序列对比，牛顿迭代法仅需3次迭代即达到高精度，不动点迭代法需7次迭代才满足精度要求。
- 下图：误差随迭代次数的对数变化曲线，牛顿迭代法误差下降斜率远大于不动点迭代法，体现其更快的收敛速度。

##### 函数值随迭代次数的变化

<img src="数值分析.assets/root_finding_function_values.png" alt="root_finding_function_values" style="zoom:10%;" />

对数坐标下，牛顿迭代法的 $|f(x)|$ 快速下降至接近机器精度，不动点迭代法下降速度显著较慢，进一步验证了牛顿迭代法的收敛优势。

### 算法理解与分析

#### 收敛速度分析

- **不动点迭代法**：本实验中迭代7次达到四位有效数字精度，绝对误差为 $1.7157 \times 10^{-5}$，属于线性收敛（一阶收敛）。其收敛速度取决于 $|g'(x^*)|$，$|g'(x^*)|$ 越小，收敛越快，本实验中 $g'(x^*) \approx 0.2778$，保证了收敛性但速度较慢。
- **牛顿迭代法**：仅需3次迭代即达到极高精度，绝对误差仅 $4.8367 \times 10^{-9}$，属于二阶收敛（平方收敛）。二阶收敛意味着每次迭代误差约为前一次误差的平方，因此收敛速度呈指数级提升。

#### 计算量分析

从总计算量来看，不动点迭代法总计算量为7次函数求值，牛顿迭代法总计算量为3次函数求值+3次导数求值（总计6次计算操作），牛顿迭代法总计算量更低。即使牛顿迭代法单次迭代计算量更大，但收敛速度的优势使其总体效率更高。

#### 算法优缺点

| 特性       | 不动点迭代法                                 | 牛顿迭代法                            |
| ---------- | -------------------------------------------- | ------------------------------------- |
| 收敛速度   | 慢（线性收敛）                               | 快（二阶收敛）                        |
| 单次计算量 | 小（仅需计算 $g(x)$）                        | 大（需计算 $f(x)$ 和 $f'(x)$）        |
| 收敛条件   | 需满足 $|g'(x^*)| < 1$，对 $g(x)$ 构造要求高 | 要求 $f'(x^*) \neq 0$，初始值需接近根 |
| 实现难度   | 低（迭代格式简单）                           | 中（需推导并实现导数）                |
| 适用场景   | 函数导数难以计算的场景                       | 函数光滑、导数易求的场景              |

#### 实验结论

1. 对于方程 $x^3 - 3x - 1 = 0$，两种迭代法均能在初始值 $x_0=2$ 处收敛到精确根 $x^* = 1.87938524$，且满足四位有效数字的精度要求。
2. 牛顿迭代法收敛速度远快于不动点迭代法，尽管单次迭代计算量更大，但总计算量更低，整体效率更高。
3. 不动点迭代法的收敛性依赖于迭代函数 $g(x)$ 的构造，选择合适的 $g(x)$ 是算法成功的关键；牛顿迭代法对初始值敏感，需保证初始值在根的邻域内且导数非零。
4. 在实际应用中，若函数导数易求，优先选择牛顿迭代法；若导数难以计算或无解析表达式，可选择不动点迭代法（需合理构造 $g(x)$）。

### 总结

本实验通过不动点迭代法和牛顿迭代法求解三次方程的根，验证了两种迭代算法的有效性，并对比了其收敛特性和计算效率。实验结果表明，牛顿迭代法凭借二阶收敛特性，在收敛速度和整体计算效率上显著优于不动点迭代法，但对函数光滑性和初始值选择要求更高。可视化结果直观展现了迭代过程的几何意义，加深了对迭代法收敛本质的理解，为后续数值计算中迭代算法的选择提供了实践依据。





## 源程序

### 牛顿插值

```python
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

```

### 复化求积

```python
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


def print_text_visualization(a, b, n, method='trapezoidal'):
    """文本可视化（当matplotlib不可用时）"""
    if method == 'trapezoidal':
        val, x_nodes, y_nodes = composite_trapezoidal(a, b, n)
        method_name = "复化梯形公式"
    else:
        val, x_nodes, y_nodes = composite_simpson(a, b, n)
        method_name = "复化辛卜生公式"
    
    print(f"\n{method_name} (n={n}) 文本可视化:")
    print("=" * 70)
    print(f"{'节点序号':<8} {'x值':<15} {'f(x)值':<20} {'区间宽度':<15}")
    print("-" * 70)
    
    h = (b - a) / n
    for i in range(len(x_nodes)):
        if i < len(x_nodes) - 1:
            width = f"{h:.6f}"
        else:
            width = "-"
        print(f"{i:<8} {x_nodes[i]:<15.8f} {y_nodes[i]:<20.10f} {width:<15}")
    
    print(f"\n积分近似值: {val:.10f}")
    print("=" * 70)


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
    print("\n所有可视化图像已生成完成！")
    
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

```

### 方程求根的迭代法

```python
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

```

### LU分解

```python

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


```



