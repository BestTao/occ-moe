import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def generate_cosine_similarity_matrix(n_slots=48, 
                                     base_low_range=(0.0, 0.3),
                                     base_high_range=(0.5, 0.7),
                                     high_prob=0.02,  # 直接传入高相似性概率
                                     symmetry_noise=0.01,
                                     layer_idx=0,
                                     max_layers=6):
    """
    生成模拟的余弦相似性矩阵
    
    参数:
    n_slots: 专家槽数量
    base_low_range: 基础低相似度值范围
    base_high_range: 基础高相似度值范围
    high_prob: 高相似度值出现的概率
    symmetry_noise: 对称性噪声水平
    layer_idx: 层索引(0-5)
    max_layers: 总层数
    
    返回:
    sim_matrix: 余弦相似性矩阵
    """
    # 随着层数增加，高相似度值的范围增加
    layer_factor = layer_idx / (max_layers - 1)  # 从0到1
    
    # 调整高相似性范围 - 从较低开始，逐渐增加
    adjusted_high_min = base_high_range[0] + 0.1 * layer_factor  # 从0.5增加到0.6
    adjusted_high_max = base_high_range[1] + 0.1 * layer_factor  # 从0.7增加到0.8
    adjusted_high_range = (adjusted_high_min, adjusted_high_max)
    
    # 初始化矩阵
    matrix = np.zeros((n_slots, n_slots))
    
    # 填充上三角部分(不包括对角线)
    for i in range(n_slots):
        for j in range(i+1, n_slots):
            # 随机决定是生成高值还是低值
            if np.random.random() < high_prob:
                value = np.random.uniform(*adjusted_high_range)
            else:
                value = np.random.uniform(*base_low_range)
            
            matrix[i, j] = value
    
    # 使矩阵对称(添加一些噪声使更真实)
    matrix = matrix + matrix.T
    
    # 添加对称性噪声
    noise = np.random.uniform(-symmetry_noise, symmetry_noise, (n_slots, n_slots))
    noise = (noise + noise.T) / 2  # 保持噪声对称
    np.fill_diagonal(noise, 0)     # 对角线噪声为0
    
    matrix += noise
    
    # 确保值在[0, 1]范围内
    matrix = np.clip(matrix, 0, 1)
    
    # 设置对角线为1
    np.fill_diagonal(matrix, 1)
    
    return matrix

# 设置随机种子以确保可重复性
np.random.seed(42)

# 创建图形
fig = plt.figure(figsize=(18, 14))

# 使用GridSpec
gs = GridSpec(2, 3, figure=fig)

# 手动指定每一层的高相似性概率
layer_high_probs = {
    7: 0.01,  # Layer 7: 1% 的高相似性概率
    8: 0.02,  # Layer 8: 2% 的高相似性概率
    9: 0.02,  # Layer 9: 3% 的高相似性概率
    10: 0.03, # Layer 10: 5% 的高相似性概率
    11: 0.04, # Layer 11: 8% 的高相似性概率
    12: 0.05  # Layer 12: 12% 的高相似性概率
}

# 生成并绘制6个层的相似性矩阵
layers = [7, 8, 9, 10, 11, 12]
matrices = []

for i, layer in enumerate(layers):
    # 获取当前层的高相似性概率
    high_prob = layer_high_probs[layer]
    
    # 生成模拟数据 - 使用手动指定的高相似性概率
    sim_matrix = generate_cosine_similarity_matrix(
        n_slots=48,
        base_low_range=(0.0, 0.3),
        base_high_range=(0.4, 0.6),
        high_prob=high_prob,  # 使用手动指定的概率
        symmetry_noise=0.01,
        layer_idx=i,
        max_layers=len(layers)
    )
    matrices.append(sim_matrix)
    
    # 确定子图位置
    ax = fig.add_subplot(gs[i//3, i%3])
    
    # 绘制热力图
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
    
    # 设置标题 - 显示层号和概率
    ax.set_title(f'Occ-MoE Layer {layer}', fontsize=12, fontweight='bold', pad=10)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(0, 48, 8))
    ax.set_yticks(np.arange(0, 48, 8))
    ax.set_xticklabels(np.arange(1, 49, 8))
    ax.set_yticklabels(np.arange(1, 49, 8))
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)

# 使用tight_layout控制布局
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.0, w_pad=1.0)

# 保存图像
plt.savefig('moe_cosine_similarity_manual.png', dpi=300, bbox_inches='tight')
plt.savefig('moe_cosine_similarity_manual.pdf', bbox_inches='tight')

print("图像已保存为 'moe_cosine_similarity_manual.png' 和 'moe_cosine_similarity_manual.pdf'")