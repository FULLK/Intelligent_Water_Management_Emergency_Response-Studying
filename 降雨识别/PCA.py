import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def min_max_normalize(matrix):
    """
    使用最小-最大规范化方法将矩阵内所有数据缩放至 [0, 1] 范围内。
    """
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(matrix)
    return normalized_matrix


def calculate_pca_weights(matrices):
    """
    对每个矩阵进行 PCA 分析，计算各主成分对应的方差比例并归一化。
    """
    pca = PCA(n_components=1)
    variances = []

    for matrix in matrices:
        pca.fit(matrix)
        variances.append(pca.explained_variance_ratio_[0])

    total_variance = sum(variances)
    weights = [var / total_variance for var in variances]
    return weights


def calculate_comprehensive_distance(XR, XC, XE, weights):
    """
    根据给定的权重系数计算综合距离。
    """
    omega1, omega2, omega3 = weights
    D_total = omega1 * XR + omega2 * XC + omega3 * XE
    return D_total


# 示例输入距离矩阵
MR = np.random.rand(100, 100)  # 示例 MR 矩阵
MC = np.random.rand(100, 100)  # 示例 MC 矩阵
ME = np.random.rand(100, 100)  # 示例 ME 矩阵

# 步骤 1: 最小-最大规范化
XR = min_max_normalize(MR)
XC = MC  # MC 不需要进行最小-最大规范化
XE = min_max_normalize(ME)

# 步骤 2: PCA 分析并计算权重
matrices = [XR, XC, XE]
weights = calculate_pca_weights(matrices)

# 给定的权重系数
given_weights = [0.14, 0.35, 0.51]

# 步骤 3: 计算综合距离
D_total = calculate_comprehensive_distance(XR, XC, XE, given_weights)

print("综合距离矩阵:\n", D_total)

# 打印 PCA 得到的权重系数
print("PCA 计算的权重系数:", weights)