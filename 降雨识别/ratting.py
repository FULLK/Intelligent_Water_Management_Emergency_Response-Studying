import numpy as np


def calculate_comprehensive_distance_matrix(XR, XC, XE, weights):
    """
    根据给定的权重系数计算综合距离矩阵。
    """
    omega1, omega2, omega3 = weights
    M_total = omega1 * XR + omega2 * XC + omega3 * XE
    return M_total


def convert_to_similarity_matrix(distance_matrix):
    """
    将综合距离矩阵转换为综合相似矩阵。
    """
    similarity_matrix = 1 - distance_matrix
    # 确保对角线元素为 1（即自身与自身的相似度）
    np.fill_diagonal(similarity_matrix, 1)
    return similarity_matrix


def rate_similarity(similarity_matrix, thresholds=[0.78, 0.70, 0.55, 0]):
    """
    根据相似度的分布情况将相似度划分为四个等级，并进行评级。
    """
    ratings = np.zeros_like(similarity_matrix, dtype=object)

    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if similarity_matrix[i, j] >= thresholds[0]:
                ratings[i, j] = 'A'
            elif similarity_matrix[i, j] >= thresholds[1]:
                ratings[i, j] = 'B'
            elif similarity_matrix[i, j] >= thresholds[2]:
                ratings[i, j] = 'C'
            else:
                ratings[i, j] = 'D'

    return ratings


def get_most_similar_rainfall(similarity_matrix, top_n=5):
    """
    对于每一行或每一列，找到最相似的其他场次降雨，并按相似度从高到低排序。
    """
    most_similar = []
    for i in range(similarity_matrix.shape[0]):
        # 获取第 i 行的相似度（除去自身）
        similarities = [(j, similarity_matrix[i, j]) for j in range(similarity_matrix.shape[1]) if i != j]
        # 按相似度从高到低排序
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
        most_similar.append(sorted_similarities)

    return most_similar


# 可视化

def main():
    # 示例输入距离矩阵 (假设已经规范化)
    XR = np.random.rand(225, 225)  # 示例 XR 矩阵
    XC = np.random.rand(225, 225)  # 示例 XC 矩阵
    XE = np.random.rand(225, 225)  # 示例 XE 矩阵

    # 给定的权重系数
    given_weights = [0.14, 0.35, 0.51]

    # 步骤 1: 计算综合距离矩阵
    M_total = calculate_comprehensive_distance_matrix(XR, XC, XE, given_weights)

    # 步骤 2: 转换为综合相似矩阵
    MS = convert_to_similarity_matrix(M_total)

    # 步骤 3: 评级
    thresholds = [0.78, 0.70, 0.55, 0]
    ratings = rate_similarity(MS, thresholds)

    # 步骤 4: 排序并输出最相似的矩阵排行
    top_n = 5  # 输出前 5 名最相似的场次降雨
    most_similar = get_most_similar_rainfall(MS, top_n=top_n)

    print("综合相似矩阵:\n", MS)
    print("评级矩阵:\n", ratings)
    print("最相似的场次降雨排行:")
    for i, similar in enumerate(most_similar):
        print(f"场次 {i + 1} 最相似的场次降雨: {[f'{idx + 1}: {sim:.3f}' for idx, sim in similar]}")


if __name__ == "__main__":
    main()




