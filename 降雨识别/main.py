import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 特征提取及距离计算
def calculate_euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def calculate_total_rainfall_distance(events):
    n = len(events)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = calculate_euclidean_distance(events[i], events[j])
    return distance_matrix

def calculate_center_similarity(center_a, center_b, total_points=450):
    top_k_indices_a = np.argsort(center_a)[-total_points:]  #最大的十个元素索引数组
    top_k_indices_b = np.argsort(center_b)[-total_points:]
    # 计算交集中元素的数量
    overlap_count = len(np.intersect1d(top_k_indices_a, top_k_indices_b))  #交集数组
    return overlap_count / total_points

def calculate_center_distance(events_centers):
    n = len(events_centers)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity = calculate_center_similarity(events_centers[i], events_centers[j])
            distance_matrix[i, j] = 1 - similarity
    return distance_matrix

def calculate_feature_distances(features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features) # 对每列进行标准化处理
    n = normalized_features.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = calculate_euclidean_distance(normalized_features[i], normalized_features[j])
    return distance_matrix

# 2. 聚类分析
def kmeans_clustering(features, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return labels, cluster_centers

def plot_cluster_boxplot(features, labels, n_clusters=4):
    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 20))
    for i in range(n_clusters):
        cluster_data = features[labels == i]
        axes[i].boxplot(cluster_data)
        axes[i].set_title(f'Cluster {i+1}')
    plt.show()

# 3. 主成分分析融合权重
def min_max_normalize(matrix):
    """
    对矩阵进行最小-最大规范化
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

def principal_component_analysis(X):
    """
    进行主成分分析
    返回方差和权重系数
    """
    pca = PCA()
    pca.fit(X)
    variances = pca.explained_variance_ratio_
    weights = np.abs(pca.components_[0])  # 使用第一主成分作为权重
    return variances, weights

def process_distance_matrices(MR, MC, ME):
    """
    处理距离矩阵
    """
    XR = min_max_normalize(MR)
    XE = min_max_normalize(ME)
    XC = MC  # MC保持不变

    # 对XR, XC, XE进行主成分分析
    pca_R = PCA().fit(XR)
    pca_C = PCA().fit(XC)
    pca_E = PCA().fit(XE)

    # 获取方差比例
    variances_R = pca_R.explained_variance_ratio_
    variances_C = pca_C.explained_variance_ratio_
    variances_E = pca_E.explained_variance_ratio_

    # 使用第一主成分的方差比例作为权重
    weight_R = variances_R[0]
    weight_C = variances_C[0]
    weight_E = variances_E[0]

    # 归一化权重系数
    total_weight = weight_R + weight_C + weight_E
    weight_R_normalized = weight_R / total_weight
    weight_C_normalized = weight_C / total_weight
    weight_E_normalized = weight_E / total_weight

    return XR, XC, XE, weight_R_normalized, weight_C_normalized, weight_E_normalized


def calculate_total_distance(XR, XC, XE, w1=0.14, w2=0.35, w3=0.51):
    """
    计算综合距离
    """
    D_total = w1 * XR + w2 * XC + w3 * XE
    return D_total



# 4. 相似性度量及评级
def calculate_similarity_matrix(total_distance_matrix):
    similarity_matrix = 1 - total_distance_matrix
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
# 5. 找到与各个降雨最相似的降雨
def get_most_similar_rainfall(similarity_matrix, top_n=5):
    """
    对于每一行或每一列，找到最相似的其他场次降雨，并按相似度从高到低排序。
    """
    most_similar = []
    for i in range(similarity_matrix.shape[0]):
        # 获取第 i 行的相似度（除去自身）  根据元组的第二个元素（相似度值）来排序
        similarities = [(j, similarity_matrix[i, j]) for j in range(similarity_matrix.shape[1]) if i != j]
        # 按相似度从高到低排序
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
        most_similar.append(sorted_similarities)

    return most_similar

# 可视化
def visualize_similar_events(event_index, similarity_matrix, most_similar, top_n=5):
    """
    根据输入的场次编号，可视化最相似的场次降雨。

    参数:
    event_index (int): 用户输入的场次索引（从0开始）
    similarity_matrix (numpy.ndarray): 综合相似度矩阵
    most_similar (list of list of tuples): 每个场次最相似的场次列表，每个元组包含索引和相似度
    top_n (int): 显示最相似的前n个场次
    """
    # 确保event_index在合法范围内
    if event_index < 0 or event_index >= len(most_similar):
        print("无效的场次编号")
        return

    # 获取最相似的场次及其相似度
    similarities = most_similar[event_index]

    # 可视化条形图
    indices, values = zip(*similarities)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values, tick_label=[f"Event {idx + 1}" for idx in indices])
    plt.title(f"Top {top_n} Most Similar Rainfall Events to Event {event_index + 1}")
    plt.xlabel('Rainfall Event')
    plt.ylabel('Similarity')
    plt.show()

    # 可视化热图
    plt.figure(figsize=(12, 8))
    heatmap_data = similarity_matrix[[event_index] + list(indices), :][:, [event_index] + list(indices)]

    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='coolwarm', cbar=False,
                xticklabels=['Event ' + str(event_index + 1)] + [f'Event {idx + 1}' for idx in indices],
                yticklabels=['Event ' + str(event_index + 1)] + [f'Event {idx + 1}' for idx in indices])
    plt.title(f'Similarity Heatmap for Event {event_index + 1} with Top {top_n} Similar Events')
    plt.show()


def plot_kmeans_clusters(features, labels, cluster_centers):
    """
    使用PCA降维并绘制K-means聚类结果。

    参数:
    features (numpy.ndarray): 特征矩阵
    labels (numpy.ndarray): 聚类标签
    cluster_centers (numpy.ndarray): 聚类中心
    """
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    centers_pca = pca.transform(cluster_centers)

    # 创建散点图以显示聚类结果
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.75, marker='X')  # 显示聚类中心
    plt.title('K-means Clustering Results with PCA')
    plt.colorbar(scatter)
    plt.show()

# 5. 主函数
def main():
    # 假设 events, events_centers, features 是已经加载好的数据
    events = np.random.rand(225, 4506)  # 示例数据 225行 4506列  225场雨，每场雨有4506个像素元
    events_centers = np.random.rand(225, 4506)  # 示例数据
    features = np.random.rand(225, 6)  #每行是每场雨，6个元素代表计算出来的本场雨的6个特征

    mr = calculate_total_rainfall_distance(events)
    mc = calculate_center_distance(events_centers)
    me = calculate_feature_distances(features)

    XR, XC, XE, weights_R, weights_C, weights_E = process_distance_matrices(mr, mc, me)

    # 计算综合距离矩阵
    total_distance_matrix = calculate_total_distance(XR,XC,XE,weights_R, weights_C, weights_E)

    # 计算综合相似度矩阵
    similarity_matrix=calculate_similarity_matrix(total_distance_matrix)

    # 相似性评级
    ratings = rate_similarity(similarity_matrix)

    # 找到与各个降雨最相似的降雨
    top_n = 5  # 输出前 5 名最相似的场次降雨
    most_similar = get_most_similar_rainfall(similarity_matrix, top_n=top_n)

    # 输出结果
    print("综合相似度矩阵：\n", similarity_matrix)
    print("评级结果：\n", ratings)
    print("最相似的场次降雨排行:")
    for i, similar in enumerate(most_similar):
        print(f"场次 {i + 1} 最相似的场次降雨: {[f'{idx + 1}: {sim:.3f}' for idx, sim in similar]}")

    event_input = input("请输入您想查看的场次编号（1-225）: ")
    try:
        event_index = int(event_input) - 1  # 转换为索引（从0开始）
        visualize_similar_events(event_index, similarity_matrix, most_similar, top_n=5)
    except ValueError:
        print("请输入有效的整数编号。")

    # 聚类分析
    labels, cluster_centers = kmeans_clustering(features)
    print("聚类标签：\n", labels)
    print("聚类中心：\n",cluster_centers)
    plot_kmeans_clusters(features, labels, cluster_centers)



if __name__ == '__main__':
    main()