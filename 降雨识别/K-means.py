import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 假设我们有一个包含 6 个降雨要素的数据集
# 数据集应为 Pandas DataFrame 格式，每行代表一个降雨事件，每列代表一个要素
# 示例数据：
# data = pd.read_csv('rainfall_data.csv')  # 如果数据来自 CSV 文件
data = pd.DataFrame(np.random.rand(225, 6), columns=['要素1', '要素2', '要素3', '要素4', '要素5', '要素6'])

# 步骤 3: 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 步骤 4: 使用手肘法确定最佳聚类数目
wcss = []  # Within Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# 绘制手肘图
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 根据手肘图选择最佳聚类数（假设为 6）
optimal_clusters = 6
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(scaled_data)

# 获取每个样本的聚类标签
labels = kmeans.labels_

# 计算轮廓系数以评估聚类效果
silhouette_avg = silhouette_score(scaled_data, labels)
print(f"Silhouette Score: {silhouette_avg}")

# 分析每一类的具体数目及聚类中心
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_info = pd.DataFrame(cluster_centers, columns=data.columns)
cluster_sizes = pd.Series(labels).value_counts().sort_index()

print("每一类的具体数目:")
print(cluster_sizes)
print("\n聚类中心:")
print(cluster_info)

# 特别处理极端事件类别（类别 5 和 6）
extreme_event_indices = [201, 202]  # 假设这是 "7·21" 和 "7·20" 的索引
extreme_event_labels = labels[extreme_event_indices]
if set(extreme_event_labels) == {4, 5}:  # 检查是否为极端事件类别
    print("检测到极端事件类别，单独保留其统计特征")
    final_labels = np.where(np.isin(labels, extreme_event_labels), labels + 2, labels)
    final_optimal_clusters = optimal_clusters - 2
else:
    final_labels = labels
    final_optimal_clusters = optimal_clusters

# 绘制所有类别样本在聚类变量上的箱型图
plt.figure(figsize=(12, 8))
for i in range(final_optimal_clusters):
    subset = scaled_data[final_labels == i]
    plt.boxplot(subset, positions=[i]*subset.shape[1], widths=0.6)
plt.title('Boxplot of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Scaled Feature Values')
plt.show()

# 输出最终聚类结果
print("最终聚类结果:")
print(pd.Series(final_labels).value_counts().sort_index())