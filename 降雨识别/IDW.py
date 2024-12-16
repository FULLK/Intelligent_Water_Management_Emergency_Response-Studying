import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# 1. 准备数据
# 假设我们有1000个已知数据点，需要估算剩下的3506个点
n_total = 4506
n_known = 1000

# 生成示例数据（在实际应用中，你需要使用你的真实数据）
np.random.seed(0)
known_points = np.random.rand(n_known, 2)  # 已知点的坐标
known_values = np.random.rand(n_known)  # 已知点的值
unknown_points = np.random.rand(n_total - n_known, 2)  # 未知点的坐标


# 2. 定义IDW函数
def idw_interpolation(known_points, known_values, unknown_points, p=2):
    """
    使用反距离权重法进行插值

    :param known_points: 已知点的坐标
    :param known_values: 已知点的值
    :param unknown_points: 未知点的坐标
    :param p: 距离的幂次，默认为2
    :return: 未知点的估算值
    """
    distances = cdist(unknown_points, known_points)
    weights = 1.0 / (distances ** p)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.dot(weights, known_values)


# 3. 应用IDW函数
estimated_values = idw_interpolation(known_points, known_values, unknown_points)

# 4. 合并已知点和估算点
all_points = np.vstack((known_points, unknown_points))
all_values = np.concatenate((known_values, estimated_values))

# 5. 创建包含所有数据的DataFrame
result_df = pd.DataFrame({
    'x': all_points[:, 0],
    'y': all_points[:, 1],
    'value': all_values
})

# 打印结果摘要
print(result_df.describe())
print(result_df)
# 可以将结果保存到CSV文件
# result_df.to_csv('interpolated_data.csv', index=False)