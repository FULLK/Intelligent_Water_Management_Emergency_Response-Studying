
# 参考
李虎, 王丽晶, 杨思敏, 等. 基于多距离融合的场次降雨时空相似性比较[J]. 水利水电技术(中英文), 2024, 55(6): 106- 119. LI Hu, WANG Lijing, YANG Simin, et al. Spatio-temporal similarity comparison of rainfall events based on multi-distance fusion  [J]. Water Resources and Hydropower Engineering, 2024, 55(6): 106- 119.
# 业务
业务：根据实时数据识别出大雨暴雨的准确性提高
#   方法
使用相似度比较法识别
- 相似度要素所属宏观对象（不同区域）
- 相似度比对选取要素
- 相似度如何比对选取要素
- 相似度比对选取数据处理


> 基于多距离融合的场次降雨时空相似性比较：多距离融合
> 【方法】 首先通过设置雨量占比阈值, 进行降雨场次划分, 共划分位于汛期(6—9 月)降雨 225 场。 随后提取降雨量、 降雨历时、 降雨强度、降雨中心范围等特征要素, 并计算 3 个方面的降雨特征距离: 降雨要素距离、 降雨中心距离、 降雨总量距离。 同时为了更好综合特征距离, 采用主成分分析方法进行距离融合计算权重, 得出最终的距离, 并根据计算结果进行相似度排序。

存在问题
- 数据问题强雨暴雨少+强降雨无上限，各个差异大
- 每次计算相似度都与历史数据每个都比较，花销大
-模型 各个距离的参数得到方法是否妥当，主成分分析
- 各个距离的计算方法是否妥当
- 选取的特征要素要科学，尽量往可能提高暴雨识别率的方向
- 不同场景还要再添加对应的特征距离
-  一场雨的评定


多距离+单核心+深度学习调整

# 复现
## 特征提取及距离计算
## 降雨特征来源
- 地域分像元，每个像元一个降雨特征
- 不可能所有像元都有实际捕获的降雨特征，所以需要反距离权重法补全数据

一个矩阵的每个元素表示一个像元

## 距离
### 降雨总量距离

两场降雨总量距离=两场的每个元素的降雨量的差的平方的和
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df00df84d52f4af98057ede9ecab4af3.png)

###  降雨中心距离
该场降雨靠前的前10%元素为降雨中心

两场雨的降雨中心相似性=两场雨的降雨中心重叠个数/总元素个数
两场雨的降雨中心距离=1-两场雨的降雨中心相似性
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e731426103694fc3a6b1de875e693341.png)

### 降雨要素距离
(1)降雨历时 T, 单位为 h, 表示降雨过程的时间要素, 以整点计算。
 (2)强降雨历时 Tt, 单位为 h, 表示降雨中心范围内雨强大于 7 mm / h 的小时
数。 
(3)降雨量平均值 Ra , 单位为 mm, 表示场次降雨总量中所有格点降雨信息的平均值。 ( 4) 降雨量最大值 Rm 。 单位为 mm, 表示场次降雨总量中前10%格点降雨信息的平均值。 
( 5) 1 h 最大雨强 I,单位为 mm / h, 表示先求出逐小时雨量中所有格点降雨信息的平均值, 再选出其中的最大值。 
( 6) 降雨中心最大雨强 Ic。 单位为 mm / h, 表示先求出逐小时雨量中降雨中心范围内降雨信息的平均值, 再选出其中的最大值。

以场来算，可能涉及到各个元素降雨特征如何得到该场雨降雨特征

该场雨的降雨要素先标准化操作

两场雨的降雨要素距离=两场雨的分别6个降雨要素的差的平方的和的开根
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b37d3a5ce3540e6ae6af934a714ceb4.png)
## 主成分分析融合权重

[主成分分析PCA的基本原理与Python实现-基于sklearn](https://www.bilibili.com/video/BV1Fe41157Tw/?spm_id_from=333.337.search-card.all.click&vd_source=a1be939c65919194c77b8a6a36c14a6e)

[用最直观的方式告诉你：什么是主成分分析PCA](https://www.bilibili.com/video/BV1E5411E71z/?spm_id_from=333.337.search-card.all.click&vd_source=a1be939c65919194c77b8a6a36c14a6e)


得到三个距离矩阵后，需要统一单位
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b4a6c128ab54c90b004bba6a6129301.png)
然后主成分分析得到模型参数
## 相似度量及评级
计算各个距离矩阵*参数=综合距离矩阵后
便知道与每场雨的距离了，那1-每个元素的矩阵=相似矩阵

然后对相似度分级，最高的就是最相似的
随着类别的增加, 样本相似度呈现整体减小的趋势, 雨量较小的类别相似度集中分布在较高的范围, 而雨量较大的类别相似度分布范围较广, 平均值较低, 相似度评估效果不理想。
## K-means 聚类分析
[十分钟掌握k-means++聚类算法原理及python实现（详细讲解代码，包教包会！新手超级友好）](https://www.bilibili.com/video/BV12NsuedEiL/?spm_id_from=333.337.search-card.all.click&vd_source=a1be939c65919194c77b8a6a36c14a6e)
聚类变量：6个降雨要素
聚类数目： 6 类，手肘法

根据同一类的计算之间相似度来评价


