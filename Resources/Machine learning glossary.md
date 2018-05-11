# 机器学习术语表

以下列表是重要的机器学习术语汇总。这些术语在构建自定义模型时非常有用。

## 准确度(Accuracy)

真实结果占总案例的比值。从0（最不准确）到1（最准确）。准确度是用来为您的模型性能评分的唯一评估手段，同时您也应该与[精确率(Precision)](#精确率(Precision))和[召回率](#召回率(Recall))一同考虑。

## 曲线下面积(Area under the curve)(AUC)

表示当假正 (flase positives) 描绘在 X 轴上，而真正 (true positives) 被描绘到 y 轴上时候的面积。范围从0.5（最差）到1（最好）。 

## 二分类(Binary classification)

一个[分类](#分类(Classification))情况，[标签](#标签(Label))指向两个类别中的一个。有关更多信息，请参阅Wikipedia上的[二分类](https://en.wikipedia.org/wiki/Binary_classification)文章。

## 分类(Classification)

当数据被用来预测一个类别时，[监督学习](#监督机器学习(Supervised machine learning))也被称为分类。[二分类](#二分类(Binary classification))是指仅预测两个类别（例如，将图像分配为“猫”或“狗”的图片）。[多分类](#多分类(Multiclass classification))是指预测多个类别（例如，将图像分类为特定品种的狗时）。 

## 决定系数(Coefficient of determination)

一个表示数据如何适合模型的数字。值为1表示模型与数据完全匹配。值为0意味着数据是随机的，或者不适合模型。这通常被称为$r^{2}$，$R^{2}$或r-squared。

## 特征(Feature)

被测量现象的可衡量属性，通常为数值（double value）。多个特征称为**特征向量**，通常存储为`double[]`。特征定义了关于正在测量的现象的重要特征。欲了解更多信息，请参阅维基百科上的[专题](https://en.wikipedia.org/wiki/Feature_(machine_learning))文章。 

## 特征工程(Feature engineering)

特征工程是开发将其他数据类型（记录，对象，...）转换为特征向量的软件的过程。由此产生的软件执行特征提取。欲了解更多信息，请参阅维基百科上的[特性工程](https://en.wikipedia.org/wiki/Feature_engineering)文章。 

## F-得分(F-score)

衡量测试精度的衡量标准，可平衡[精确度](#精确率(Precision))和[召回率](#召回率(Recall))。 

## 超参数(Hyperparameter)

机器学习算法的参数。示例包括决策树中要学习的树的数量或梯度下降算法中的步长。这些参数被称为*超参数*，因为学习的过程是识别所述预测函数的正确的参数的过程。例如，线性模型中的系数或树中的比较点。查找这些参数的过程由超参数管理。欲了解更多信息，请参阅在维基百科的[超参数](https://en.wikipedia.org/wiki/Hyperparameter)文章。 

## 标签(Label)

要用机器学习模型预测的元素。例如，狗的品种或未来的股票价格。 

## 对数损失(Log loss)

损失是指训练数据上的模型的准确性的算法和任务特定度量。对数损失是相同损失量的对数。 

## 平均绝对偏差（MAE）

基于平均所有模型偏差的模型评估度量标准，其中偏差是预测值距真实值的距离。 

## 模型(Model)

传统上，这指的是预测函数的参数。例如，线性模型中的权重或树中的分割点。在 ML.NET 中，模型包含预测域对象（例如图像或文本）标签所需的所有信息。这意味着 ML.NET 模型包含必要的特征化步骤以及预测函数的参数。 

## 多分类(Multiclass classification)

一种[分类](#分类(Classification))情况。[标签](#标签(Label))是三个或更多的类中之一。有关更多信息，请参阅Wikipedia上的[多分类](https://en.wikipedia.org/wiki/Multiclass_classification)文章。 

## N元(N-grams)

文本数据的特征提取方案。任何N个字的序列都变成一个[特征](#特征(Feature))。

## 数值特征向量(Numerical feature vectors)

一个仅包含数值的特征向量。这与 double [] 类似。

## 流水线(Pipeline)

将模型拟合到数据集所需的所有操作。流水线由数据导入、转换、特征化和学习步骤组成。一旦流水线被训练，它就变成了一个模型。

## 精确率(Precision)

真实结果与积极结果的比例。

## 召回率(Recall)

所有结果中的所有正确结果的所占分数比。

## 回归(Regression)

一个有监督的机器学习任务，其输出是一个实际值，例如，double。例子包括预言和预测股票价格。

## 相对绝对偏差(Relative absolute error)

一个评估指标，将误差表示为真实值的百分比。

## 相对平方偏差(Relative squared error)

通过除以预测值的总平方误差来对总平方误差进行归一化的评估度量。

## 均方根误差（RMSE）

偏差的平方的平均值，然后取该值的根，作为评估模型的评估度量之一。

## 监督机器学习(Supervised machine learning)

机器学习的一个子类，其中需要模型来预测尚未看到的数据的标签。示例包括分类，回归和结构化预测。欲了解更多信息，请参阅维基百科上的 [监督学习](https://en.wikipedia.org/wiki/Supervised_learning)文章。

## 训练(Training)

确定给定训练数据集的模型的过程。对于线性模型，这意味着找到权重。对于树来说，它涉及识别分割点。

## 转换(Transform)

一个流水线组件，用于转换数据。例如，从文本到数字向量。

## 无监督机器学习(Unsupervised machine learning)

机器学习的一个子类，其中需要一个模型来发现数据中隐藏的（或潜在的）结构。示例包括聚类，主题建模和降维。欲了解更多信息，请参阅维基百科上的[无监督学习](https://en.wikipedia.org/wiki/Unsupervised_learning)文章。

