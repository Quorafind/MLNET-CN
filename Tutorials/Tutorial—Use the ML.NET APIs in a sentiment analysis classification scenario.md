# 教程：在情感分析分类场景中使用 ML.NET API

本示例教程演示了如何使用 ML.NET API 在 Visual Studio 2017 中使用 C# 通过 .NET Core 控制台应用程序创建情感分类器。

在教程中，您可以学习如何：

- [x] 了解问题
- [x] 创建学习流水线 (pipeline) 
- [x] 加载分类器
- [x] 训练模型
- [x] 预测模型
- [x] 用不同的数据集评估模型



## 情感分析样本概述

本示例是一个控制台应用程序，它使用 ML.NET API 来训练将情感分类并预测情感为正面或负面的模型。它还将使用第二个数据集来评估模型并进行质量分析。情感数据集来自加利福尼亚大学欧文分校（UCI），其将自动下载并请您解压缩其到数据目录中。

预测和评估结果会相应显示，以便进行分析和采取行动。

情绪分析既可以是正面的，也可以是负面的。因此，您可以使用分类器来训练模型，进行预测和评估。



## 机器学习流程

本教程遵循机器学习流程，使您能够有序地进行流程。 

流程阶段如下： 

1. **了解问题**
2. **提取数据**
3. **数据预处理和特征工程**
4. **训练和预测模型**
5. **评估模型**
6. **运行模型**



### 了解问题

首先，您需要了解问题，以便将其分解为可以支持构建和训练模型的部分。将问题细分，预测并评估结果。

本教程的问题是要了解传入的网站评论的情感以采取适当的行动。

您可以将问题分解为要训练模型的数据的情感文本和情感值，以及可以评估并随后使用的预测情感值。

然后您需要**确定**（Determine）情感，这有助于您选择机器学习模型。

有了这个问题，可以知道以下问题：

训练数据集：可以是正面或负面（**情感**）的网站评论。预测新网站评论的**情感**，无论是正面还是负面。

## 预设条件

运行环境：安装了 “ .NET Core 跨平台开发” 工作负载的 [Visual Studio 2017 15.6 或更高版本](https://www.visualstudio.com/downloads/?utm_medium=microsoft&utm_source=docs.microsoft.com&utm_campaign=button+cta&utm_content=download+vs2017)。

数据集：[UCI Sentiment Labeled Sentences 数据集 zip 文件](https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip)

## 创建控制台应用程序

1. 打开 Visual Studio 2017. 从菜单栏中选择**文件 (File) ** > **新建 (New) ** > **项目 (Project) **。在 *New Project* * 对话框中，选择 **Visual C#** 节点，然后选择 **.NET Core** 节点。然后选择 **Console App (.NET Core)**  项目模板。在**名称 (Name) **文本框中，输入 “**SentimentAnalysis**” ，然后选择**确定 (OK)** 按钮。

2. 在项目的 *bin* 目录中创建一个名为 Data 的目录：

   在解决方案资源管理器 (Solution Explorer) 中，单击**解决方案和文件夹 (Solutions and Folders) **图标。右键单击 *bin* 文件夹，选择**添加 (Add)** > **新文件夹 (New Folder) **。输入“**数据 (Data) **”，然后按Enter键。再次单击**解决方案和文件夹**图标以返回到解决方案视图。

3. 安装 **Microsoft ML.NET NuGet Package **：

   在解决方案资源管理器中，右键单击您的项目并选择**管理 (Manage) NuGet包**。选择 “nuget.org” 作为软件包源，选择浏览 (Browse) 选项卡，搜索 **Microsoft ML.NET **，在列表中选择该软件包，然后选择**安装 (Install) **按钮。如果提示选择包管理格式，请**在项目文件中选择 PackageReference**。

4. 下载 [UCI Sentiment Labeled Sentences数据集zip文件（请参阅下面的注释中的引文）](https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip)，然后解压缩到您创建的 *数据* 目录中。

> 注意：
>
> 本教程使用的数据集来自 “From Group to Individual Labels using Deep Features ” ，由Kotzias等人制作。出自 KDD 2015，托管于UCI Machine Learning Repository - Dua, D. and Karra Taniskidou, E. (2017) 。UCI机器学习库[ <http://archive.ics.uci.edu/ml> ]。加州大学欧文分校：加州大学信息与计算机科学学院。 

## 内务处理

将下方的 `using` 语句添加到 *Program.cs* 文件头中

```C#
using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
```

您需要创建两个全局变量来指示刚刚下载好的文件的路径：

- `_datapath` 指示用来训练模型的数据集的路径。
- `_testdatapath` 指示用来评估模型的数据集的路径。

将以下代码添加到 `Main` 方法的上方的行中：

```C#
const string _dataPath = @"..\..\data\sentiment labelled sentences\imdb_labelled.txt";
const string _testDataPath = @"..\..\data\sentiment labelled sentences\yelp_labelled.txt";
```

您需要为您的输入数据和预测结果创建一些类。给您的项目添加一个新的类吧：

1. 在**解决方案资源管理器中**，选择 SentimentAnalysis 项目，然后在**项目**菜单上选择**添加类 (Add Class)**。

2. 在**添加新文件 (Add New Item) **对话框中，将**名称**字段更改为 “SentimentData.cs” ，然后选择**添加**按钮。

   在代码编辑器中打开该 *SentimentData.cs* 文件。将以下 `using` 语句添加到 *SentimentData.cs* 的顶部：

   ```C#
   using Microsoft.ML.Runtime.Api;
   ```

添加以下包含了两个类（分别是 `SentimentData` 和  `SentimentPrediction` ）的代码到 *SentimentData.cs* 文件中：

```C#
public class SentimentData
{
    [Column(ordinal: "0")]
    public string SentimentText;
    [Column(ordinal: "1", name: "Label")]
    public float Sentiment;
}

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Sentiment;
}
```

`SentimentData` 是输入数据集类，带有 comment (`SentimentText`) 字符串和内含正面或负面的情感值的 `float`(`Sentiment`) 值。这两个字段都附有属性 `Column` 。该属性描述数据文件中每个字段的顺序，以及哪个是 `Label` 字段。`SentimentPrediction` 是模型训练后用于预测的类。它有 boolean（`Sentiment`）值和一个`PredictedLabel` `ColumnName`属性。`Label`会被用于创建和训练模型，它也有第二个数据集用于评估模型。在预测和评估过程中会使用`PredictedLabel`。为了评估，使用带有训练数据，预测值和模型的输入。 

在 *Program.cs* 文件中，将以下代码行替换掉 `Main` 方法中的 `Console.WriteLine("Hello World!")` ：=

```c#
var model = TrainAndPredict();
```

`TrainAndPredict` 方法执行以下任务： 

- 加载或摄取数据。
- 预处理和特征化数据。
- 训练模型。
- 根据测试数据预测情感。

在 `Main` 方法后，用以下代码，创建 `TrainAndPredict`  方法：

```C#
public static PredictionModel<SentimentData, SentimentPrediction> TrainAndPredict()
```

## 加载数据

初始化一个新的 [LearningPipeline ](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.learningpipeline) 实例，该实例将包含数据加载，数据处理/特征化和模型。添加下面的代码作为 `TrainAndPredict` 方法的第一行：

```C#
var pipeline = new LearningPipeline();
```

所述 [TextLoader<TInput> ](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.textloader-1)对象是流水线的第一部分，并加载训练文件数据。 

```
pipeline.Add(new TextLoader<SentimentData>(_dataPath, useHeader: false, separator: "tab"));
```

## 数据预处理和特征工程

预处理和数据清洗是在把数据集用于机器学习前的重要任务。原始数据通常带有噪音且不可靠，并且可能缺少某些值。在没有执行这些建模的步骤的情况下使用数据会使得模型产生误导性的结果。ML.NET 的转换流水线允许您在训练或测试模型之前编写一组自定义的转换，这些转换适用于您的数据。而转换的主要目的是为了特征化数据。转换流水线的优点是在您定义了转换流水线以后，您可以保存流水线，从而可以将其应用于测试数据集。

应用 [TextFeaturizer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.textfeaturizer)将 `SentimentText` 列转换为机器学习算法使用的数字向量 `Features` 。这是预处理/特征化步骤。使用 ML.NET 中的其他组件可以使您的模型获得更好的结果。将 `TextFeaturizer` 作为下一行代码添加到流水线中：

```
pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
```

### 关于分类模型

分类是一种机器学习方法，它使用数据来**确定**文段或数据行的类别，类型或类（Class）。例如，您可以使用分类来：

- 将情感识别为正面或负面。
- 将电子邮件过滤器分类为垃圾邮件，无用邮件或优秀邮件。
- 确定患者的实验室样本是否癌变。
- 根据客户对销售活动的反应倾向对客户进行分类。

分类任务通常是以下类型之一：

- 二分类：A或B.
- 多类：可以使用单一模型预测的多个类别。

该 [FastTreeBinaryClassifier](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.fasttreebinaryclassifier) 对象是您会在这次的流水线中使用的决策树学习器 (Learner) 。与特征化步骤类似，尝试 ML.NET 中可用的不同学习器并更改它们的参数，看看会导致哪些不同的效果。对于调参，您可以设置超参数，如 [NumTrees](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.fasttreebinaryclassifier.numtrees#Microsoft_ML_Trainers_FastTreeBinaryClassifier_NumTrees) ，[NumLeaves](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.fasttreebinaryclassifier.numleaves#Microsoft_ML_Trainers_FastTreeBinaryClassifier_NumLeaves) 和 [MinDocumentsInLeafs](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.fasttreebinaryclassifier.mindocumentsinleafs#Microsoft_ML_Trainers_FastTreeBinaryClassifier_MinDocumentsInLeafs) 。这些超参数应当在模型没被训练前设置，此外这些都是模型特定的。它们面向于调整决策树的性能，因此较大的值会对性能产生负面影响。

将下面的代码添加到 `TrainAndPredict` 方法中：

```c#
pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
```

## 训练模型

基于已加载和转换的数据集训练模型 [PredictionModel ](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.predictionmodel-2) 。`pipeline.Train<SentimentData, SentimentPrediction>()` 训练流水线（加载数据，训练 featurizer 和学习器 ）。在这开始之前，不会开始训练。

将下面的代码添加到 `TrainAndPredict` 方法中：

```C#
PredictionModel<SentimentData, SentimentPrediction> model = 
    pipeline.Train<SentimentData, SentimentPrediction>();
```

## 预测模型

添加一些评论以测试这个训练模型在`TrainAndPredict` 方法中的预测：

```C#
IEnumerable<SentimentData> sentiments = new[]
 {
    new SentimentData
    {
        SentimentText = "Contoso's 11 is a wonderful experience",
        Sentiment = 0
    },
    new SentimentData
    {
        SentimentText = "The acting in this movie is very bad",
        Sentiment = 0
    },
    new SentimentData
    {
        SentimentText = "Joe versus the Volcano Coffee Company is a great film.",
        Sentiment = 0
    }
};
```

现在您已经有了一个模型，您可以使用 [PredictionModel.Predict](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.predictionmodel.predict) 方法来 [预测](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.predictionmodel.predict) 评论数据的情感是正面或负面。要获得预测，请在新数据上使用`Predict`。请注意，输入数据是一个字符串，模型包含了特征化。在训练和预测期间，您的流水线是同步的。您不必编写专门用于预测的预处理/特征化代码，此外，同一个 API 可以同时处理批量预测和一次性预测。 

```
IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);
```

### 模型操作化 (operationalization) ：预测

显示`SentimentText`和相应的情感预测结果，以分享结果并据此采取下一步行动。这就是所谓的操作化 ，使用返回的数据作为操作策略的一部分。使用以下 [Console.WriteLine()](https://docs.microsoft.com/en-us/dotnet/api/system.console.writeline#System_Console_WriteLine) 代码为结果创建标题：

```
Console.WriteLine();
Console.WriteLine("Sentiment Predictions");
Console.WriteLine("---------------------");
```

在显示预测结果之前，将情感和预测结合在一起，以便根据预测的情绪查看原始评论。下面的代码使用 [Zip](https://docs.microsoft.com/en-us/dotnet/api/system.linq.enumerable.zip) 方法来实现这一点，所以接下来添加下面的代码： 

```
var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
```

现在您已将 `SentimentText` 类和`Sentiment`类组合在一起，您可以使用 [Console.WriteLine()](https://docs.microsoft.com/en-us/dotnet/api/system.console.writeline#System_Console_WriteLine) 方法显示结果： 

```
foreach (var item in sentimentsAndPredictions)
{
    Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
}
Console.WriteLine();
```

#### 将经过训练的模型返回用于评估

在`TrainAndPredict` 方法结尾处返回模型。此时，您可以将其保存到一个 zip 文件或继续使用它。但是在本教程中，您仍将继续使用它，因此请将以下代码添加到`TrainAndPredict`的下一行中：

```
return model;
```

## 评估模型

现在您已经创建和训练了模型，您需要使用不同的数据集对其进行评估以确保质量。在 `Evaluate` 方法中，在`TrainAndPredict` 创建的模型会被传入函数以进行评估。像下面的代码那样在`TrainAndPredict`后创建`Evaluate`方法：

```
public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
```

使用以下代码，在`TrainAndPredict`方法下边添加对从`Main`而来的新方法的调用： 

```
Evaluate(model);
```

所述 [TextLoader  ](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.textloader-1)类加载具有相同的架构的新的测试数据集。您可以使用此数据集作为质量检查来评估模型。将其用以下代码的方式添加到`Evaluate`的调用上去：

```
 var testData = new TextLoader<SentimentData>(_testDataPath, useHeader: false, separator: "tab");
```

所述 [BinaryClassificationEvaluator](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.models.binaryclassificationevaluator) 对象通过使用指定的数据集为 `PredictionModel` 计算质量指标。要查看这些指标，请使用以下代码将评估程序添加`Evaluate`方法的下一行：

```
 var evaluator = new BinaryClassificationEvaluator();
```

该 [BinaryClassificationMetrics](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.models.binaryclassificationmetrics) 包含二分类评估计算的整体指标。要显示这些来确定模型的指标，我们需要首先获取指标。添加下面的代码： 

```
BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);
```

### 显示模型验证的指标

使用以下代码显示评价标准，共享结果并对其执行相应的操作：

```
Console.WriteLine();
Console.WriteLine("PredictionModel quality metrics evaluation");
Console.WriteLine("------------------------------------------");
Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"Auc: {metrics.Auc:P2}");
Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
```

## 结果

您的结果应该类似于以下内容。随着流水线处理，它显示消息。您可能会看到警告或处理消息。为清晰起见，这些已从以下结果中删除。

```
Sentiment Predictions
---------------------
Sentiment: Contoso's 11 is a wonderful experience | Prediction: Positive
Sentiment:The acting in this movie is really bad | Prediction: Negative
Sentiment: Joe versus the Volcano Coffee Company is a great film. | Prediction: Positive


PredictionModel quality metrics evaluation
------------------------------------------
Accuracy: 67.30%
Auc: 73.78%
F1Score: 65.25%
Press any key to continue . . .
```

## 下一步

在这次的教程中，您已经学习了该如何：

- [x] 了解问题
- [x] 创建学习流水线 (pipeline) 
- [x] 加载分类器
- [x] 训练模型
- [x] 预测模型
- [x] 用不同的数据集评估模型

### 请阅览进一步的教程来获取更多相关知识

[出租车费用预测](https://github.com/Quorafind/MLNET-CN/blob/master/Tutorials/Tutorial%E2%80%94Use%20ML.NET%20to%20Predict%20New%20York%20Taxi%20Fares%20(Regression).md)

