# 教程：使用ML.NET预测纽约出租车票价（回归）

本教程演示了您该如何使用ML.NET构建预测纽约市出租车票价的[回归模型](https://github.com/Quorafind/MLNET-CN/blob/master/Resources/Machine%20learning%20glossary.md#%E5%9B%9E%E5%BD%92)。 

在本教程中，您将学习如何：

- [x] 了解问题
- [x] 选择适当的机器学习任务
- [x] 准备并理解你的数据
- [x] 创建学习流水线
- [x] 加载和转换您的数据
- [x] 选择一种学习算法
- [x] 训练模型
- [x] 评估模型
- [x] 使用模型进行预测

## 预设条件

- 运行环境：安装了 “ .NET Core 跨平台开发” 工作负载的 [Visual Studio 2017 15.6 或更高版本](https://www.visualstudio.com/downloads/?utm_medium=microsoft&utm_source=docs.microsoft.com&utm_campaign=button+cta&utm_content=download+vs2017)。
- 数据集：[NYC TLC Taxi Trip 数据集](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml)可以用来训练机器学习模型，并且可用于评估模型的准确性。 

## 了解问题

这个问题主要集中在**预测纽约市出租车行程的相应票价上**。乍看之下，它可能似乎仅取决于旅行的距离。然而，纽约的出租车供应商会为其他因素，例如额外的乘客或使用信用卡而不是现金支付而收取额外的费用。 

## 选择适当的机器学习任务

为了预测出租车费用，您首先选择适当的机器学习任务。您正在根据数据集中的其他因素预测真实价值（双倍于价格）。所以，您决定选择一个[**回归**](https://github.com/Quorafind/MLNET-CN/blob/master/Resources/Machine%20learning%20glossary.md#%E5%9B%9E%E5%BD%92)任务。

训练模型的过程确定数据集中的哪些因素对预测最终票价影响最大。

## 创建控制台应用程序

1. 启动 Visual Studio 2017 。新建一个名为 “TaxiFarePrediction” 的 C＃**控制台应用程序( Console App )(.NET Core)**项目。

2. 在项目的 *bin* 目录中创建一个名为 Data 的目录。

3. 安装 ML.NET NuGet 包

   点击 **Tools** 菜单，然后选择 **NuGet Package Manager** ，然后选择 **Package Manager Console** 。在**提示符窗口**下键入 “Install-Package Microsoft.ML” 。

   ```
   PM > Install-Package Microsoft.ML
   ```

### 准备和理解您的数据

下载[ taxitrip-train.csv 和 taxitrip-test.csv 数据集](https://github.com/dotnet/machinelearning/tree/master/test/data)并将它们保存到先前创建的 Data 文件夹中。 

在代码编辑器中打开 **taxitrip-train.csv** 数据集并查看第一行的列标题。看看每一列。理解数据并确定哪些列是**特征(features)**，哪些是**标签(label)**。 

这**标签**是你正在试图预测的列的标识符。识别的**特征**用于预测标签。 

- **vendor_id：**出租车供应商的ID是一项特征。
- **rate_code：**出租车行程的费率类型是一项特征。
- **passenger_count：**旅行中的乘客人数是一项特征。
- **trip_time_in_secs：**旅程花费的时间。你不知道旅程需要多长时间才能完成。您应该从模型中排除此列。
- **trip_distance：**行程的距离是一项特征。
- **payment_type：**付款方式（现金或信用卡）是一项特征。
- **fare_amount：**所支付的出租车总价是标签。

### 创建类并定义路径

 将以下`using`语句添加到Program.cs的顶部： 

```
using System;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
```

您可以定义变量来保存您的 datapath（训练模型的数据集），testdatapath（评估模型的数据集）和 modelpath（存储训练模型的位置）。将以下代码添加到 `Main` 方法上面的行中以指定最近下载的文件： 

```
const string DataPath = @"..\..\..\Data\taxi-fare-train.csv";
const string TestDataPath = @"..\..\..\Data\taxi-fare-test.csv";
const string ModelPath = @"..\..\..\Models\Model.zip";
```

接下来，为输入数据和预测创建类：

1. 在**解决方案资源管理器( Solution Explorer )中**，选择 TaxiFarePredicion 项目，然后在 **项目** 菜单上，选择 **添加类(Add Class)** 。
2. 在**添加新项目(Add New Item)**对话框中，将**名称**更改为`TaxiTrip.cs`，然后单击**添加**。
3. 添加以下`using`语句：

```
using Microsoft.ML.Runtime.Api;
```

将两个类添加到此文件中`TaxiTrip` 。输入数据集类具有上面发现的每个列的定义和您预测的fare_amount列的属性相对应的`Label`。将以下代码添加到文件中： 

```
public class TaxiTrip
{
    [Column(ordinal: "0")]
    public string vendor_id;
    [Column(ordinal: "1")]
    public string rate_code;
    [Column(ordinal: "2")]
    public float passenger_count;
    [Column(ordinal: "3")]
    public float trip_time_in_secs;
    [Column(ordinal: "4")]
    public float trip_distance;
    [Column(ordinal: "5")]
    public string payment_type;
    [Column(ordinal: "6")]
    public float fare_amount;
}
```

在模型被训练后，`TaxiTripFarePrediction` 类将被用于预测。它有一个 float (fare_amount) 和一个`Score` `ColumnName`属性。将以下代码添加到文件中的`TaxiTrip`类下面：

```
public class TaxiTripFarePrediction
{
    [ColumnName("Score")]
    public float fare_amount;
}
```

现在回到**Program.cs**文件。在`Main`方法中，用以下代码替换掉`Console.WriteLine("Hello World!")`： 

```
PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = Train();
```

该`Train()`函数会训练您的模型。使用以下代码在`Main`下面创建该函数： 

```
public static PredictionModel<TaxiTrip, TaxiTripFarePrediction> Train()
{

}
```

## 创建学习流水线

现在创建学习流水线。学习流水线加载训练模型所需的所有数据和算法。然后，请您将以下代码复制到该`Train()`方法中： 

```
var pipeline = new LearningPipeline();
```

## 加载和转换您的数据

接下来，将数据加载到流水线中。指向最初创建的数据路径并指定.csv文件的分隔符(,)。将以下代码复制到在上一步以下的`Train()`方法中： 

```
var pipeline = new LearningPipeline();
```

使用 `ColumnCopier()`  函数将“fare_amount ”列复制到名为“标签”的新列中。此列是**标签**。 

```
pipeline.Add(new ColumnCopier(("fare_amount", "Label")));
```

进行一些**特征工程**来转换数据，以便它可以有效地用于机器学习。该训练模型需要算法的**数字**特征，您转换分类数据（`vendor_id`，`rate_code`，和`payment_type`）为数字。`CategoricalOneHotVectorizer()`函数为每个列中的值分配一个数字键。通过添加以下代码来转换您的数据： 

```
pipeline.Add(new CategoricalOneHotVectorizer("vendor_id",
                                             "rate_code",
                                             "payment_type"));
```

数据准备的最后一步是使用`ColumnConcatenator()`函数将所有**特征组合**到一个向量中。这一必要步骤有助于算法轻松处理您的特征。添加以下代码到您编写的内容下边：

```
pipeline.Add(new ColumnConcatenator("Features",
                                    "vendor_id",
                                    "rate_code",
                                    "passenger_count",
                                    "trip_distance",
                                    "payment_type"));
```

请注意，“trip_time_in_secs”列不包括在内。你已经确定它不是一个有用的预测特征。 

> 注意：
>
> 这些步骤必须按照上面指定的顺序添加到流水线中才能成功执行。 

## 选择一种学习算法

在将数据添加到流水线并将其转换为正确的输入格式之后，您可以选择一种学习算法（**学习器(Learner)**）。学习算法训练模型。您应该为这个问题选择一个**回归任务**，所以您增加了一个`FastTreeRegressor()`学习器到使用**梯度提升( gradient boosting )**的流水线中。

梯度提升是回归问题的机器学习技巧。它以逐步(step-wise)的方式构建每个回归树。它使用预定义的损失函数来测量每个步骤中的错误，并在下一步中对其进行修正。结果是一个训练好的预测模型实际上是较弱预测模型的集合。您可以了解更多关于Azure机器学习[梯度提升](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/boosted-decision-tree-regression)信息。

将以下代码添加到上一步中添加的数据处理代码之后的`Train()`方法中：

```
pipeline.Add(new FastTreeRegressor());
```

您将所有前面的步骤作为单独的语句添加到流水线中，但 C# 具有方便的集合初始化语法，可以使创建和初始化流水线更简单：

```
 var pipeline = new LearningPipeline
{
    new TextLoader<TaxiTrip>(DataPath, useHeader: true, separator: ","),
    new ColumnCopier(("fare_amount", "Label")),
    new CategoricalOneHotVectorizer("vendor_id",
                                 "rate_code",
                                 "payment_type"),
    new ColumnConcatenator("Features",
                                    "vendor_id",
                                    "rate_code",
                                    "passenger_count",
                                    "trip_distance",
                                    "payment_type"),
    new FastTreeRegressor()
};
```

## 训练模型

最后一步是训练模型。在此之前，流水线中没有任何东西被执行。`pipeline.Train<T_Input, T_Output>()`函数接受预定义的 `TaxiTrip` 类类型并输出一个 `TaxiTripFarePrediction` 类型。将这代码添加到`Train()`函数最后一段中：

```
PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
```

就是这样！你已经成功地训练了一个可以预测纽约市出租车票价的机器学习模型。现在看一看，了解您的模型有多准确，并学习该如何使用它。 

## 保存模型

在进入下一步之前，通过在`Train()`函数结尾处添加以下代码将模型保存到 .zip 文件中：

```
await model.WriteAsync(ModelPath);
return model;
```

将该`await`语句添加到`model.WriteAsync()`调用中意味着该`Train()`方法必须更改为返回 `Task` 的异步方法。用以下的代码修改`Train`的签名：

```
public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
```

改变`Train`方法的返回类型意味着您必须增加 `await`到在`Method`中调用 `Train` 的代码当中，代码如下所示： 

```
PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
```

在你的`Main`方法中添加一个`await`意味着该`Main`方法必须具有`async`修饰符并返回 `Task`：

```
static async Task Main(string[] args)
```

您还需要在文件顶部添加以下using语句： 

```
using System.Threading.Tasks;
```

## 评估模型

评估是检查模型运行情况的步骤。重要的是，您的模型能够很好地基于它在训练时不使用的数据预测。一种方法是将数据分解为训练数据集和测试数据集，就像您在本教程中所做的一样。现在您已经在训练数据上训练了模型，您将看到它在测试数据上的表现如何。

现在回到你的`Main`函数并在调用`Train()`方法的下面添加下面的代码：

```
Evaluate(model);
```

该`Evaluate()`函数评估你的模型。在`Train()`下边创建了函数。添加下面的代码：

```
private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
```

使用该`TextLoader()`函数加载测试数据。将下面的代码添加到`Evaluate()`方法中： 

```
var testData = new TextLoader<TaxiTrip>(TestDataPath, useHeader: true, separator: ",");
```

添加以下代码以评估模型并为其生成度量标准： 

```
var evaluator = new RegressionEvaluator();
RegressionMetrics metrics = evaluator.Evaluate(model, testData);
```

RMS 是评估回归问题的一个指标。它越低，你的模型就越好。将以下代码添加到该`Evaluate()`函数中以打印模型的RMS。 

```
// Rms 应该在 2.795276 附近
Console.WriteLine("Rms=" + metrics.Rms);
```

RSquared是评估回归问题的另一个指标。RSquared将是介于0和1之间的值。越接近1，模型越好。将下面的代码添加到该`Evaluate()`函数中以打印模型的RSquared值。 

```
Console.WriteLine("RSquared = " + metrics.RSquared);
```

## 使用模型进行预测

在该`Evaluate()`函数之后，创建一个类来容纳可以用来确保模型正常工作的测试场景。使用以下代码定义一个类来容纳测试数据：

```
static class TestTrips
{

}
```

本教程在该课程中使用一次 test trip 。稍后，您可以添加其他场景来尝试此示例。将以下代码添加到`TestTrips`类中： 

```
internal static readonly TaxiTrip Trip1 = new TaxiTrip
{
    vendor_id = "VTS",
    rate_code = "1",
    passenger_count = 1,
    trip_distance = 10.33f,
    payment_type = "CSH",
    fare_amount = 0 // 预测这个，实际是 actual = 29.5
};
```

此行的实际票价为29.5，但会使用0作为占位符。机器学习算法将预测票价。

在你的`Main`函数中添加下面的代码。它使用`TestTrip`数据测试你的模型：

```
var prediction = model.Predict(TestTrips.Trip1);
Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.fare_amount);
```

运行该程序查看您的测试用例的预计出租车票价。

恭喜！您现在已经成功建立了一个机器学习模型，用于预测出租车票价，评估其准确性并对其进行测试。

## 下一步

在本教程中，您学习了如何：

- [x] 了解问题
- [x] 选择适当的机器学习任务
- [x] 准备并理解你的数据
- [x] 创建学习流水线
- [x] 加载和转换您的数据
- [x] 选择一种学习算法
- [x] 训练模型
- [x] 评估模型
- [x] 使用模型进行预测

查看我们的[GitHub存储库](https://github.com/dotnet/machinelearning/)以继续学习并找到更多示例。