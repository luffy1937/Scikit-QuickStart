
# 模型选择与评价 之 交叉验证 

既将数据用来训练模型又用来验证模型是一种方法错误：这样会造成模型在这些重复数据上表现很出色，但是对于新的数据却无能为力（过拟合造成泛化能力弱）。

为了避免这种情况，实践中，有监督机器学习的实验过程中通常会保留一部分数据作为测试集。当然，即使是在生产的机器学习实践中，这种实验也是必不可少的。

在scikit-learn中 函数train_test_split可以随机的将数据分成测试集和训练集。


```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris.data.shape, iris.target.shape
```




    ((150, 4), (150,))



我们将样本通过 train_test_split函数随机划分出40%作为测试集 用来 评估 我们的分类器：


```python
X_train, X_test, y_train, y_test = train_test_split(
    #特征集
    iris.data,
    #标签集
    iris.target,
    #缺省是0.25；如果是浮点数既‘0.’开头则代表测试集的比例；如果是int类型，则代表测试集的样本数量
    test_size=0.4,
    # train_size: test_size 与 train_size只需要指定其一
    
    #当然，这是一个伪随机，这个是seed
    random_state = 1,
    # 缺省是None;总体样本中各类样本有一定比例，当值为“None”时划分出来的训练集和测试集不能保证这个比例，当值为样本labels时，可以维持这个比例
   stratify=iris.target 
)
```


stratify参数赋值为总样本标签， 训练集和测试集中各类的比例与总体样本一致：


```python
from collections import Counter
Counter(iris.target), Counter(y_train), Counter(y_test)
```




    (Counter({0: 50, 1: 50, 2: 50}),
     Counter({0: 30, 1: 30, 2: 30}),
     Counter({0: 20, 1: 20, 2: 20}))



训练集训练模型，测试集做评估


```python
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
```




    1.0



在评估估计器的不同超参数时（比如SVM的惩罚因子“C”就需要调试），仍然会有过拟合的风险，因为这个参数会根据测试集被调整到最优值。

这其实也将测试集的一些信息在模型训练阶段给“泄露”了，从而使得在测试集上的一些度量指标不能反映模型真正的泛化能力。

为了解决这个问题，我们会再分出验证集（validation set）：用训练集（training set）做模型训练，用验证集（validation set）做模型评价，迭代这个过程来调整模型到最优，用测试集做最终的评估。

然而，将样本分成三个集合，会大大减少用来训练模型的样本数量。

通过交叉验证（cross-validation）的方式，就不需要单独划分出一个验证集了。

在基本的方式中（也就是K折交叉验证）：训练集会被分成K个同样大小集合，k-1个用来作为训练集，剩下的一个用来做验证集（比如可以用它来计算下准确率）。

这个过程可以循环K次，每个集合都会做一次验证集。模型的评价值就是这K次循环的平均值。这种方式的计算量较大，但是不会浪费样本。这在样本数量较少的情况下是一个巨大的优势。

>## 1. 计算交叉验证指标

使用函数cross_val_score 是使用交叉验证最简单的方式。

下面的例子示范了 线性核函数支持向量机 在iris数据集上用五折交叉验证得到准确率：


```python
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
scores
```




    array([ 0.96666667,  1.        ,  0.96666667,  0.96666667,  1.        ])



求这个估计器的准确率均值和95%置信区间：


```python
print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    accuracy: 0.98 (+/- 0.03)
    

每次交叉验证的迭代过程都会计算一个“score”,score的计算方式是可以指定的，如下，指定F1作为“score”：


```python
from sklearn import metrics
scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
scores
```




    array([ 0.96658312,  1.        ,  0.96658312,  0.96658312,  1.        ])



当“cv”参数的赋值是一个整形数的时候，cross_val_score会使用 K折(KFold) 或者 分层K折(StratifiedKFold) 策略。

当然也可以通过传递一个交叉验证迭代器来使用其它的交叉验证策略,如下：


```python
from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cross_val_score(clf, iris.data, iris.target, cv=cv)
```




    array([ 0.97777778,  0.97777778,  1.        ])



>### 数据预处理和特征工程阶段也不能掺入测试集

如下在预处理阶段对数据做标准化，标准化器只能由训练集来拟合：


```python
from sklearn import preprocessing
#划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
#拟合标准化器
scaler = preprocessing.StandardScaler().fit(X_train)
#训练数据标准化
X_train_standard = scaler.transform(X_train)
#训练分类模型
clf = svm.SVC(C=1).fit(X_train_standard, y_train)
#测试数据标准化
X_test_standard = scaler.transform(X_test)
#模型评估
clf.score(X_test_standard, y_test)
```




    0.93333333333333335



以上过程有些繁琐，标准化和模型训练（或者预测）是固定的步骤；如果将数据称为流，那么这个过程可以称作管道即`Pipeline`。

`Pipeline` 可以将这个过程结合起来，并将这个过程融入到交叉验证:


```python
from sklearn.pipeline import make_pipeline
#标准化和模型训练（或预测）过程构成一个管道
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
#交叉验证
cross_val_score(clf, iris.data, iris.target, cv=cv)
```




    array([ 0.97777778,  0.93333333,  0.95555556])



>###  1.1 cross_validate函数及多种度量指标

`cross_validate` 函数 和 `cross_val_score` 函数不同，体现在以下两点:
+ 允许指定多个指标来做模型评估
+ 返回的数据类型是字典（dict）,除了测试的得分（score）还有训练得分、训练时间、打分时间


当`score`参数是一个字符串、函数或者None,就是单指标评价，那么所返回的字典的keys为

`['test_score', 'fit_time', 'score_time']`

对于多指标评价，返回的字典的keys为

`['test<scorer1_name>', 'test_<scorer2_name>', 'test_<score...>', 'fit_time', 'score_time']`

`return_train_score`属性缺省为`True`，这会为所有的打分器（scorer）增加`train_<scorer_name>`这个key。如果不需要，可以将这个值设置为`False`

多个指标可以通过包含多个打分器名称的列表（List）、元组(tuple)或者集合(Set）来指定：


```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
#选择准确率和召回率两个打分器
scoring = ['precision_macro', 'recall_macro']
#线性核函数支持向量机惩罚因子为1
clf = svm.SVC(kernel='linear', C=1, random_state=666)
#交叉验证并打分
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring, cv=5, return_train_score=True)
for key, value in scores.items():
    print(key, value )
```

    fit_time [ 0.  0.  0.  0.  0.]
    score_time [ 0.  0.  0.  0.  0.]
    test_precision_macro [ 0.96969697  1.          0.96969697  0.96969697  1.        ]
    train_precision_macro [ 0.97674419  0.97674419  0.99186992  0.98412698  0.98333333]
    test_recall_macro [ 0.96666667  1.          0.96666667  0.96666667  1.        ]
    train_recall_macro [ 0.975       0.975       0.99166667  0.98333333  0.98333333]
    

当然，也可以定制自己的打分器:


```python
from sklearn.metrics.scorer import make_scorer
scoring = {"prec_macro": "precision_macro",
          "rec_micro": make_scorer(recall_score, average='macro')}
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring, cv=5, return_train_score=True)
for key, value in scores.items():
    print(key, value)
```

    fit_time [ 0.  0.  0.  0.  0.]
    score_time [ 0.  0.  0.  0.  0.]
    test_prec_macro [ 0.96969697  1.          0.96969697  0.96969697  1.        ]
    train_prec_macro [ 0.97674419  0.97674419  0.99186992  0.98412698  0.98333333]
    test_rec_micro [ 0.96666667  1.          0.96666667  0.96666667  1.        ]
    train_rec_micro [ 0.975       0.975       0.99166667  0.98333333  0.98333333]
    

这是一个单指标交叉验证的例子:


```python
scores = cross_validate(clf, iris.data, iris.target, scoring='precision_macro')
for key, value in scores.items():
    print(key, value)
```

    fit_time [ 0.  0.  0.]
    score_time [ 0.          0.01562691  0.        ]
    test_score [ 1.          0.96491228  0.98039216]
    train_score [ 0.98095238  1.          0.99047619]
    

>### 1.2 通过交叉验证得到预测值

函数`cross_val_predict`与`cross_val_score`有相似的接口，但是它会返回输入元素作为测试集时所得到的预测值。

当然，只有当交叉验证策略使得每个元素都有且只有一次作为验证集的机会时才可以使用这个函数：


```python
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
print(predicted)
metrics.accuracy_score(iris.target, predicted)
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 1
     1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2]
    




    0.97333333333333338



当然这个结果可能与通过cross_val_score函数所得到的有细微的偏差，这是因为分组时会有不同

>## 2 交叉验证迭代器

这一章节会罗列一些生成索引的程序，根据不同的交叉验证策略,这些索引可以来划分数据集。

>### 2.1 面向独立同分布（i.i.d.）数据的交叉验证迭代器

假设一些数据是独立同分布的，也就是说所有的样本都来自于同一个生成过程，并且这个生成过程对过去产生的样本没有记忆（典型的模型就是抛硬币实验）。

下面的交叉迭代器就是用在这种场景中。

注意：
虽然在机器学习理论中，大多都假设数据是独立同分布的，但是在实践中却是少见的。如果我们知道样本来自于依赖时间的生成过程，那最好选择可以感知时间序列的交叉验证方案；同样的，如果我们知道样本来自于具有分组结构的生成过程（样本从不同的主题、实验、设备上收集而来），那么最好选择 分组交叉验证。

>#### 2.1.1 K折交叉验证（K-fold）

KFold 将所有样本分为K个相同规模（如果可能）的组（称为折）；如果k=n(总样本数量)，那么就是留一交叉验证（Leave One Out）；估计器以（K-1）折作为训练集，剩下的1折作为验证集。

四个样本使用2折交叉验证：


```python
import numpy as np
from sklearn.model_selection import KFold

X = ['a', 'b', 'c', 'd']
kf = KFold(n_splits=2)
for train_index, test_index in kf.split(X):
    print(train, test)
```

    [0 1] [2 3]
    [0 1] [2 3]
    

如上，2折交叉验证会有两组划分：每组有两个array,第一个是训练集的索引，第二个是验证集的索引。

可以通过这些索引来划分训练集与测试集，如下：


```python
#特征集
X= np.array([[0., 0.], [1., 1.], [-1., -1], [2., 2.]])
#对应的标签
y= np.array([0, 1, 0, 1])
#根据索引进行划分
print(train, test)
X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
print(X_train, '\n', X_test, '\n', y_train, '\n', y_test)
```

    [0 1] [2 3]
    [[ 0.  0.]
     [ 1.  1.]] 
     [[-1. -1.]
     [ 2.  2.]] 
     [0 1] 
     [0 1]
    

>#### 2.1.2 重复K折（Repeated K-Fold）

RepeatedKFold 会重复n次KFold的过程。当需要多次KFold，并且每次划分不同时，可以考虑用这个函数。

重复两次2折交叉验证：


```python
import numpy as np
from sklearn.model_selection import RepeatedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 888
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(X):
    print(train, test)
```

    [2 3] [0 1]
    [0 1] [2 3]
    [2 3] [0 1]
    [0 1] [2 3]
    

>#### 2.1.3 留一交叉验证（Leave One Out） LOO

每次留一个作为验证集，其余用来做训练集。所以，如果有n个样本，那就会有n种训练集和测试集的划分方式。这种方式不会浪费数据。


```python
from sklearn.model_selection import LeaveOneOut

X = [1, 2, 3, 4]
loo = LeaveOneOut()
for train, test in loo.split(X):
    print(train, test)
```

    [1 2 3] [0]
    [0 2 3] [1]
    [0 1 3] [2]
    [0 1 2] [3]
    

用户想用这种交叉验证策略需要权衡一些风险：
+ 会得到与样本规模一致的模型
+ 每个模型都由n-1个样本训练得到
当n很大时，这种方式相对于普通的K折，需要很大的计算量。

在准确率上，留一交叉验证的方差较大，这是评估模型的一项重要指标。

直观上，使用n-1个样本来作为训练样本训练模型，这些模型几乎与使用所有样本所训练出的模型相同。

然而，如果学习曲线对于训练规模来说是很陡的，那么，五折或者十折交叉验证会高估泛化误差。

一般的，大多数作者和实验数据，都支持五折或者十折交叉验证比留一交叉验证要好。

>#### 2.1.4 留P交叉验证 Leave P Out (LPO)

LeavePOut 与 LeaveOneOut很相似，它留P个样本作为验证集，其余都用来做训练集。对于n个样本的集合来说，会有$ \begin{pmatrix} p \\ n \\ \end{pmatrix} $ 个train-test对。与LeaveOneOut和KFold不同，当 p > 1 时， 样本会不止一次作为验证集（即验证集会有重叠）。

四个样本数据集，留二交叉验证：


```python
from sklearn.model_selection import LeavePOut

X = np.ones(4)
lpo = LeavePOut(p=2)
for train, test in lpo.split(X):
    print(train, test)
```

    [2 3] [0 1]
    [1 3] [0 2]
    [1 2] [0 3]
    [0 3] [1 2]
    [0 2] [1 3]
    [0 1] [2 3]
    

>#### 2.1.5  随机排列(Random permutations)交叉验证  也称  洗牌再划分(Shuffle & Split)

ShuffleSplit 迭代器可以根据用户指定的数量生成 train-test集合。样本首先会被洗牌然后被划分成训练集和验证集。

可以通过控制 伪随机生成器的“种子”（seed）参数，也就是random_state，来控制随机结果的再现：


```python
from sklearn.model_selection import ShuffleSplit
X = np.arange(5)
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=777)
for train_index, test_index in ss.split(X):
    print(train_index, test_index)
```

    [1 2 3] [0 4]
    [1 3 4] [2 0]
    [4 0 2] [3 1]
    

ShuffleSplit 是 K折交叉验证的一个备选，它同样可以控制迭代的次数、训练集或者测试集的划分比例

>### 2.2 基于类标签的分层交叉验证迭代器（Cross-validation iterators with stratification based on class labels）

一些分裂问题在目标类的分布比例上很不平衡：比如，负样本的数量是正样本的好多倍。 在这种情况下，推荐使用分层抽样，比如StratifiedKFold 和 StratifiedShuffleSplit,以保持所划分的训练集/测试集中类标签的相对比例。

>#### 2.2.1 分层的K折交叉验证 Stratified KFold

StratifiedKFlod 是 KFold 的变种，它划分的每一折都是分层的（保持各类标签的相对比例与总体样本一致）

在有十个样本的数据集上使用分层的三折交叉验证：


```python
from sklearn.model_selection import StratifiedKFold

X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train_index, test_index in skf.split(X,y):
    print(train_index, test_index)
```

    [2 3 6 7 8 9] [0 1 4 5]
    [0 1 3 4 5 8 9] [2 6 7]
    [0 1 2 4 5 6 7] [3 8 9]
    

重复分层K折交叉验证 RepeatedStratifiedKFold ,可以重复 StratifiedKFold 的步骤，并且保证每次的随意选择不一样。

>#### 2.2.2 分层的 洗牌再划分 交叉验证  Stratified Shuffle Split

StratifiedShuffleSplit  是 ShuffleSplit的变种，可以保证每次划分的训练集和验证集中的类标签比例与总体样本基本一致。

>### 2.3 不同组别数据的交叉验证迭代器

如果样本间有依赖关系那么独立同分布的假设就不成立。

比如医疗数据是从不同的病人身上收集来的，这种数据就很可能依赖于不同的组。在这个例子中， 病人的id 就是每条数据的组标识。

这种情况下，我们想知道，模型训练时知晓这些组的存在比不知道要好吗？为了验证这个猜想，我们需要确保所有在验证集里的样本，其所在的组不会出现在对应的训练集中。

以下几个交叉验证划分器可以达到这种效果。组标识通过groups 参数来指定。

>#### 2.3.1 分组K折 Group K-fold

GroupKFold 是 KFold 的变种，它可以确保相同组的数据不会同时出现在训练集和验证集。

比如，数据是从不同主题收集来的，并且每个主题下都有几个，如果模型足够敏感，能学习到这些主题的高级特征，那么这个模型在新的主题下就会失去泛化能力。

GroupKFold帮我们避免这种过拟合的情况

假设你有三个主题， 分别用1，2, 3标识：


```python
from sklearn.model_selection import GroupKFold
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ['a', 'b', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd']
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
gkf = GroupKFold(n_splits=3)
for train_index, test_index in gkf.split(X, y, groups=groups):
    print(train_index, test_index)
```

    [0 1 2 3 4 5] [6 7 8 9]
    [0 1 2 6 7 8 9] [3 4 5]
    [3 4 5 6 7 8 9] [0 1 2]
    

每个主题会在不同的的折里，相同的主题不会同时出现在验证集和训练集中。当然，这也造成每个折不能保证具有完全相同的尺寸（因为不能保证每个组下的样本数量一致）。

>#### 2.3.2 留一组交叉验证 Leave One Group Out

LeaveOneGroupOut ，其大体过程是，首先按照组（比如同一个病人的数据分为一折）分出折，每一折分别作为验证集，其余折作为训练集。

这样，做验证的折是在训练集中没有的，如果在验证集里模型表现好 也就说明这个模型 `对没有出现过的组具有好的泛化能力!`




```python
from sklearn.model_selection import LeaveOneGroupOut
X = [1, 5, 10, 50, 60, 70, 80]
y = [0, 1, 1, 2, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3, 3]
logo = LeaveOneGroupOut()
for train_index, test_index in logo.split(X, y, groups=groups):
    print(train_index, test_index)
```

    [2 3 4 5 6] [0 1]
    [0 1 4 5 6] [2 3]
    [0 1 2 3] [4 5 6]
    

一个经常用的的场景是 时间信息：比如如果数据生成过程跟年份有关系，那么年份就是所谓的“组”，这时就可以将相同年份的数据作为一“折”，来做交叉验证，从而使模型考虑到在不同年份中的泛化能力。

>#### 2.3.3 留P组交叉验证 Leave P Group Out

LeavePGroupsOut 与 LeaveOneGroupOut 很相似，同样要保证测试集中的组不能出现在训练集中，但是会将P个不同组的数据作为验证集。




```python
from sklearn.model_selection import  LeavePGroupsOut

X = np.arange(6)
y = [1, 1, 1, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3]
lpgo = LeavePGroupsOut(n_groups=2)
for train_index, test_index in lpgo.split(X, y, groups=groups):
    print(train_index, test_index)
```

    [4 5] [0 1 2 3]
    [2 3] [0 1 4 5]
    [0 1] [2 3 4 5]
    

>#### 2.3.4 组洗牌再划分 Group Shuffle Split

GroupShuffleSplit 迭代器组合了 ShuffleSplit 和 LeavePGroupsOut,数据会按组为基本单位，再随机划分成训练集和验证集。




```python
from sklearn.model_selection import GroupShuffleSplit
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ['a', 'b', 'b', 'b', 'c', 'c', 'c', 'a']
groups = [1, 1, 2, 2, 3, 3, 4, 4]
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=999)
for train_index, test_index in gss.split(X, y, groups=groups):
    print(train_index, test_index)
```

    [0 1 6 7] [2 3 4 5]
    [0 1 2 3] [4 5 6 7]
    [2 3 6 7] [0 1 4 5]
    [0 1 2 3] [4 5 6 7]
    

当我们想使用LeavePGroupsOut,但是组的数目特别多，所有可能划分方式将是一个很大的数目，这将会带来庞大的计算量。在这种情况下，GroupShuffleSplit 的优势就体现出来，它可以指定划分的次数。

>### 2.4 预定义划分（或验证集） Predefined Fold-Splits/Validation-Sets

在一些数据集中，我们想预定义数据划分的方案。可以通过 PredefinedSplit函数的test_fold参数来设置。

test_fold参数是一个有着样本规模大小的数组，test_fold[i]的值是样本i所在的测试集的索引，当这个值赋值为-1时，这个样本不能用在测试集中。


```python
from sklearn.model_selection import PredefinedSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
test_fold = [0, 1, -1, 1]
ps = PredefinedSplit(test_fold)
for train_index, test_index in ps.split(X, y):
    print(train_index, test_index)

```

    [1 2 3] [0]
    [0 2] [1 3]
    

>### 2.5 时间序列数据的交叉验证 Cross validation of time series data

时间序列数据的特点体现在时间接近的数据所具有的关系上。然而，经典的交叉验证技术比如KFold 和 ShuffleSplit 假设数据是独立同分布的，如果用在时间序列数据上，将会导致训练集与验证集之间的不合理关联。

TimeSeriesSplit可以解决这个问题。

>#### 2.5.1 时间序列划分  Time Series Split

在每次划分中，验证集的索引一定要比训练集的索引要高（即时间序列上靠后），因此“洗牌”对于这种时序交叉验证是不适应的。

其划分的基本原理是：在第K次划分时，会将前K折作为训练集，第（k+1）折作为验证集；因此，其后续的训练集是之前训练集的超集。



```python
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(X):
    print(train_index, test_index)
```

    [0] [1]
    [0 1] [2]
    [0 1 2] [3]
    

>### 3. 关于“洗牌” A note on shuffling

如果数据的顺序不是任意的（比如相同类别的样本是连续出现的），那么只有经过洗牌，才能得到一个有意义的交叉验证结果。然而，如果数据不是独立同分布的，这种洗牌就不合适了。比如：样本是新闻报道，是按发表时间排序的，那么“洗牌”将导致模型过拟合，得到一个虚高的打分（因为验证过程中的样本会因为洗牌导致和训练集相似）。

一些交叉验证的迭代器，比如KFold,有内置的洗牌选项：
+ 这个并不会消耗很多内存
+ 缺省情况下并不会“洗牌”
+ random_state 参数缺省是“None”,就是说每次洗牌都不同，相当于迭代KFold(..., shuffle=True)多次，但random_state都不同。
+ 为了得到可复现的结果，可是赋值random_state为一个整形数

>### 4. 交叉验证和模型选择

交叉验证迭代器也会被直接用于网格搜索(Grid Search)寻找模型最优超参数的过程。这个主题会在下个章节介绍。

***
> 参考：

>> http://scikit-learn.org/stable/modules/cross_validation.html

> 欢迎指正
>> <yuefeng_liu@foxmail.com>


```python

```
