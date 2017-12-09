[TOC]

## scikit-learn介绍

[scikit-learn](http://scikit-learn.org/)是Python的一个机器学习库，目前最新版本为0.19.1，该库特点如下：

- 用来做数据挖掘、数据分析等简单、高效
- 每个人都可以获取到，而且可以很容易的复用在自己的代码或产品中
- 底层主要使用Numpy、Scipy和Matplotlib
- 开源，使用BSD协议（知道各种开源协议的人应该知道该协议是对商业特别友好的一个协议）

目前的几大功能块：

- **Classification**，Identifying to which category an object belongs to. 
- **Regression**，Predicting a continuous-valued attribute associated with an object. 
- **Clustering**，Automatic grouping of similar objects into sets.
- **Dimensionality reduction**，Reducing the number of random variables to consider.
- **Model selection**，Comparing, validating and choosing parameters and models.
- **Preprocessing**，Feature extraction and normalization.

## 数字识别实例

下面我们通过一个识别书写数字识别的例子来学习一下scikit-learn，或者应该说是机器学习不研究步骤吧。

### 加载数据

加载数据是我们做机器学习的第一步。一般来说会有专门收集数据的人，我们只负责去研究数据就可以。如果只是学习的话，现在有很多地方也都有大量的数据，我们可以拿来学习，比如[UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets)、[Kaggle](https://www.datacamp.com/community/tutorials/www.kaggle.com)网站等。庆幸的是，scikit-learn库中就带了很多数据，方便我们用来学习。

这里以手写数字数据集为例介绍在scikit-learn中如何导入数据集：

```python
from sklearn import datasets

digits = datasets.load_digits()
```

datasets模块中有很多`load`开头的方法，都是用来导入各种数据集的。

### 探索数据

#### 获取数据信息

数据加载完之后，我们接着要做的事情就是了解数据集。在scikit-learn中，每个数据集都有一个`keys()`方法，调用该方法可以看到数据集有哪些属性，每个数据集都有一个`DESCR`属性，用来详细的描述该数据集。

```python
# 打印所有属性
print(digits.keys())
# 输出为 dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

# 查看该数据集的描述
print(digits.DESCR)

# 查看特征数据
print(digits.data)
print(digits.shape)

# 查看目标变量或者称为标签
print(digits.target)
print(digits.target_names)

# 查看图像数据
print(digits.images)
print(digits.shape)
```

一般我们需要关注所有矩阵的shape，从而得知有多少条数据、多少个维度、多少个标签等。这里我们需要关注一下`data`和`images`两个属性。通过查看shape属性我们可以得知data的shape为`(1797, 64)`，而images的属性为`(1797, 8, 8)`，其实images只是把data中的64表示成了一个`8*8`的图片而已，他们的数据是一样的，我们通过下面的测试就可以证实：

```python
import numpy as np

np.all(digits.images.reshape(1797, 64) == digits.data)  # 输出为True
```

#### 使用matplotlib展示数据

我们可以使用`matplotlib`库将数据进行可视化，从而更好的观察数据：

```python
from sklearn import datasets

digits = datasets.load_digits()

# Import matplotlib
import matplotlib.pyplot as plt

# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

# Show the plot
plt.show()
```

展示结果如下：

![digits_1](/Users/allan/git/allan/private/machine-learning-tutorials/ScikitLearnTutorial/digits_1.png)

#### 降维展示数据

我们从数据中看到该数据集中有64个特征，这么多的维度往往是灾难性的，在机器学习中维度太多往往会产生很多问题，一般我们将这种情况将[维度灾难](https://en.wikipedia.org/wiki/Curse_of_dimensionality)（curse of dimensionality）。这个时候我们往往通过一些手段来降维（Dimensionality reduction），降维技术也可以说是机器学习中的一个分支。这里我们使用比较常用的[主成分分析](https://en.wikipedia.org/wiki/Principal_component_analysis)（PCA，principal components analysis）法降低本例中的数据维度（降到2维），然后展示出来。

```python
from sklearn import datasets
from sklearn.decomposition import PCA 

digits = datasets.load_digits()

# Create a Randomized PCA model that takes two components, it performs
# better when there's a high number of dimensions
randomized_pca = PCA(svd_solver='randomized', n_components=2)   

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Create a regular PCA model 
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

# Inspect the shape
reduced_data_pca.shape

# Print out the data
print(reduced_data_rpca)
print(reduced_data_pca)
```

我们将降维后的数据展示出来：

```python
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA 

digits = datasets.load_digits()

# Create a Randomized PCA model that takes two components
randomized_pca = PCA(svd_solver='randomized', n_components=2)   

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Create a regular PCA model 
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

# Inspect the shape
reduced_data_pca.shape

# Print out the data
print(reduced_data_rpca)
print(reduced_data_pca)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_rpca[:, 0][digits.target == i]
    y = reduced_data_rpca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()
```

结果图如下：

![digits_2](/Users/allan/git/allan/private/machine-learning-tutorials/ScikitLearnTutorial/digits_2.png)

### 数据预处理

在探索我们的数据后，开始使用算法建模型之前，我们还需要对数据做一些预处理，这里我们讲两个最常见的预处理：归一化(normalization)和划分数据集。

#### 归一化

数据归一化往往可以提高模型的收敛速度和模型的精度，常见的归一化有两种：

- 线性归一化，把输入数据都转换到[0 1]的范围。公式为：
	
	$Xnorm = \frac{X-Xmin}{Xmax-Xmin}$
	
- 0均值归一化，把原始数据集归一化为均值为0、方差1的数据集。公式为：

	$Xnorm = \frac{X-μ}{σ}$

	其中，μ、σ分别为原始数据集的均值和方法。该种归一化方式要求原始数据的分布可以近似为高斯分布，否则归一化的效果会变得很糟糕。
	
scikit-learn中可使用下面方法进行0均值归一化：

```python
# Import
from sklearn.preprocessing import scale

# Apply `scale()` to the `digits` data
data = scale(digits.data)
```

#### 划分数据集

之前我们就说过，一般会将数据集安装八二或者七三划分为训练集和测试集，训练集用来训练模型，测试集用来测试模型的正确率。scikit-learn提供了划分的方法：

```python
from sklearn.model_selection import train_test_split

# Split the `digits` data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)
```

前面几个是要划分的数据集；`test_size`表示测试数据所占的比例，此时是0~1之间的浮点数，如果是整数，则带包测试数据的条数；`random_state`是随机种子。

### 选择建模算法

一般对于一个做机器学习已经很有经验的人来说，根据数据情况以及场景大概可以判断出来该使用哪个或哪类或者哪些算法，但这对于新手来说往往是困难的。为此scikit-learn总结了一个地图，即使没有经验，我们根据这个地图也能简单的对算法进行选择：

![算法地图](/Users/allan/git/allan/private/machine-learning-tutorials/ScikitLearnTutorial/ml_map.png)

#### SVC

根据地图我们可以看到我们应该选择的模型是linear SVC，该算法属于监督学习中分类算法中的一个，这里我们先不关注算法细节，先看如何使用。

```python
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
data = scale(digits.data)

# Split the `digits` data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

from sklearn import svm
# Create the SVC model 
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')

# Fit the data to the SVC model
svc_model.fit(X_train, y_train)

# Predict the label of `X_test`
print(svc_model.predict(X_test))

# Print `y_test` to check the results
print(y_test)

# Apply the classifier to the test data, and view the accuracy score
print(svc_model.score(X_test, y_test))
```

这里我们建立了一个模型`svc_model`，并用它来预测测试数据，并且查看了它的准确性。我们也可以借助`Isomap()`来图形化的显示测试结果与实际结果：

```python
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
data = scale(digits.data)

# Split the `digits` data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

from sklearn import svm
# Create the SVC model 
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')

# Fit the data to the SVC model
svc_model.fit(X_train, y_train)

from matplotlib import pyplot as plt
# Import `Isomap()`
from sklearn.manifold import Isomap

# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
predicted = svc_model.predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust the layout
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Labels')


# Add title
fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')

# Show the plot
plt.show()
```

输出为：

![对比图](/Users/allan/git/allan/private/machine-learning-tutorials/ScikitLearnTutorial/digits_3.png)

#### Clustering

Clustering即聚类，是非监督学习算法中的一种。对于上面的手写数据，假设我们不知道label，那我们就可以使用该算法。从前面的地图知道我们应该使用KMeans算法，和前面一样，先不关注算法细节，先看如何使用：

```python
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

digits = datasets.load_digits()
data = scale(digits.data)

# Split the `digits` data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

from sklearn import cluster

# Create the KMeans model
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

# Fit the training data to the model
clf.fit(X_train)
```

我们画出KMean算法计算出的中心：

```python
# 接上面代码

# Import matplotlib
import matplotlib.pyplot as plt

# Figure size in inches
fig = plt.figure(figsize=(8, 3))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')

# Show the plot
plt.show()
```

如图：

![中心](/Users/allan/git/allan/private/machine-learning-tutorials/ScikitLearnTutorial/digits_4.png)

然后预测数据：

```python
# 接上面代码

# Predict the labels for `X_test`
y_pred=clf.predict(X_test)

# Print out the first 100 instances of `y_pred`
print(y_pred[:100])

# Print out the first 100 instances of `y_test`
print(y_test[:100])

# Study the shape of the cluster centers
print(clf.cluster_centers_.shape)
```

不是非常直观，我们继续使用`Isomap()`画出对比图：

```python
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

digits = datasets.load_digits()
data = scale(digits.data)

# Split the `digits` data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

# Import the `cluster` module
from sklearn import cluster

# Create the KMeans model
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

# Fit the training data to the model
clf.fit(X_train)

# Predict the labels for `X_test`
y_pred=clf.predict(X_test)

# Print out the first 100 instances of `y_pred`
print(y_pred[:100])

# Print out the first 100 instances of `y_test`
print(y_test[:100])

# Study the shape of the cluster centers
clf.cluster_centers_.shape

# Import `Isomap()`
from sklearn.manifold import Isomap

# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')

# Show the plots
plt.show()
```

输出结果为图：

![对比图](/Users/allan/git/allan/private/machine-learning-tutorials/ScikitLearnTutorial/digits_5.png)