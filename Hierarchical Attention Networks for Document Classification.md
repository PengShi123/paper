# Hierarchical Attention Networks for Document Classification
## Abstract
* 作者针对文章分类问题提出了一种多层注意力网络
* 该模型有两个独特的特点：
  * 它具有反映文档层次结构的层次结构的层次结构
  * 它有两个级别的注意力机制应用于单词和句子级别，使其能够在构建文档表示时以不同的方式关注越来越重要的内容
* 论文包含了六个大规模的文本分类任务、
* 注意力层的可视化说明了该模型选择定性信息丰富的单词和句子
## 1、Introduction
* 模型目标是给文本一个独有的标签
  * 话题标签
  * 情感分类
  * 垃圾邮件检测
* 传统的文本分类方法表示具有稀疏词汇特征的文档
* 最近使用更多的是CNN和LSTM 
* 论文中做了测试可以通过在模型架构中加入文档结构的知识来获得更好的表示。
* 确定相关部分涉及对单词的交互进行建模，而不仅仅是它们孤立地的存在。
* 同样通过构建句子的表示然后将它们聚合到文档表示来构建文档表示。、
* 论文中的模型包括两个层次的注意机制
  * 一个是句子级别的一个是单词级别的
  * 目的是让模型或多或少地关注单个单词和句子
* 注意力机制的好处：
  * 有很好的性能表现
  * 它也提供了对哪些单词和句子有助于分类决策的洞察，这在应用程序和分析中可能是有价值的
* 与其他模型的不同之处在于：
  * 当一系列标记是相关的，而不是简单地过滤（序列）标记，取自上下文，系统使用上下文来发现
## 2 Hierarchical Attention Networks
* 组成：
  * 单词序列编码器
  * 单词级别的注意力层
  * 句子编码器
  * 句子级别的注意力层
### 2.1 GRU-based sequence encoder
* GRU使用门控机制来跟踪序列的状态，而不使用单独的存储单元
* GRU使用两个gate：reset gate和updata gate
### 2.2 Hierarchical Attention
## 3 Experiments
### 3.1 Data sets
* 数据类型可以被分为两类：情感估计和话题分类
* 模型使用80%作为训练集，10%作为验证集，10%作为测试集
* 数据集包含以下几个：
  * Yelp reviews
  * IMDB reviews
  * Yahoo answers：包含10个话题分类
  * Amazon reviews
### 3.2 Baselines
* 论文将HAN与几个常见的baseline进行比较：
  * linear methods
  * SVMs
  * paragraph embeddings using neural networks
  * LSTMs
  * word-based CNN 
  * character-based CNN
  * Conv-GRNN
  * LSTMGRNN
#### 3.2.1 Linear methods
* 基于多项式逻辑回归
* 数据集：
  * BOW and BOW+TFIDF
  * n-grams and n-grams+TFIDF
  * Bag-of-means
#### 3.2.2 SVMs
* 方法包括：
  * SVM+Unigrams
  * Bigrams,
  * Text Features
  * AverageSG
  * SSWE
* 数据集：
  * Text Features
  * AverageSG
  * SSWE
#### 3.2.3 Neural Network methods
* CNN-word
* CNN-char
* LSTM
* Conv-GRNN and LSTM-GRNN
### 3.3 Model configuration and training
* 使用CoreNLP将文章切分为sentence和token
* 词汇表中只包含文章中出现五次的单词，并使用一种特殊的UNK token在替换这些单词
* 构造一个word embeding通过训练一个无监督的word2vec
* 在模型超参数方面：
  * 单词嵌入层维度为200
  * GRU的维度为50
  * 联合前向和反向GRU
* 在训练方面：
  * 使用一个mini—bath64和微长的文章
  * 长度调整可以将训练速度提高了三倍
  * 模型使用随机梯度下降训练所有模型，momentum为0.9
  * 论文使用grid search找到最合适的学习率
### 3.4 Results and analysis
* 论文的参考模型为：HN-{AVE, MAX, ATT}
  * AVE表示平均：HN-AVE 等效于使用非信息性全局词/句子上下文向量
  * MAX表示最大池化的结果
  * ATT表示论文提到的多层注意力机制
* 结果是：HN-ATT得到了最好的成绩
* 与HA-AVG相比证明了所提出的全局词和句子重要性向量对 HAN 的有效性。
### 3.5 Context dependent attention weights
* 如果单词有自己的权重，模型没有自注意力机制会工作的更好，因为该模型可以自动为不相关的词分配低权重
* 但是单词的重要性是依赖上下文
* 文章对good这个词和bad这个词在上下文的依赖性做了可视化
  * 从good词的结果可得：模型捕捉不同的上下文，并为单词分配上下文相关的权重。
  * 随着评价分数的上升，good在好评论的权重比例会增大，bad会随着分数的上升权重比例减小
### 3.6 Visualization of attention
* 由于层次结构，可以通过句子权重来规范单词权重，以确保只强调重要句子中的重要单词。
* √ps术语显示不重要句子中的重要单词，以确保它们不是完全看不见的。
* 图像表面模型不仅可以挑选出情感色彩重的词语，也可以处理复杂的上下文关系
## 4、Related Work
* 论文主要举了几个例子来介绍模型是如何判定情感重要色彩重的词，如何判定哪些词哪些句子会对文章的分类做出影响
## 5、Conclusion
* 文章提出了HAN来对文档进行分类
* 模型借助信息量大的组件获得了很好的可视化效果
* 模型将重要的单词聚合成句子向量
* 将重要的句子聚合成文档向量