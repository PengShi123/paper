# CNN
## Abstract
* 摘要对文章所想表达的东西进行了大致的阐述：可以通过很少的超参数调整和静态向量实现句子分类任务。
* 通过微调学习特定于任务的向量可以进一步提高性能。与此同时对架构进行了简单的修改，以允许同时使用特定于任务的向量和静态向量。
* 讨论的问题是：情感分析和问题分类

## 1.Introduction
* CNN是利用带有卷积滤波器的层来得到特征的
* 从预训练的深度学习模型获得的特征提取器在各种任务中表现良好

## 2.Model
* 模型图
  * ![模型图](/DeepLearing/paper/image/CNN-1.png)
* 处理数据过程
  * 将Xi连接起来
  * 将连接起来的Xi与权重矩阵（这里指滤波器）做点积
  * ![Ci](/DeepLearing/paper/image/Ci.png)
  * 将上述的点积加上偏差b，带入函数中求得Ci
  * 求出多个Ci得到一个feature map：C
  * 对每个C进行最大池化操作得到C中的最大值
  * 这个最大值所对应的特征是最重要的特征
  * 多个重要的特征被传入全连接的softmax层输出在每个标签上的可能性
## 2.1Regularization
* ![y](/DeepLearing/paper/image/y.png)
* 。逐元素乘法运算符
* r是掩码向量，梯度仅通过未屏蔽单元反向传播
* 在测试时，学习到的权重向量按 p 缩放，使得 ^w = pw，并使用 ^w（没有 dropout）对看不见的句子进行评分。
* 通过在梯度下降步骤之后重新缩放 w 以具有 ||w||2 = s 来约束权重向量的 l2 范数。
## 3.Datasets and Experimental Setup
* MR（Movie reviews）: 电影评论，每条评论一个句子，分类涉及检测正面/负面评论
* SST-1（Stanford Sentiment Treebank）：斯坦福情绪树库，MR 的扩展，但使用提供的训练/开发/测试拆分和细粒度标签（非常积极、积极、中性、消极、非常消极）
* SST-2： 与 SST-1 相同，但删除了中性评论和二进制标签
* Subj（Subjectivity dataset）:任务是将句子分类为主观或客观的主观性数据集
* TREC（TREC question dataset）：TREC问题数据集，任务涉及将问题分类为 6 种问题类型（问题是关于人、位置、数字信息等）
* CR（Customer reviews of various products）：各种产品的客户评论，任务是预测正面/负面评论
* MPQA：MPQA数据集的意见极性检测子任务
### 3.1 Hyperparameters and Training
* 滤波器窗口 (h) 为 3, 4, 5，每个特征图有 100 个，dropout rate (p) 为 0.5，l2 约束 (s) 为 3，小批量大小为 50
* 对于没有标准开发集的数据集，随机选择 10% 的训练数据作为开发集
* 使用Adadelta对每个batch进行更新
### 3.2 Pre-trained Word Vectors
* 使用从无监督神经语言模型获得的词向量初始化词向量是一种流行的方法，可以在没有大型监督训练集的情况下提高性能
* 使用word2vec vectors ，向量维度为300，bag-of-words实现
* 未出现在预训练单词集中的单词是随机初始化的
### 3.3 Model Variations
* CNN-rand：所有单词是随机初始化的，然后在训练期间进行修改。
* CNN-static：具有 word2vec 预训练向量的模型，所有单词——包括随机初始化的未知单词——都保持静态，只学习模型的其他参数。
* CNN-non-static：与上述相同，但针对每个任务微调预训练的向量
* CNN-multichannel：具有两组词向量的模型。每组向量都被视为一个“通道”，并应用每个过滤器到两个通道，但梯度仅通过其中一个通道反向传播。因此，该模型能够在保持其他静态的同时微调一组向量。两个通道都使用 word2vec 初始化。
* 通过在每个数据中保持一致来消除其他随机性来源——CV 折叠分配、未知词向量的初始化、CNN 参数的初始化。
## 4 Results and Discussion
* 所有随机初始化单词的基线模型（CNN rand）本身表现不佳。虽然期望通过使用预先训练的向量来获得性能增益，但增益的幅度很大。即使是具有静态向量的简单模型也表现得非常好
### 4.1 Multichannel vs. Single Channel Models
* 作者最初希望多通道架构能够防止过拟合（通过确保学习的向量不会偏离原始值太远），从而比单通道模型更好地工作，尤其是在较小的数据集上，然而结果不好
### 4.2 Static vs. Non-static Representations
* 与单通道非静态模型的情况一样，多通道模型能够微调非静态通道，使其更适合手头的任务
* 对于不在预先训练的向量集中的（随机初始化的）标记，微调可以让它们学习更有意义的表示
### 4.3 Further Observations
* Kalchbrenner等人（2014）报告了CNN的更糟糕的结果，该CNN的架构与我们的单通道模型基本相同
  * 将这种差异归因于我们的CNN具有更大的容量（多个滤波器宽度和特征图）
* Dropout被证明是一个很好的正则化器，因此使用一个比必要的网络更大的网络并简单地让Dropout正则化是可以的
  * Dropout始终增加2%-4%的相对性能
  * 当随机初始化不在word2vec中的单词时，通过对U[−a，a]中的每个维度进行采样来获得轻微的改进，其中选择了a，使得随机初始化的向量与预训练的向量具有相同的方差
## 5、Conclusion
* 作者的结论：研究结果进一步证明了单词向量的无监督预训练是NLP深度学习的重要组成部分