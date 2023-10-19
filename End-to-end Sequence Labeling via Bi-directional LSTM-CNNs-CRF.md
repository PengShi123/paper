# End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
## Abstract：
* 传统的标签序列系统要求对数据进行预处理，该论文介绍一种不需要对数据进行预处理的模型，模型跑在数据集POS和NER上。
## 1、Introduction
* 传统的性能好的序列标签模型一般使用线性统计模型，包括HMM,CRF。
* 因为论文中的模型不需要对数据进行预处理，所以该模型能够很容易的应用在不同语言和领域的序列标记任务。
* 整个模型大概的流程：
  * 首先使用CNN将输入的单词编码成字符级的表现形式。
  * 然后将字符级和单词级的表征联合起来，再将联合后的结果放到BLSTM中对每个单词的上下文进行建模。
  * 将BLSTM的结果和CRF进行结合解码整个句子的标签。
* 模型所做的贡献：
  * 为语义序列标签提供一个新的神经网络结构。
  * 给出了该模型在两个经典NLP任务的基准数据集上的经验评估。
  * 实现了真正意义上的高性能端到端系统。
## 2、Neural Network Architecture
### 2.1、CNN for Character-level Representation
* 只使用字符的编码作为CNN的输入，输入中没有字符类型的特征。
* 在将字符进行编码输入到CNN前先进行dropout。
* ![Figure1](/DeepLearing/paper/image/Figure1.png)
### 2.2 Bi-directional LSTM
#### 2.2.1 LSTM Unit
* RNN通过循环捕获时间动态。
* RNN能够捕捉上距离依赖，但是效果不好是因为存在梯度消失和梯度爆炸。
* LSTM是RNN的变种是设计出来解决梯度消失的问题。
 * 一个LSTM是由三个乘法门组成，乘法门负责控制信息的比例忘记然后过度到下一步的时间。
 * ![Figure2](/DeepLearing/paper/image/Figure2.png)
 * it = σ(W iht−1 + U ixt + bi)
 * ft = σ(W f ht−1 + U f xt + bf )
 * ̃ct = tanh(W cht−1 + U cxt + bc)
 * ct = ft ct−1 + it ̃ct
 * ot = σ(W oht−1 + U oxt + bo)
 * ht = ot tanh(ct)
 * σ是逻辑回归函数。
 * Xt是在t时刻输入的向量。
 * ht是隐藏层的输出向量，存储t时刻及之前的有用的信息。
 * U i, U f , U c, U o：对于输入Xt不同门的不同权重矩阵。
 * W i, W f , W c, W o：隐藏状态ht. 
 * bi, bf , bc, bo的权重矩阵表示偏差矩阵。
##### 2.2.2 BLSTM
* BLSTM对访问过去和未来的上下文是有益的，而LSTM的隐藏层状态只来自过去，与过去无关。
* BLSTM的想法：
  * 将每个序列的前向和后向呈现为两个独立的隐藏层状态，以此来捕获过去和未来的信息。
  * 将两个隐藏层的最终状态结合起来作为最后的输出
### 2.3CRF
* 考虑邻域中标签之间的相关性并联合解码给定输入句子的最佳标签链是有益的
* Zi:代码输入序列的第i个词的向量
* y：表示z所对应标签的序列
* Y（z）：表示可能标签序列的集合
* 条件概率公式：![p（y|z；W，b）](/DeepLearing/paper/image/p（y|z；W，b）.png)
* ![ψi](/DeepLearing/paper/image/ψi.png)
* ![L(W,b)](/DeepLearing/paper/image/L(W,b).png)
* ![y∗](/DeepLearing/paper/image/y∗.png)
#### 2.4 BLSTM-CNNs-CRF
* 字符级表示计算用CNN
  * 将字符级向量和单词嵌入向量连接输入到BLSTM网络中
  * BLSTM输出向量输入到CRF层和解码器结合输出最好的标签序列
  * dropout层在BLSTM的输入和输出都用到了
  * ![Figure3](/DeepLearing/paper/image/Figure3.png)
## 3、Network Training
### 3.1、Parameter Initialization
* Word Embedding：通常情况下使用GloVe 100-dimensional embeddings
* 论文同样也在两个其他的集中进行实验：
  * Senna 50-dimensional embeddings
  * Google's Word2Vec 300-dimensional embeddings
### 3.2、Optimization Algorithm
* 参数优化使用mini-batch SGD
* batch size：10
* momentum = 0.9
* learning rate = η0，在POS数据集中η为0.01，在NER数据集中为0.015
* ηt = η0/（1+ρt）
* decay rate ρ= 0.05，t为完成迭代的数量
* 梯度阈值为5.0
* Early Stopping
  * 最好的参数出现在迭代50次时
* Fine Tuning
  * 通过反向传播，在神经网络模型梯度更新期间，对初始化的嵌入层进行微调
  * 可以有效的解决语序和结构预测问题    
* Dropout Training ：使用Dropout是为了防止过拟合
  * 在将数据输入到CNN前和BLSTM进行输入输出时进行dropout、
### 3.3 Tuning Hyper-Parameters
* 由于时间限制在整个超参空间进行随机搜索是不可行的，论文中是将大部分最后的超参数设置为相同的，除了最开始的学习率
* 论文设置LSTM的状态大小为200
## 4 Experiments
### 4.1 Data Sets
* 评估模型的两个任务是POS tagging 和 NER
* POS tagging：
  * 在于English POS tagging论文中使用PTB的WSJ，里面包括45个不同的POS tags
* 为了与之前的论文进行比较，文章采用了标准的数据划分：0-18的部分作为训练数据，19-21的部分作为发展数据，22-24的部分作为测试集
### 4.2 Main Results
* 通过消融研究来剖析我们神经网络架构的每个组件（层）的有效性
* 论文比较了三种baseline系统BRNN,BLSTN,BLSTM-CNNs，上述模型都使用标准的GloVe 100 dimensional word embeddings和相同的超参数
  * BLSTM的表现效果比BRNN好，在两个任务上
  * BLTSM-CNN的表现比BLSTM好，字符级别表示对于语义序列标签的重要性的任务
  * 最后增加CRF会得到一个显著的提升
### 4.3、Comparsion with Previous Work
#### 4.3.1、POS Tagging
* 作者使用CNN建模字符级表征
* 通过作者的比较得出结论：对于模型化的顺序数据BLSTM是高效的
* 联合解码结构化预测模型是重要的
#### 4.3.2 NER
* Chiu and Nichols (2015)的模型与作者的模型无法进行对比，因为他们的最终的模型是训练和发展数据集的联合数据上进行的
* 该模型比其他模型好的原因是：它没有对数据进行预处理就得到了一个较好的结果
## 5、Related Work
* Huang et al.(2015)与该文的模型差异在于：
  * 没有利用CNN建模字符级信息
  * 没有实现端到端的系统，因为他们的数据是经过预处理的
* Chiu and Nichols(2015)与该文的模型差异在于：
  * 论文中的模型使用了CRF
  * 前者也进行了数据的预处理
## 6、Conclusion
* 该模型真正实现了端到端的模型
* 作者认为有几个潜在方向：
  * 论文中提到的模型可以探索更多多任务方法，通过联合更多有用的和相关联的信息
  * 另一个方向是将模型用到其他的方面




