# Abstract
* 虽然Dropout有效和表现良好，但 dropout 引入的随机性会导致训练和推理之间的不一致不可忽略。
* R-Drop迫使由dropout生成的不同子模型的输出分布彼此一致。具体而言，对于每个训练样本，R-Drop使通过dropout采样的两个子模型的输出分布之间的双向KL偏差最小化。R-Drop减少了上述不一致性。
# 1 Introduction
* 在训练深度神经网络时，正则化技术对于防止过度拟合和提高深度模型的泛化能力是必不可少的。其中，使用最广泛的dropout technique旨在防止协同适应，并通过在训练过程中简单地从神经网络中丢弃一定比例的隐藏单元来执行隐式集成。通过对不一致的隐藏状态施加L2正则化，这方法可以在一定程度上缓解不一致问题，但远未被广泛使用。、
* 在每个小批量训练中，每个数据样本经过两次正向传递，每个传递由不同的子模型通过随机丢弃一些隐藏单元来处理。R-Drop迫使相同的两个分布通过最小化两个分布之间的双向Kullback-Leibler（KL）偏差。R-Drop对训练中的每个数据样本从dropout中随机采样的两个子模型的输出进行正则化。与传统神经网络训练中的丢弃策略相比，R-Drop只增加了KL发散损失，而没有任何结构修改。
* 与以前大多数只处理每一层的隐藏单元或模型参数的方法不同。R-Drop同时适用于隐藏单元和通过dropout采样的子模型的输出。
* contributions：
	* 1、propose R-Drop.
	* 2、R-Drop can reduce the inconsistency between training and inference of the dropout based models.
	* 3、R-Drop achieves extremely strong performances, including multiple SOTA results.
# 2 Approach
* training dataset $D= {(x_i, y_i)}^n_i=1$,训练的目标是学习模型$P^w（y|x）$n是训练样本的数量,$（x_i，y_i）$是标记的数据对,$x_i$是输入数据，$y_i$是标签,映射函数的概率分布也表示为$P^w（y|x）$,两个分布$P_1$和$P_2$之间的Kullback-Leibler（KL）散度由$D_{KL}（P1||P2）$表示.
## 2.1 R-Drop Regularization
* 深度学习模型的主要学习目标是最小化负对数似然损失函数:
	* $L_{nll} = 1/n∑i=1n− log P^w(y_i|x_i).$
* 训练阶段采用具有随机丢弃单元的子模型，而推理阶段采用没有丢弃单元的完整模型,此外，由随机采样的丢弃单元引起的子模型在没有任何约束的情况下也是不同的。
* input data $x_i$,馈送xi两次通过网络的正向通路,获得模型预测的两个分布:$P^w_1（y_i|x_i）$和$P^w_2（y_i|x_i）$由于丢弃算子随机丢弃模型中的单元，因此两次前向传递确实基于两个不同的子模型（尽管在同一模型中,用于输出预测$P^w_1（y_i|x_i）$的左路径的每一层中的丢弃单元不同于用于输出分布$P^w_2（y_i|x_i）$的右路径的丢弃单元。然后，在这个训练步骤中，R-Drop方法试图通过最小化同一样本的这两个输出分布之间的双向Kullback-Leibler（KL）散度来正则化模型预测:
	* $L^i_{KL} = 1/2 (D_{KL}(P^w_1（y_i|x_i）||P^w_2（y_i|x_i）) + D_{KL}P^w_2（y_i|x_i）||P^w_1（y_i|x_i)))$
* 利用两次前向传球的基本负对数似然学习目标$L_{nll}^i$：
	* $L_{nll}^i$ = $− log P^w_1（y_i|x_i） − log P^w_2（y_i|x_i）$,
* 最终训练目标是最小化数据的Li（xi，yi）：
	* $L_i = L^i_{NLL} + α · L^i_{KL} = − log P^w_1（y_i|x_i） − log P^w_1（y_i|x_i）+ α/2 [DKLP^w_1（y_i|x_i）||P^w_1（y_i|x_i）) + DKL(P^w_1（y_i|x_i）||P^w_1（y_i|x_i）)],$
	* $α$是控制$L^i_{KL}$的系数权重
## 2.2 Training Algorithm
* Input: Training data $D= {(x_i, y_i)}^n_i=1$,
* Output: model parameter $w$.
* 1: Initialize model with parameters $w$.
* 2: while not converged do
	* 3: randomly sample data pair $(x_i, y_i) ∼ D$,
	* 4: repeat input data twice as $[x_i; x_i]$ and obtain the output distribution $P^w_1（y_i|x_i）$和$P^w_2（y_i|x_i）$
	* 5: calculate the negative log-likelihood loss $L_{nll}^i$ by Equation (3)
	* 6: calculate the KL-divergence loss  $L^i_{KL}$ by Equation (2),
	* 7: update the model parameters by minimizing loss $L^i$ of Equation (4).
* 8: end while
## 2.3 Theoretical Analysis
* R-Drop增强训练通过强制子结构相似来减少这种不一致性
## 2.4 Discussion
* R-Drop has key differences with ELD  and FD:
	* （1） 差距控制来自不同的观点。ELD致力于直接缩小有遗漏的子模型（训练）和没有遗漏的预期完整模型（推理）之间的差距，而R-Drop和FD都致力于惩罚子模型之间的差异，在FD中已经证明了正则化子模型的优越性。
	* （2） 正则化效率不同。ELD只在没有完整模型的情况下通过子模型反向传播梯度，这比更新两个子模型的R-Drop效率低。
	* （3）正则化效果不同。ELD和FD都使用隐藏状态上的L2距离作为正则化损失函数。然而，这与最小化模型输出分布上的负对数似然的主要训练目标相去甚远。由于log softmax对优化有很大影响，因此隐藏状态的距离与概率分布不在同一空间内。相比之下，R-Drop利用输出概率分布之间的KL散度作为一致性正则化，其与训练目标在同一空间中。
# 3 Experiments
## 3.1 Application to Neural Machine Translation
* Datasets：
	* 低资源场景的数据集来自IWSLT比赛，其中包括IWSLT14英语↔德语（En↔De），英语↔西班牙语（En↔Es）和IWSLT17英语↔法语（En↔Fr），英语↔中文（En↔Zh）翻译。
	* 丰富的资源数据集来自公认的WMT翻译任务，采用WMT14英语→德语和英语→法国任务。
	* IWSLT数据集包含约170k个训练语句对、7k个有效语句对和7k个测试语句对。WMT数据大小为4.5M，En为36M→De和En→有效数据和测试数据分别来自相应的新闻测试数据。
* Model & Training：
	* 以最流行的Transformer网络作为模型结构。transformer_iwslt_de_en和transformer_vaswani_wmt_en_de_big分别是iwslt和wmt翻译的配置。对于所有翻译任务，权重α设置为5。实现是在Fairseq上开发的。
## 3.2 Application to Language Understanding
* Dataset：
	* 通过微调预先训练的模型3来进一步评估提出的语言理解任务的方法，这些模型是GLUE基准的标准开发集。GLUE基准包括8个不同的文本分类或回归任务，它们是MNLI、MRPC、QNLI、QQP、RTE、SST-2、STS-B（回归）和CoLA。
* Model & Training：
	* BERT-base and strong RoBERTa-large
	* 对于每个设置，在{0.1，0.5，1.0}之间动态调整系数α。
	* 对于回归任务STS-B，使用MSE而不是KL散度来正则化输出
## 3.3 Application to Summarization
* Dataset：
	* the CNN/Daily Mail dataset
	* 该数据集包含从CNN和Daily Mail网站抓取的新闻文档（来源）及其相应的亮点（目标）。它包含287226份培训文件、13368份验证文件和11490份测试文件。
* Model & Training：
	* BART
	* 在该任务中，系数权重α被设置为0.7以控制KL发散。对于其他超参数，不做任何修改。
## 3.4 Application to Language Modeling
* Dataset：
	* Wikitext-103 dataset
	* WikiText-103包含来自维基百科上28K篇文章的约103M个训练令牌，每篇文章的平均令牌长度约为3.6K
*  Model & Training：
	* 一种是基本的Transformer解码器，另一种是更先进的：自适应输入变换器，它将自适应输入嵌入引入到变换器模型中。
	* 使用开源的Fairseq工具包，对应的模型配置为transformer和Adaptive Input transformer的transformer_lm_gpt和transformer_lm_wiki103。我们只需将权重α设置为1.0，而无需在训练过程中进行调整。
## 3.5 Application to Image Classification
* Dataset：
	* CIFAR-100  and the ILSVRC-2012 ImageNet dataset
	* CIFAR-100数据集由100个类别的60k幅图像组成，每个类别有600幅图像，其中500幅用于训练，100幅用于测试。
	* ImageNet数据集由1000个分类类的130万个图像样本组成
* Model & Training：
	* Vision Transformer (ViT)
	* 采用两个公开发布的预训练模型，ViT-B/16和ViT-L/16，分别具有86M和307M个参数，并在CIFAR-100和ImageNet数据集上进行模型微调。在微调过程中，两个模型的权重α都设置为0.6
# 4 Study
## 4.1 Regularization and Cost Analysis
* 随着训练的进行，Transformer很快变得过拟合，训练与Transformer的有效损耗之间的间隙很大，而R-Drop的有效损耗较低。这很好地证明了R-Drop可以在训练期间提供持久的正则化。在早期训练阶段，Transformer快速提高了BLEU得分，但很快就会收敛到糟糕的局部最优。相比之下，R-Drop逐渐提高了BLEU分数，并实现了更优越的性能。尽管它需要更多的训练才能收敛，但最终的最优效果更好。这与其他正则化方法相同。R-Drop确实增加了每一步的训练成本，因为它需要在小批量中重复输入x进行另一次计算。注意，这类似于没有KL分歧的批量大小加倍训练。
## 4.2 $k$-step R-Drop
* R-Drop可以获得更强的性能，但收敛性较低，因此研究了另一种训练策略，即每k步执行一次R-Drop，以提高训练效率，而不是在每一步都应用。我们在{1,2,5,10}中改变k以查看差异，其中k=1是当前的训练策略。有效BLEU曲线以及训练更新次数和训练时间如图3所示。从曲线中，可以得出结论，尽管k越大，收敛速度越快，但训练不能陷入良好的最优状态，这会迅速过拟合，并且当增加k时，BLEU分数变得越来越差。这证明了在每一步的R-Drop都能很好地规范训练，并获得优异的性能。
## 4.3 m-time R-Drop
* IWSLT14-De的BLEU分数→当m=3时，En测试集为37.30，与m=2时（37.25 BLEU得分）相似。这反映出R-Drop在两个分布之间已经具有很强的正则化效应，而不需要更强的正则化。
## 4.4 Two Dropout Rates
在训练期间对两个输出分布使用两个不同的dropout值
* 1）相同值（0.3，0.3）的脱落率是最佳选择（当前设置）
* 2）当两个脱落率在合理范围（0.3～0.5）内时，R-Drop可以稳定地获得强大的结果，而不会有太大的性能差异。一个有趣的点是，即使两个丢弃值都是0.5，这意味着一半的单元预计会被丢弃，与基线Transformer（34.64 BLEU）相比，R-Drop仍然可以获得令人满意的结果（36.48 BLEU）
## 4.5 Effect of Weight α
* 小α（如1）的性能不如大α（如5），这意味着我们应该更多地关注KLdivergence正则化。然而，过多的正则化（α=10）也是不好的，最佳平衡选择是α=5。请注意，对于不同的任务（例如，NMT、语言理解），α的选择是不同的，这取决于每个任务的特定数据大小和模型大小导致的过度拟合的容易程度。
# 5 Related Work
* Regularization Methods：
	* many regularization techniques have been proposed, e.g., weight decay , dropout, normalization , adding noise , layer-wise pre-training and initialization , label-smoothing  and so on.
	* dropout及其变体由于其有效性和适度的成本以及与其他正则化方法的良好兼容性而最受欢迎，这些方法已成功应用于正则化广泛的神经网络9架构，例如卷积神经网络层、递归神经网络、Transformer。
	* 丢弃方法的成功可以通过防止神经元的共同适应和执行子模型的隐式集成来解释。由于在促进权重稀疏性和随机性方面的作用，丢弃方法也适用于其他应用，通过在训练阶段利用KL散度，鼓励从丢弃中采样的任何两个子模型对输入数据产生一致的模型预测。也就是说，我们在模型输出级别上进行正则化。在这样做的过程中，由丢弃的随机性产生的子模型输出被正则化，以减少参数自由度，这将增强推理的泛化能力。
* Consistency Training：
	* R-Drop通过更有效的双向KL损失对输出空间的丢失进行一致性训练。与上述子模型上的一致性训练方法不同，Cutoff类似于从数据的角度启动一致性训练，通过正则化原始数据和增强样本之间的不一致性，消除输入句子中的部分信息。
* Self-distillation：
	* 最小化两个不同模型的输出分布之间的KL差异与知识提取相关，其中两个模型分别指教师和学生。在环境中，教师和学生是同一模型的辍学实例，因此它类似于自我知识提取的场景。与现有的利用模型本身的暗知识或在不同层之间提取知识的方法不同，可以被视为实例式的自知识提取，即每对采样子模型对相同的输入在彼此之间执行提取，这也与相互学习有关。
# 6 Conclusions and Future Work
* 在本文中，提出了一种基于丢弃的简单但非常有效的一致性训练方法，即R-Drop，它最小化了在模型训练中从丢弃中采样的任何一对子模型的输出分布的双向KL发散。在18个流行的深度学习数据集上的实验结果表明，R-Drop不仅可以有效地增强强模型，例如ViT、BART、Roberta large，而且在大规模数据集上也能很好地工作，甚至在WMT14 English上与vanilla Transformer相结合时也能实现SOTA性能→德语和英语→法语翻译。
* 由于计算资源的限制，对于预训练相关的任务，在这项工作中只在下游任务微调上测试了R-Drop。