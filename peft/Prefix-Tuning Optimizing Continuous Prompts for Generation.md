# Abstract
* 提出了prefix-tuning,为自然语言生成任务提供了一种轻量级的微调替代方案，它保持语言模型参数冻结，但优化了一个小的连续任务特定向量（称为前缀）。
* 作者将prefix-tuning用在了GPT2生成表到文本，用在了BATR做总结上
* 作者发现，通过只学习0.1%的参数，prefix-tuning在全数据设置中获得了可比的性能，在低数据设置中优于微调，并更好地外推到训练中未发现主题的示例。
# 1 Introduction
* 作者提出prefix-tuning的目的是解决大型预训练模型进行微调时需要对所有的参数进行调整，成本高耗时长的问题。
* 解决的思想是：冻结大部分预训练参数，用小型可训练模块扩充模型
	* adapter-tuning：在预先训练的语言模型层之间插入额外的任务特定层。
	* adapter-tuning在自然语言理解和生成基准测试方面具有很好的性能，通过微调可以获得相当的性能，同时只添加约2-4%的特定于任务的参数。
	* in-context learning or prompting：用户为任务输入准备自然语言任务指令和一些示例；然后生成来自LM的输出。
* ![[prefix_tuning.png]]
	* 任务的输入是一个线性表格。
	* Prefix-tuning预先准备一个特定任务的连续序列的向量放在输入，通常叫一个prefix，在图上用红色的条形矩阵表示。
* 与fine—tuning相反，prefix-tuning是模块化的：训练一个上游prefix，该prefix控制下游LM，该LM保持不变,因此一个LM可以控制多个任务。
* 基于prefix的体系结构使我们甚至能够在一个批次中处理来自多个用户/任务的示例。
* 在lowdata settings中，prefix-tuning在两个任务上的平均性能都优于微调。
# 2 Related Work
## Fine-tuning for natural language generation.
* 对于表格到文本生成，微调是一个序列到序列的模型（T5）
* 对于抽取和抽象总结，微调是掩码语言模型（BERT）或者编码器解码器模型（BART）
* 对于其他的任务来说，微调也是一个通用的范式
## Lightweight fine-tuning
* 轻量级微调冻结了大多数预训练的参数，并用小的可训练模块修改了预训练的模型。
* 关键的挑战是确定模块的高性能架构和要调整的预训练参数的子集。
## Prompting
* Prompting主要是预先准备好指令和输入示例，并从LM输出，这个框架被称为 *in-context learning*。
* *in-context learning*无法充分利用比*in-context learning*窗口更长的训练集。
* AutoPrompt搜索离散触发词的序列，并将其与每个输入连接，以从掩蔽的LM引出情感或事实知识。
* Continuous vectors已被用于指导语言模型。
* 经过预训练的LSTM语言模型可以通过优化每个句子的连续向量来重构任意句子，使向量输入特定。相比之下，prefix-tuning优化了应用于该任务的所有实例的特定于任务的前缀。
## Controllable generation
* Controllable generation 旨在引导预先训练的语言模型与句子级别的属性相匹配。
* 例如控制可能发生在训练的时候，对语言模型（CTRL）进行预训练，使其以元数据为条件预训练语言模型（CTRL）以对元数据（如关键字或URL）进行调节。
# 3 Problem Statement
* input is a context $x$ and the output $y$ is a sequence of tokens.在table-to-text中，$x$对应一个线性表，$y$对应文字描述。在summarization中，$x$是文章，$y$是一段总结。
## 3.1 Autoregressive LM
* 假设有一个自回归模型${P_φ}(y|x)$,参数化$φ$.$z = [x; y]$，z是x和y的concatenation
* $X_{idx}$表示对应$x$的索引序列， $Y_{idx}$表示对应$y$的索引序列
* $h_i$是第i个时间步的激活值
## 3.2 Encoder-Decoder Architecture
* ${P_φ}(y|x)$，$x$通过双向编码器进行编码，解码器预测$y$
## 3.3 Method: Fine-tuning
* 初始化$φ$,$p_φ$是一个可训练语言模型的概率分布，对数似然目标执行梯度更新：
	* $maxφ log p_φ(y | x) = ∑i∈Y_{idx}log pφ(z_i | h<i)$
# 4 Prefix-Tuning
## 4.1 Intuition
* 基于提示的直觉，我们相信拥有适当的上下文可以在不改变其参数的情况下引导LM,可以将其常见的搭配作为上下文，并且LM将为所需单词分配更高的概率.上下文可以通过指导从x中提取东西来影响x的编码，通过引导下一个token的分发来影响y的生成。
* 可以将指令优化为连续词嵌入其效果将向上传播到所有Transformer激活层，并向右传播到后续的令牌这比需要匹配真实单词嵌入的离散提示更具表现力。这比干预激活的所有层次更缺乏表现力，避免长期依赖，并包含更多可调参数。
* 因此Prefix-tuning优化的是全部神经网络层的prefix。
## 4.2 Method
* ![[model.png]]
* 前缀调优为自回归LM添加前缀以获得$z= [PREFIX; x; y]$,为编码器和编码器都添加前缀以获取$z=[PREFIX; x; PREFIX^′; y]$
* $P_{idx}$表示前缀索引的序列，|$P_{idx}$|表示前缀的长度。
* Prefix-tuning初始化一个维度为|$P_{idx}$| × dim($h_i$)的矩阵存储prefix的参数：
	* $h_i$=$\begin{cases}{Pθ[i, :] ,ifi∈P_{idx},}\\{LM_φ(z_i, h<i),otherwise.}\end{cases}$
	* $h_i$是一个以$P_θ$为自变量的函数，当i属于$P_{idx}$时，这很明显，因为$h_i$直接从$P_θ$复制。当i不属于$P_{idx}$时，$h_i$仍然依赖于$P_θ$。因为前缀激活总是在左侧上下文中，因此会影响其右侧的任何激活。
## 4.3 Parametrization of $P_θ$
* 直接更新$P_θ$参数会导致优化不稳定，性能略有下降。作者将矩阵$P_θ [i，:] = MLP_θ(P ' θ[i，:])$用一个更小的矩阵($P_θ '$)和一个大前馈神经网络($MLP_θ$)组成的矩阵($P_θ '$)重新参数化。$P_θ$和$P_θ '$有相同的行维度不同的列维度
# 5 Experimental Setup
## 5.1 Datasets and Metrics
* 评估了三个数据集：E2E，WebNLG，DART
	* E2E只有1个域(即餐厅评论);WebNLG有14个域，而DART是开放域使用维基百科的开放域表。
	* E2E数据集包含大约50K个具有8个不同字段的示例;它包含一个源表的多个测试引用，平均输出长度为22.9。
	* WebNLG 数据集由22K个示例组成，输入x是(主题，属性，对象)三元组的序列。平均输出长度为22.5。在训练和验证部分，输入描述了来自9个不同DBpedia类别的实体(例如，Monument)。测试分割由两部分组成:前半部分包含训练数据中看到的DB类别，后半部分包含5个未见的类别。这些看不见的类别用于评估外推。
	* DART是一个开放域表到文本数据集，具有与WebNLG相似的输入格式(实体-关系-实体三元组)。平均输出长度为21.6。它由来自WikiSQL、WikiTableQuestions、E2E和WebNLG的82K个示例组成，并应用了一些手动或自动转换
	* 对于摘要任务，使用XSUM 数据集，有225K个例子。文章的平均长度为431字，摘要的平均长度为23.3字。
## 5.2 Methods
* 比较了三种方法：微调(FINE-TUNE)、仅对最顶层2层进行微调(FT-TOP2)和适配器调优(ADAPTER)
## 5.3 Architectures and Hyperparameters
* 对于表格到文本，使用$GPT-2_{MEDIUM}$和$GPT2_{LARGE}$,source tables是线性化的,对于总结任务，使用$BART_{LARGE}$,并且source articles被截断为512个BPE tokens。
* 在训练时：
	* 使用AdamW作为优化器，使用线性学习率调度
	* hyperparameters：epochs，batch size,learning rate, and prefix length
	* 默认设置训练10个epoch，batch size大小为5，learning rate为5·10−5， prefix length为10.
* 在解码时：
	* 使用波束搜索，波束大小为5。
	* 对于摘要任务，使用6的波束大小，长度归一化0.8。
	* 从表到文本的解码每句话需要1.2秒(不进行批处理)，摘要每批需要2.6秒(使用批处理大小为10)
# 6 Main Results
## 6.1 Table-to-text Generation
* 作者发现仅增加0.1%的特定任务的参数，是非常有效的对于表格到文本的生成，能够实现一个非常好的效果。
* 对prefix-tuning 和 adapter-tuning进行比较，prefix-tuning显然比adapter-tuning效果更好,这表明prefix-tuning比adapter-tuning更有效，在提高生成质量的同时显著减少了参数.prefix-tuning可以推广到具有不同域和大量关系的表。
* prefix-tuning是一种有效且空间高效的方法，使GPT-2适应表到文本生成.
## 6.2 Summarization
* XSUM 和三个表到文本数据集之间有几个差异，这可以解释为什么prefix-tuning在表格到文本方面具有比较优势：
	* (1) XSUM 平均包含比三个表到文本数据集多 4 倍的示例；
	* (2) 输入文章平均比表到文本数据集的线性化表输入长 17 倍；
	* (3) 摘要可能比表到文本更复杂，因为它需要阅读理解并从文章中识别关键内容。
## 6.3 Low-data Setting
* 当训练示例数量较少时，prefix-tuning具有比较优势,为了构建低数据设置，作者对完整数据集（表到文本的 E2E 和用于摘要的 XSUM）进行子采样，以获得大小为 {50, 100, 200, 500} 的小数据集。对于每个大小，我们对 5 个不同的数据集进行采样，平均超过 2 个训练随机种子。因此，我们平均超过 10 个模型以获得每个低数据设置的估计.
* prefix-tuning在低数据状态下平均比fine-tuning高出 2.9 BLEU
## 6.4 Extrapolation
* WebNLG:标签是table topics,有9个类别出现在train数据集和dev数据集中，表示为SEEN, 5个类别只出现在测试时，表示为UNSEEN。
# 7 Intrinsic Evaluation
* 作者比较了prefix-tuning的不同变体。
	* 7.1节研究前缀长度的影响。
	* 7.2节研究只调优嵌入层，这更类似于调优离散提示符
	* 7.3节比较prefixing和infixing，在$x$和$y$之间插入可训练的激活
	* 7.4节研究了各种前缀初始化策略的影响。
## 7.1 Prefix Length
* 当前缀长度增加到一个阈值(摘要为200，表到文本为10)时，性能会增加，然后出现轻微的性能下降,较长的前缀对推理速度的影响可以忽略不计，因为整个前缀的注意力计算在gpu上是并行的。
## 7.2 Full vs Embedding-only
* embedding-only消融:优化“虚拟令牌”的连续嵌入的选项。
* 词嵌入是自由参数，上层激活层由Transformer计算。性能显著下降，表明仅调优嵌入层的表现力不够。
* 离散的prompt限制嵌入层精确匹配真实单词的嵌入\
* 作者提出了不断增强的表达能力链:离散的prompt<embedding-only消融<prefix-tuning。
## 7.3 Prefixing vs Infixing
* 作者研究了可训练激活在序列中的位置如何影响性能
* Infixing调优的性能略低于Prefixing调优,因为前缀调优可以影响x和y的激活，而中缀调优只能影响y的激活。
## 7.4 Initialization
* 作者发现前缀的初始化方式在低数据设置中有很大的影响,随机初始化导致高方差的低性能.用真实单词的激活初始化前缀可以显著提高生成,用任务相关的词(如“summarization”和“table-to-text”)初始化的性能略好于任务无关的词(如“elephant”和“divide”)，但使用真实的词仍然比随机的好。
# 8 Discussion
## 8.1 Personalization
* 为了保护用户隐私，需要对每个用户的数据进行分离，并为每个用户独立训练个性化模型。因此，每个用户都可以看作是一个独立的任务。如果有数百万用户，前缀调优可以扩展到此设置并维护模块化，通过添加或删除用户的前缀来实现灵活的添加或删除用户，而不会交叉污染.
## 8.2 Batching Across Users
* 前缀调优允许批量处理不同用户的查询，即使它们由不同的前缀支持。
* 在相同的个性化设置下，前缀调优允许对不同用户的查询进行批处理，即使它们有不同的前缀支持。当多个用户使用他们的输入查询云GPU设备时，将这些用户放在同一个批处理中可以提高计算效率。前缀调优使共享LM保持完整;因此，批处理只需要一个简单的步骤，即在用户输入前添加个性化前缀，而所有剩余的计算都保持不变。相反，在适配器调优中，我们不能跨不同的用户进行批处理，它在共享Transformer层之间具有个性化的适配器。
## 8.3 Inductive Bias of Prefix-tuning
* 保留LM参数可能有助于泛化到训练过程中未见过的领域.
* 前缀调优和适配器调优都会冻结预训练的参数,它们调优不同的参数集来影响Transformer的激活层.
* 适配器调优在LM层之间插入可训练模块，直接向激活中添加残差向量.
* 与适配器调优相比，前置调优需要的参数要少得多，同时保持相当的性能
# 9 Conclusion
* 作者提出了前缀调优，这是一种轻量级的替代调优方法，可以为NLG任务添加可训练的连续前缀。我们发现，尽管学习的参数比微调少1000倍，前缀调优可以在完整的数据设置中保持相当的性能，并且在低数据和外推设置中都优于微调。