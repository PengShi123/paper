# Abstract
* Prompt tuning，它只对固定语言模型的连续提示进行调优，大大减少了训练时每个任务的存储和内存使用。
* 现有的快速调整方法不能处理困难的序列标记任务，表明缺乏通用性。我们提出了一个新的实证发现，适当优化的Prompt Tuning可以在广泛的模型尺度和NLU任务中普遍有效。
* P-Tuning v2是Deep Prompt Tuning的实现优化和适应NLU。
# 1 Introduction
* 预训练语言模型提高在广泛的自然语言理解(NLU)任务中的表现。一种广泛使用的方法是微调，它更新目标任务的整个模型参数集。虽然微调获得了良好的性能，但在训练期间会消耗内存，因为必须存储所有参数的梯度和优化器状态。此外，在推理期间为每个任务保留模型参数的副本是不方便的，因为预训练的模型通常很大。
## Prompting
* Prompting根本不需要训练，并且存储模型参数的单一副本。
* 离散Prompting与微调相比，在许多情况下会导致次优性能。
## Prompt tuning（an idea of tuning only the continuous prompts）
* 在训练期间，只有连续的提示才会更新。虽然在许多任务上，prompt tuning优于Prompt。当模型规模不大时，特别是小于100亿个参数时，它的微调性能仍然较差
* 与几个硬序列标记任务的微调相比，提示调优的性能较差比如提取问答答案。
* 我们在本文中的主要贡献是一个新的实证发现，即适当优化的提示调整可以与各种模型尺度和NLU任务的普遍微调相媲美。与先前工作的观察结果相反，作者的发现揭示了NLU快速调整的普遍性和潜力。
* Deep prompt tuning增加了continuous prompts的能力，并缩小了跨各种设置fine-tuning的差距，特别是对于小型模型和艰巨的任务，它可以被视为为生成和知识探测而设计的深度提示调优的优化和改编实现。最显著的改进源于对预训练模型的每一层应用连续提示，而不仅仅是输入层。
* 与fine-tuning相比，P-tuning v2每个任务有0.1%到3%的可训练参数，这大大降低了训练时间、内存成本和每个任务的存储成本。
# 2 Preliminaries
## NLU Tasks：simple classification tasksand hard sequence labeling tasks
* Simple classification tasks involve classification over a label space
* Hard sequence labeling tasks involve classification over a sequence of tokens, Most datasets from GLUE [(Wang et al., 2018)] and SuperGLUE [(Wang et al., 2019) ]are in this categorysuch as named entity recognition and extractive question answering.
## Prompt Tuning
* Let $V$ be the vocabulary of a language model $M$ and let $e$ be the embedding layer of $M$.
* prompt tokens {"$It$", "$is$", "$[MASK]$"}$⊂ V$ can be used to classify a movie review.给定输入文本$x$ =“$Amazing movie!$”，输入嵌入序列公式化为  [e(x)， e(“It”)，e(“is”)，e(“[MASK]”)]。给定可训练连续嵌入$[h_0，…,h_i]$， hi]，则输入嵌入序列写成\[e(x)， $h_0，…,h_i$ , e("[MASK]")\]，
# 3 P-Tuning v2
## 3.1 Lack of Universality
* Lack of universality across scales:当模型扩展到超过100亿个参数时，这种prompt tuning可以与fine-tuning相媲美。然而，对于广泛使用的中型型号(从100M到1B)，prompt tuning的性能远不如fine-tuning。
* Lack of universality across tasks:在一些NLU基准测试中，prompt tuning在硬序列标记任务上的有效性尚未得到验证,序列标记预测每个输入标记的标签序列，这可能比较困难，并且与语言器不兼容,与fine-tuning相比，在典型的序列标记任务上执行得很差。
## 3.2 Deep Prompt Tuning
* 只将continuous prompts插入到输入的编码序列中，这会导致两种挑战：
	* 由于序列长度的限制，可调参数的数量有限
	* 输入嵌入对模型预测有相对间接的影响。、
*  不同层中的提示被添加为prefix tokens，P-tuning v2具有更多可调的特定于任务的参数(从0.01%到0.1%-3%)，以便使参数高效的同时实现每个任务的能力。添加到更深层的提示对模型预测有更直接的影响。
## 3.3 Optimization and Implementation
* Reparameterization：
	* NLU的有用性取决于任务和数据集，对于某些数据集(例如，RTE和CoNLL04)， MLP带来了一致的改进;对于其他方面来说，MLP对结果的影响很小，甚至是负面的
* Prompt Length：
	* 提示符长度在P-Tuning v2中起着关键作用，不同的NLU任务通常在不同的提示长度下达到最佳性能，简单分类任务喜欢较短的提示(少于20个);复杂的序列标记任务倾向于较长的序列(大约100个)
* Multi-task Learning：
	* 多任务学习在对单个任务进行微调之前，通过共享的连续提示共同优化多个任务。对于P-Tuning v2来说，多任务是可选的，但是可以通过提供更好的初始化来进一步提高性能
* Classification Head：
	* P-tuning v2在标记上应用随机初始化的分类头。
# 4 Experiments
* 在不同的常用预训练模型和NLU任务上进行了广泛的实验，以验证P-tuning v2的有效性。在本工作中，除了微调之外的所有方法都是在固定的语言模型主干下进行的，这符合的设置，但不同于的调整设置。特定任务参数的比率(例如，0.1%)是通过比较连续提示“参数”与变压器“参数”得出的。另一件需要注意的事情是，实验都是在完全监督的环境下进行的，而不是在少数情况下进行的。
* NLU Tasks.:
	* SuperGLUE的数据集来测试P-tuning v2的NLU能力.
	* sequence labeling tasks:
		* name dentity recognition
		* extractive Question Answering
		* semantic role labeling
* Pre-trained Models:
	* BERT-large
	* RoBERa-large
	* DeBERTa-xlarge
	* GLMxlarge/xxlarge
* Multitask Learning:
	* 对于多任务设置，将每个任务类型的数据集的训练集进行组合(例如，将语义角色标注的所有训练集进行组合)。对每个数据集使用单独的线性分类器，同时共享连续的提示。
## 4.1 P-tuning v2: Across Scales
* 在SuperGLUE中，[Lester等人(2021)]和P-tuning在较小尺度上的性能可能相当差,P-tuning v2在较小的范围内匹配所有任务的fine-tuning性能。P-tuning v2甚至明显优于RTE上的fine-tuning
* 按10B的规模，prompt-tuning与fine-tuning相比更有竞争力，P-tuning v2总是与所有尺度的fine-tuning相媲美，但与fine-tuning相比，只需要0.1%的任务特定参数。
## 4.2 P-tuning v2: Across Tasks
* P-tuning v2 通常可以与所有任务的调优相媲美,Ptuning和Lester等人(2021)表现得更差，特别是在QA上，这可能是三个任务中最具挑战性的.除了QA之外，多任务学习通常会给P-Tuning v2带来显著的改进。]
## 4.3 Ablation Study
* Verbalizer with LM head v.s. [CLS] label with linear head:
	* 对于在监督设置下的P-tuning v2，用大约几千个参数调优线性磁头是可以承受的。我们保留其他超参数，仅将线性头的[CLS]标签更改为LM头的语言器
	* no significant difference between performances of verbalizer and [CLS].
* Prompt depth:
	* [Lester et al. (2021);(Liu et al.， 2021)]和P-tuning v2最主要的区别是多层连续提示。为了验证其确切的影响，给定一定数量$k$层来添加提示，按升序和降序选择它们来添加提示;对于其余的层，我们保持不变。对于相同数量的参数(即要添加提示符的变压器层数)，按降序添加总是比按升序添加好。在RTE的情况下，仅向17-24层添加提示可以产生与所有层非常接近的性能。
# 5 Conclusions
* 本文提出了一种快速调优方法P-tuning v2。尽管它的技术新颖性相对有限，但它有助于一个新的发现，即快速调整可以与跨尺度(从330M到10B参数)和任务的普遍微调相比较。由于具有高精度和参数效率，PTuning v2可以成为调优的潜在替代方案和未来工作的强大基线。