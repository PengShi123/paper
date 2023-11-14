# ABSTRACT
* An important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains.
* Author propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.
* Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.
* https://github.com/microsoft/LoRA.
# 1 INTRODUCTION
* As larger models are trained every few months, this changes from a mere "inconvenience" for GPT-2 or RoBERTa large  to a critical deployment challenge for GPT-3  with 175 billion trainable parameters.
* Many sought to mitigate this by adapting only some parameters or learning external modules for new tasks.This way, we only need to store and load a small number of task-specific parameters in addition to the pre-trained model for each task, greatly boosting the operational efficiency when deployed.However, existing techniques often introduce inference latency by extending model depth or reduce the model's usable sequence length
* the learned over-parametrized models in fact reside on a low intrinsic dimension.
* LoRA allows us to train some dense layers in a neural network indirectly by optimizing rank decomposition matrices of the dense layers' change during adaptation instead, while keeping the pre-trained weights frozen.
* Using GPT-3 175B as an example, we show that a very low rank  suffices even when the full rank  is as high as 12,288, making LoRA both storage- and compute-efficient.
* LoRA possesses several key advantages:
	* A pre-trained model can be shared and used to build many small LoRA modules for different tasks. We can freeze the shared model and efficiently switch tasks by replacing the matrices A and B in Figure 1, reducing the storage requirement and task-switching overhead significantly.
	* LoRA makes training more efficient and lowers the hardware barrier to entry by up to 3 times when using adaptive optimizers since we do not need to calculate the gradients or maintain the optimizer states for most parameters. Instead, we only optimize the injected, much smaller low-rank matrices.
	* Our simple linear design allows us to merge the trainable matrices with the frozen weights when deployed, introducing no inference latency compared to a fully fine-tuned model, by construction.
	* LoRA is orthogonal to many prior methods and can be combined with many of them, such as prefix-tuning.
	* Terminologies and Conventions:
		* the input and output dimension size of a Transformer layer $d_{model}$.
		* use $W_q$ , $W_k$, $W_v$ , and $W_o$ to refer to the query/key/value/output projection matrices in the self-attention module.
		* $W$ or $W0$ refers to a pretrained weight matrix and $∆W$ its accumulated gradient update during adaptation.
		* use $r$ to denote the rank of a LoRA module. 
		* use Adam for model optimization 
		* use a Transformer MLP feedforward dimension $d_{f f n}$ = 4 × dmodel.
# 2 PROBLEM STATEMENT
* a training dataset of context-target pairs: $Z$ = $\{{(x_i, y_i)}\}_i=1,..,N$,where both $x_i$ and ${y_i}$ are sequences of tokens.
* During full fine-tuning, the model is initialized to pre-trained weights $Φ_0$ and updated to $Φ_0$ + $∆_Φ$ by repeatedly following the gradient to maximize the conditional language modeling objective:
* $maxΦ∑(x,y)∈Z |y|∑t=1log (P_Φ(y_t|x, y<t))$
* One of the main drawbacks for full fine-tuning is that for each downstream task, we learn a differentset of parameters $∆_Φ$ whose dimension$|∆_Φ|$ equals $|Φ_0|$. Thus, if the pre-trained model is large (such as GPT-3 with$|Φ_0|$ ≈ 175 Billion), storing and deploying many independent instances of fine-tuned models can be challenging, if at all feasible.
# 3 AREN'T EXISTING SOLUTIONS GOOD ENOUGH?
* there are two prominent strategies when it comes to efficient adaptations: adding adapter layers or optimizing some forms of the input layer activations.
* Adapter Layers Introduce Inference Latency:
	* two adapter layers per Transformer block and a more recent one by [Lin et al. (2020)] which has only one per block but with an additional LayerNorm
	* adapter layers are designed to have few parameters (sometimes <1% of the original model) by having a small bottleneck dimension, which limits the FLOPs they can add.
	* because the additional depth requires more synchronous GPU operations such asAllReduce and Broadcast, unless we store the adapter parameters redundantly many times.
* Directly Optimizing the Prompt is Hard:
	* prefix tuning is difficult to optimize and that its performance changes non-monotonically in trainable parameters, confirming similar observations in the original paper.
	* More fundamentally, reserving a part of the sequence length for adaptation necessarily reduces the sequence length available to process a downstream task
# 4 OUR METHOD
## 4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES
* A neural network contains many dense layers which perform matrix multiplication.The weight matrices in these layers typically have full-rank.
* the pre-trained language models have a low "instrisic dimension" and can still learn efficiently despite a random projection to a smaller subspace.
* For a pre-trained weight matrix$W_0$ ∈$R^{d×k}$, we constrain its update by representing the latter with a low-rank decomposition $W_0$ + $∆W$ = $W_0$ + $BA$, where B ∈ $R^{d×r}$ , A ∈ $R^{r×k}$, and the rank $r\ll{min(d, k)}$.During training, W0 is frozen and does not receive gradient updates, while A and B contain trainable parameters
* For h = W0x, our modified forward pass yields:
	* $h = W_0x + ∆Wx = W_0x + BAx$
* A Generalization of Full Fine-tuning:
	* A more general form of fine-tuning allows the training of a subset of the pre-trained parameters.
	* LoRA takes a step further and does not require the accumulated gradient update to weight matrices to have full-rank during adaptation.
	* adapter-based methods converges to an MLP and prefix-based methods to a model that cannot take long input sequences.
* No Additional Inference Latency:
	* When deployed in production, we can explicitly compute and store W = W0 + BA and perform inference as usual. Note that both W0 and BA are in Rd×k. When we need to switch to another downstream task, we can recover W0 by subtracting BA and then adding a different B′A′, a quick operation with very little memory overhead.
## 4.2 APPLYING LORA TO TRANSFORMER
* In the Transformer architecture, there are four weight matrices in the self-attention module ($W_q$ , $W_k$, $W_v$ , $W_o$) and two in the MLP module.
* Practical Benefits and Limitations:
	* The most significant benefit comes from the reduction in memory and storage usage.
	* For a large Transformer trained with Adam, we reduce that VRAM usage by up to 2/3 if$r\ll{d_{model}}$ as we do not need to store the optimizer states for the frozen parameters.
	* Another benefit is that we can switch between tasks while deployed at a much lower cost by only swapping the LoRA weights as opposed to all the parameters.
	* LoRA also has its limitations:it is not straightforward to batch inputs to different tasks with different A and B in a single forward pass, if one chooses to absorb A and B into W to eliminate additional inference latency.
# 5 EMPIRICAL EXPERIMENTS
## 5.1 BASELINES
* Fine-Tuning (FT) is a common approach for adaptation. During fine-tuning, the model is initialized to the pre-trained weights and biases, and all model parameters undergo gradient updates.
* Bias-only or BitFit is a baseline where we only train the bias vectors while freezing everything else.
* Prefix-embedding tuning (PreEmbed) inserts special tokens among the input tokens.
* Prefix-layer tuning (PreLayer) is an extension to prefix-embedding tuning.
* Adapter tuning as proposed in [Houlsby et al. (2019)]inserts adapter layers between the selfattention module (and the MLP module) and the subsequent residual connection.
* LoRA adds trainable pairs of rank decomposition matrices in parallel to existing weight matrices.
## 5.2 ROBERTA BASE/LARGE
* First, we use the same batch size for all tasks and use a sequence length of 128 to match the adapter baselines. Second, we initialize the model to the pre-trained model for MRPC, RTE, and STS-B, not a model already adapted to MNLI like the fine-tuning baseline. Runs following this more restricted setup from[ Houlsby et al. (2019]) are labeled with †.
## 5.3 DEBERTA XXL
* evaluate if LoRA can still match the performance of a fully fine-tuned DeBERTa XXL (1.5B) on GLUE.
## 5.4 GPT-2 MEDIUM/LARGE
## 5.5 SCALING UP TO GPT-3 175B
* LoRA matches or exceeds the fine-tuning baseline on all three datasets.
* Author observe a significant performance drop when we use more than 256 special tokens for prefix-embedding tuning or more than 32 special tokens for prefix-layer tuning.
# 6 RELATED WORKS
* Transformer Language Models:Transformer is a sequence-to-sequence architecture that makes heavy use of self-attention
* A new paradigm emerged with BERT and GPT-2  – both are large Transformer lan8 guage models trained on a large amount of text – where fine-tuning on task-specific data after pretraining on general domain data provides a significant performance gain compared to training on task-specific data directly.
* Prompt Engineering and Fine-Tuning:
	* While GPT-3 175B can adapt its behavior with just a few additional training examples, the result depends heavily on the input prompt.
	* Fine-tuning retrains a model pre-trained on general domains to a specific task.
* Parameter-Efficient Adaptation:
	* uses a similar bottleneck structure to impose a low-rank constraint on the weight updates.
	* A comtenporary extension of adapter is COMPACTER,which essentially parametrizes the adapter layers using Kronecker products with some predetermined weight sharing scheme.
	* combining LoRA with other tensor product-based methods could potentially improve its parameter efficiency.
* Low-Rank Structures in Deep Learning:
	* Another theoretical result in Allen-Zhu & Li (2020b) suggests that low-rank adaptations can be useful for adversarial training. In sum, author believe that our proposed low-rank adaptation update is well-motivated by the literature.
#  7 UNDERSTANDING THE LOW-RANK UPDATES
* the low-rank structure not only lowers the hardware barrier to entry which allows us to run multiple experiments in parallel, but also gives better interpretability of how the update weights are correlated with the pre-trained weights.
* Author perform a sequence of empirical studies to answer the following questions:
	* 1) Given a parameter budget constraint, which subset of weight matrices in a pre-trained Transformer should we adapt 9 to maximize downstream performance?
	* 2) Is the "optimal" adaptation matrix ∆W really rankdeficient? If so, what is a good rank to use in practice?
	* 3) What is the connection between ∆W andW ? Does ∆W highly correlate with W ? How large is ∆W comparing to W ?
## 7.1 WHICH WEIGHT MATRICES IN TRANSFORMER SHOULD WE APPLY LORA TO?
* only consider weight matrices in the self-attention module.
## 7.2 WHAT IS THE OPTIMAL RANK r FOR LORA?
LoRA already performs competitively with a very small $r$ (more so for$\{W_q , W_v \}$ than just $W_q$ ).
increasing r does not cover a more meaningful subspace, which suggests that a low-rank adaptation matrix is sufficient.
# 8 CONCLUSION AND FUTURE WORK
* Author propose LoRA, an efficient adaptation strategy that neither introduces inference latency nor reduces input sequence length while retaining high model quality. Importantly, it allows for quick task-switching when deployed as a service by sharing the vast majority of the model parameters.Importantly, it allows for quick task-switching when deployed as a service by sharing the vast majority of the model parameters.
* There are many directions for future works:
	* 1) LoRA can be combined with other efficient adaptation methods, potentially providing orthogonal improvement.
	* 2) The mechanism behind fine-tuning or LoRA is far from clear – how are features learned during pre-training transformed to do well on downstream tasks? We believe that LoRA makes it more tractable to answer this than full fine tuning. 
	* 3)We mostly depend on heuristics to select the weight matrices to apply LoRA to. Are there more principled ways to do it?
	* 4)Finally, the rank-deficiency of ∆W suggests that W could be rank-deficient as well, which can also be a source of inspiration for future works.