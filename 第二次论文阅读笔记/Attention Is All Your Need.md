## Abstruct
* author propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
* *Experiments on two machine translation tasks.
* Model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.
* On the WMT 2014 English-to-French translation task, model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs
* Conclusion:Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
## 1、Introduction
* Recurrent models typically factor computation along the symbol positions of the input and output sequences.
* In order to align the positions to steps in computation time,they generate a sequence of hidden states $h_t$ ,the previous hidden state $h_{t-1}$ ,the input for position t , This inherently sequential nature precludes parallelization within training examples . parallelization within training examples becomes critical at longer sequence lengths, as memory constraints limit batching across examples.
* Attention mechanisms allow modeling of dependencies without regard to their distance in the input or output sequences.In all but a few cases , however, such attention mechanisms are used in conjunction with a recurrent network.
* Transformer is a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
## 2、Background
* In Extended Neural GPU, ByteNet  and ConvS2S, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet.
* In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section.
* Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations.
* End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks
## 3、Model Architecture
* the encoder maps an input sequence of symbol representations ($x_1$, ..., $x_n$) to a sequence of continuous representations z = ($z_1$, ..., $z_n$),Given z, the decoder then generates an output sequence ($y_1$, ..., $y_m$) of symbols one element at a time.At each step the model is auto-regressive,consuming the previously generated symbols as additional input when generating the next.
* The Transformer follows this overall architecture using stacked self-attention and point-wise,, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.![Figure1](D:\学习笔记\笔记\论文笔记\image\Attention_figure1.png)
### 3.1、Encoder and Decoder Stacks
#### Encoder:
* The encoder is composed of a stack of $N$ = 6 identical layers,Each layer has two sub-layers,The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network.
* Author employ a residual connection around each of the two sub-layers, followed by layer normalization the output of each sub-layer isLayerNorm($x$ + Sublayer($x$)),Sublayer($x$) is the function implemented by the sub-layer itself
* To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model}$ = 512.
#### Decoder:
* The decoder is also composed of a stack of $N$ = 6 identical layers,In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.
* Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.
* This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$ .
### 3.2、Attention
* An attention function can be described as mapping a $query$ and a set of $key-value$ pairs to an output, where the $query$, $keys$, $values$, and $output$ are all vectors.The output is computed as a weighted sum of the values,the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
#### 3.2.1、Scaled Dot-Product Attention
* ![Figure2](D:\学习笔记\笔记\论文笔记\image\Attention_figure2.png)                                                                  Figure2
* The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$ ,Author compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$ , and apply a softmax function to obtain the weights on the values.
* Author compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$.The keys and values are also packed together into matrices $K$ and $V$ .
* $Attention$($Q, K, V$ ) = $softmax$($\frac{QK^T}{\sqrt{d_k}}$)$V$ 
* The two most commonly used attention functions are additive attention and dot-product (multiplicative) attention.Dot-product attention is identical to author algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$.Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
* While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of ${d_k}$ , Author suspect that for large values of ${d_k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.
#### 3.2.2、Multi-Head Attention
* Instead of performing a single attention function with ${d_{model}}$-dimensional keys, values and queries, author found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to ${d_k}$, ${d_k}$ and ${d_v}$ dimensions, respectively.
	* MultiHead($Q, K, V$ ) = Concat($head_1$, ..., $head_h$)$W^O$
		* where $head_i$=Attention($QW_i^Q$,$KW_i^K$,$VW_i^V$)
	* Where the projections are parameter matrices: $W_i^Q$ ∈ $R^{d_{model}×d_k}$,$W_i^K$ ∈ $R^{d_{model}×d_k}$,$W_i^V$ ∈ $R^{d_{model}×d_v}$and$W^O$ ∈ $R^{hd_v×d_{model}}$.
* For each of these  use $d_k$ = $d_v$ = $d_{model}/h$ = 64
#### 3.2.3 Applications of Attention in our Model
* The Transformer uses multi-head attention in three different ways:
	* In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.This allows every position in the decoder to attend over all positions in the input sequence.
	* The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
	* self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.
### 3.3 Position-wise Feed-Forward Networks
* In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
  FFN$(x)$ = $max(0, xW_1 + b_1)W_2 + b_2$
  While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is $d_{model}$ = 512, and the inner-layer has dimensionality$d_f$ = 2048.
### 3.4 Embeddings and Softmax
* Author use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$, Author also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.In the embedding layers, Author multiply those weights by $\sqrt{d_{mode}}$.
	### 3.5 Positional Encoding
* To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodelas the embeddings, so that the two can be summed.
* In this work, Author use sine and cosine functions of different frequencies:
		$PE(pos,2i) = sin(pos/10000^{\frac{2_i}{d_{model}}})$
		$PE(pos,2i+1) = cos(pos/10000^{\frac{2_i}{d_{model}}})$
	* $pos$ is the position and $i$ is the dimension 
	* The wavelengths form a geometric progression from $2π$ to 10000 · $2π$.
	* since for any fixed offset k, $PE_{pos+k}$ can be represented as a linear function of$PE_{pos}$.
* Author also experimented with using learned positional embeddings  instead, and found that the two versions produced nearly identical results . Author chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training. 
## 4 Why Self-Attention
* Motivating our use of self-attention we consider three desiderata:
	* One is the total computational complexity per layer.
	* Two is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.
	* The third is the path length between long-range dependencies in the network.One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies
* In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$,which is most often the case with sentence representations used by state-of-the-art models in machine translations.
* To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size $r$ in the input sequence centered around the respective output position.
* A single convolutional layer with kernel width $k < n$ does not connect all pairs of input and output positions.Doing so requires a stack of $O(n/k)$ convolutional layers in the case of contiguous kernels, or $O(log_{k}(n))$ in the case of dilated convolutions , increasing the length of the longest paths between any two positions in the network.
* Convolutional layers are generally more expensive than recurrent layers, by a factor of $k$.Separable convolutions decrease the complexity considerably to $O(k · n · d + n · d^2)$.
* Even with $k = n$, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer,
* self-attention could yield more interpretable models.
## 5 Training
### 5.1 Training Data and Batching
* Training Data: 
	* For English-German WMT 2014 English-German dataset
	* For English-French larger WMT 2014 English-French dataset
* Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.
### 5.2 Hardware and Schedule
* The big models were trained for 300,000 steps.
### 5.3 Optimizer
* Author used the Adam optimizer  with $β1$ = 0.9, $β2$ = 0.98 and    $\epsilon$ = 10−9.
$lrate$ = $d^{−0.5}_{model}$ · $min(step\_num^{−0.5}, step\_num · warmup\_steps^{−1.5})$.
* This corresponds to increasing the learning rate linearly for the first $warmup\_steps$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used $warmup\_step$s = 4000.
### 5.4 Regularization
* three types of regularization
	* Residual Dropout : apply dropout  to the output of each sub-layer before it is added to the sub-layer input and normalized.apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model,  use a rate of $P_{drop}$ = 0.1.
	* Label Smoothing : employed label smoothing of value $\epsilon_{ls}$ = 0.1 This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
##  6 Results
### 6.1 Machine Translation
* On the WMT 2014 English-to-German translation task, the big transformer model  outperforms the best previously reported models  by more than 2.0BLEU, establishing a new state-of-the-art BLEU score of 28.4.
* On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model.dropout rate $P_{drop}$ = 0.1, instead of 0.3.
* We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU.
### 6.2 Model Variations
* While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.
* Author observe that reducing the attention key size $d_k$ hurts model quality.
* determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial.
* bigger models are better, and dropout is very helpful in avoiding over-fitting.
* Author replace our sinusoidal positional encoding with learned positional embeddings , and observe nearly identical results to the base model.
### 6.3 English Constituency Parsing
* This task presents specific challenges: the output is subject to strong structural  constraints and is significantly longer than the input.RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes.
* despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar.
* In contrast to RNN sequence-to-sequence models , the Transformer outperforms the BerkeleyParser  even when training only on the WSJ training set of 40K sentences.
## 7 Conclusion
* In this work, Auhtor presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.
* For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.
* Author plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.