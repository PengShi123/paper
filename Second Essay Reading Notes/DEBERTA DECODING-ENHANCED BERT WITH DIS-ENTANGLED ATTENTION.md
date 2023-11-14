
# Abstract:
* In this paper  propose a new model architecture DeBERTa (Decoding-enhanced BERT with disentangled attention) that improves the BERT and RoBERTa models using two novel techniques.
* disentangled attention mechanism:each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions, respectively.
* an enhanced mask decoder is used to incorporate absolute positions in the decoding layer to predict the masked tokens in model pre-training.
* these techniques significantly improve the efficiency of model pre-training and the performance of both natural language understand (NLU) and natural langauge generation (NLG) downstream tasks.
* The pre-trained DeBERTa models and the source code were released at:https://github.com/microsoft/DeBERTa1.
# 1 INTRODUCTION
* Author propose a new Transformer-based neural language model DeBERTa (Decoding-enhanced BERT with disentangled attention), which improves previous state-of-the-art PLMs using two novel techniques: a disentangled attention mechanism, and an enhanced mask decoder.
* Disentangled attention:each word in DeBERTa is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices based on their contents and relative positions.This is motivated by the observation that the attention weight of a word pair depends on not only their contents but their relative positions.
* Enhanced mask decoder:DeBERTa uses the content and position information of the context words for MLM.
* These syntactical nuances depend, to a large degree, upon the words' absolute positions in the sentence, and so it is important to account for a word's absolute position in the language modeling process.
* DeBERTa incorporates absolute word position embeddings right before the softmax layer where the model decodes the masked words based on the aggregated contextual embeddings of word contents and positions.
* In the NLU tasks, compared to RoBERTa-Large, a DeBERTa model trained on half the training data performs consistently better on a wide range of NLP tasks, achieving improvements on MNLI by +0.9% (90.2% vs. 91.1%), on SQuAD v2.0 by +2.3%(88.4% vs. 90.7%), and RACE by +3.6% (83.2% vs. 86.8%).
* In the NLG tasks, DeBERTa reduces the perplexity from 21.6 to 19.5 on the Wikitext-103 dataset.
* The single 1.5B-parameter DeBERTa model substantially outperforms T5 with 11 billion parameters on the SuperGLUE benchmark.
# 2 BACKGROUND
## 2.1 TRANSFORMER
* Each block contains a multi-head self-attention layer followed by a fully connected positional feed-forward network.The standard self-attention mechanism lacks a natural way to encode word position information.existing approaches add a positional bias to each input word embedding so that each input word is represented by a vector whose value depends on its content and position.relative position representations are more effective for natural language understanding and generation tasks.
* The proposed disentangled attention mechanism differs from all existing approaches in that  represent each input word using two separate vectors that encode a word's content and position, respectively, and attention weights among words are computed using disentangled matrices on their contents and relative positions, respectively.
## 2.2 MASKED LANGUAGE MODEL
* given a sequence $X$ = {$x_i$}.$\tilde{X}$ by masking 15% of its tokens at random and then train a language model parameterized by $θ$ to reconstruct $X$ by predicting the masked tokens $\tilde{x}$ conditioned on  $\tilde{X}$:
* $max$ $log$ $p_θ$$(X|\tilde{X})$ = $max$ $\sum$ $log$ $p_θ$($\tilde{x_i}$ = $x_i$|$\tilde{X}$)
* $C$ is the index set of the masked tokens in the sequence
# 3 THE DEBERTA ARCHITECTURE
## 3.1 DISENTANGLED ATTENTION: 
* A TWO-VECTOR APPROACH TO CONTENT AND POSITIONEMBEDDING
* For a token at position $i$ in a sequence, we represent it using two vectors, {$H_i$} and {$P_{i|j}$}.
* The calculation of the cross attention score between tokens i and j can be decomposed into four components as: $A_{i,j}$={$H_i$,$P_{i|j}$} x $\{H_i,P_{i|j}\}^\mathrm T$=$H_iH_j^\mathrm T$+$H_iP_{j|i}^\mathrm T$+$P_{i|j}H_j^\mathrm T$+$P_{i|j}P_{i|j}^\mathrm T$
* the attention weight of a word pair can be computed as a sum of four attention scores using disentangled matrices on their contents and positions as $content-to-content$, $content-to-position$,$position-to-content$, and $position-to-position$.
* the attention weight of a word pair depends not only on their contents but on their relative positions, which can only be fully modeled using both the content-to-position and position-to-content terms.
* $Q = HW_q$ , $K =HW_k$, $V=HW_v$ , $A =\frac{QKᵀ}{\sqrt{d}}$ $H_o = softmax(A)V$
* $H\in{R^{Nxd}}$ represents the input hidden vectors, $H_o\in{R^{Nxd}}$the output of self-attention,$W_q , W_k, W_v \in {R^{dxd}}$ the projection matrices, $A\in {R^{NxN}}$ the attention matrix, $N$ the length of the input sequence, and d the dimension of hidden states.
* Denote $k$ as the maximum relative distance, $δ(i,j)\in{[0,2k)}$, as the relative distance from token $i$ to token $j$, which is defined as:
$δ(i,j)$ = $\begin{cases}{0 ,for,i-j\leq{-k}}\\{2k-1,for,i-j\geq{k}}\\{i-j+k,others}\end{cases}$
### 3.1.1 EFFICIENT IMPLEMENTATION
* For an input sequence of length N , it requires a space complexity of $O(N^2d)$  to store the relative position embedding for each token.
## 3.2 ENHANCED MASK DECODER ACCOUNTS FOR ABSOLUTE WORD POSITIONS
* DeBERTa is pretrained using MLM, where a model is trained to use the words surrounding a mask token to predict what the masked word should be.
* There are two methods of incorporating absolute positions:
	* The BERT model incorporates absolute positions in the input layer. In DeBERTa, we incorporate them right after all the Transformer layers but before the softmax layer for masked token prediction.
	* Author compare these two methods of incorporating absolute positions and observe that EMD works much better
# 4 SCALE INVARIANT FINE-TUNING
* Scale-invariant-Fine-Tuning (SiFT):a variant to the algorithm described in [Miyato et al. (2018); Jiang et al. (2020)], for fine-tuning.
* Virtual adversarial training is a regularization method for improving models' generalization.It does so by improving a model's robustness to adversarial examples, which are created by making small perturbations to the input.
* the SiFT algorithm that improves the training stability by applying the perturbations to the normalized word embeddings. 
* when fine-tuning DeBERTa to a downstream NLP task in our experiments, SiFT first normalizes the word embedding vectors into stochastic vectors, and then applies the perturbation to the normalized embedding vectors.normalization substantially improves the performance of the fine-tuned models.
# 5 EXPERIMENT
## 5.1 MAIN RESULTS ON NLU TASKS
### 5.1.1 PERFORMANCE ON LARGE MODELS
* Author use Wikipedia (12GB), BookCorpus (6GB), OPENWEBTEXT (38GB), and STORIES (a subset of CommonCrawl (31GB). The total data size after data deduplication is about 78G.
* DeBERTa is pre-trained for one million steps with 2K samples in each step. compared to BERT and RoBERTa, DeBERTa performs consistently better across all the tasks.
* Among all GLUE tasks, MNLI is most often used as an indicative task to monitor the research progress of PLMs. DeBERTa significantly outperforms all existing PLMs of similar size on MNLI and creates a new state of the art.
* DeBERTa is evaluated on three categories of NLU benchmarks: (1) Question Answering: SQuAD v1.1 , SQuAD v2.0 , RACE , ReCoRD  and SWAG (; (2) Natural Language Inference: MNLI ; and (3) NER: CoNLL-2003. 
### 5.1.2 PERFORMANCE ON BASE MODELS
* The base model structure follows that of the BERT base model, $L$ =12, $H$ = 768, $A$ = 12
* DeBERTa consistently outperforms RoBERTa and XLNet by a larger margin than that in large models.
## 5.2 MODEL ANALYSIS
### 5.2.1 ABLATION STUDY
* To investigate the relative contributions of different components in DeBERTa, author develop three variations:
	* -EMD is the DeBERTa base model without EMD.
	* -C2P is the DeBERTa base model without the content-to-position term 
	* P2C is the DeBERTa base model without the position-to-content term . As XLNet also uses the relative position bias, this model is close to XLNet plus EMD.
## 5.3 SCALE UP TO 1.5 BILLION PARAMETERS
* Larger pre-trained models have shown better generalization result.
* $DeBERTa_{1.5B}$ is trained on a pre-training dataset amounting to 160G
# 6 CONCLUSIONS
* The first is the disentangled attention mechanism，an enhanced mask decoder.
* a new virtual adversarial training method is used for fine-tuning to improve model's generalization on downstream tasks.
* The DeBERTa model with 1.5 billion parameters surpasses the human performance on the SuperGLUE benchmark for the first time in terms of macro-average score.
* Moving forward, it is worth exploring how to make DeBERTa incorporate compositional structures in a more explicit manner, which could allow combining neural and symbolic computation of natural language similar to what humans do.
