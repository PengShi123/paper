# Abstract
* BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.
* the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks.
# 1、Introduction
* There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based （such as ELMo）and fine-tuning（such as the Generative Pre-trained Transformer）
* current techniques restrict the power of the pre-trained representations，The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training。
* Conclusion：it is crucial to incorporate context from both directions.
* BERT alleviates the previously mentioned unidirectionality constraint by using a "masked language model" (MLM) pre-training objective。
* 将输入的tokens随机掩盖，在利用上下文获得被掩盖token的id。
* the MLM enables the representation to fuse the left and the right context.
* pre-trained representations reduce the need for many heavily-engineered taskspecific architectures.
# 2、Related Work
## 2.1 Unsupervised Feature-based Approaches
* learning contextual representations through a task to predict a single word from both left and right context using LSTMs.
* the cloze task can be used to improve the robustness of text generation models.
## 2.2 Unsupervised Fine-tuning Approaches
* As with the feature-based approaches, the first works in this direction only pre-trained word embedding parameters from unlabeled text
## 2.3 Transfer Learning from Supervised Data
# 3 BERT
* During pre-training, the model is trained on unlabeled data over different pre-training tasks
* For finetuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks
* A distinctive feature of BERT is its unified architecture across different tasks
### Model Architecture
* BERT's model architecture is a multi-layer bidirectional Transformer encoder
* 《The Annotated Transformer》
* the number of layers as *L*，the hidden size as *H*，the number of self-attention heads as *A*. $BERT_{BASE}$(L=12, H=768, A=12, Total Parameters=110M),$BERT_{LARGE}$(L=24, H=1024, A=16, Total Parameters=340M).
* the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left
### Input/Output Representations
* A "sequence" refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together
* Use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary
* The first token of every sequence is always a special classification token ([CLS]).
* The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks
* First,separate them with a special token ([SEP]),second,add a learned embedding to every token indicating whether it belongs to sentence *A* or sentence *B*.
* we denote input embedding as E, the final hidden vector of the special [CLS] token as *C* ∈ $R_{H}$, and the final hidden vector for the $i^{th}$ input token as $T_{i}$ ∈ $R_{H}$.
* For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings.
## 3.1 Pre-training BERT
* Pre-train BERT using two unsupervised tasks, described in this section。
	* Task #1: Masked LM:
		a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-toright and a right-to-left model.standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly "see itself".the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM.Mask 15% of all WordPiece tokens in each sequence at random.only predict the masked words rather than reconstructing the entire input.author are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning.The training data generator chooses 15% of the token positions at random for prediction. If the i-th token is chosen, we replace the i-th token with (1) the [MASK] token 80% of the time (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time.$T_{i}$ will be used to predict the original token with cross entropy loss.
	* Task #2: Next Sentence Prediction (NSP)
		when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext),$C$ is used for next sentence prediction $(NSP)$.BERT transfers all parameters to initialize end-task model parameters.
* Pre-training data
		the pre-training corpus we use the BooksCorpus (800M words)  and English Wikipedia (2,500M words).For Wikipedia anthor extract only the text passages and ignore lists, tables, and headers.
## 3.2 Fine-tuning BERT
* For applications involving text pairs, a common pattern is to independently encode text pairs before applying bidirectional cross attention.
* BERT instead uses the self-attention mechanism to unify these two stages, as encoding a concatenated text pair with self-attention effectively includes bidirectional cross attention between two sentences.
* For each task, we simply plug in the taskspecific inputs and outputs into BERT and finetune all the parameters end-to-end.
* At the input, sentence A and sentence B from pre-training are analogous to
	* (1) sentence pairs in paraphrasing
	* (2) hypothesis-premise pairs in entailment
	* (3) question-passage pairs in question answering
	* (4) a degenerate text-∅ pair in text classification or sequence tagging
* At the output, the token representations are fed into an output layer for tokenlevel tasks.sequence tagging or question answering, and the [CLS] representation is fed into an output layer for classification, such as entailment or sentiment analysis.
# 4 Experiments
## 4.1 GLUE
* GLUE benchmark is a collection of diverse natural language understanding tasks.
* use the final hidden vector $C$ ∈ $R^{H}$ corresponding to the first input token ([CLS]) as the aggregate representation.
* The only new parameters introduced during fine-tuning are classification layer weights $W$ ∈$R^{K×H}$.$K$ is the number of labels.
* We compute a standard classification loss with $C$ and $W$ , i.e., log(softmax($CW^{T}$ )).
* We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks
* For each task, we selected the best fine-tuning learning rate (among 5e-5, 4e-5, 3e-5, and 2e-5) on the Dev set.
* for BERTLARGE we found that finetuning was sometimes unstable on small datasets.
* Both BERTBASE and BERTLARGE outperform all systems on all tasks by a substantial margin, obtaining 4.5% and 7.0% respective average accuracy improvement over the prior state of the art.
* BERTBASE and OpenAI GPT are nearly identical in terms of model architecture apart from the attention masking.
* For the largest and most widely reported GLUE task, MNLI, BERT obtains a 4.6% absolute accuracy improvement.
* We find that BERTLARGE significantly outperforms BERTBASE across all tasks, especially those with very little training data.
## 4.2 SQuAD v1.1
* The Stanford Question Answering Dataset (SQuAD v1.1) is a collection of 100k crowdsourced question/answer pairs.
* We only introduce a start vector $S$ ∈ $R^H$ and an end vector $E$ ∈ $R^H$ during fine-tuning.
* The probability of word $i$ being the start of the answer span is computed as a dot product between $T^i$ and $S$ followed by a softmax over all of the words in the paragraph:$P_i$= $\frac{e^{S.T^i}}{\sum_j{e^{S.T^i}}}$.
* The score of a candidate span from position i to position j is defined as $S·T^i$ + $E·T^j$ , and the maximum scoring span where $j$ ≥ $i$ is used as a prediction.
* The training objective is the sum of the log-likelihoods of the correct start and end positions.
* We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32.
* We therefore use modest data augmentation in our system by first fine-tuning on TriviaQA befor fine-tuning on SQuAD.
## 4.3 SQuAD v2.0
* The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic.
* For prediction, we compare the score of the no-answer span: $s_{null}$ =$S·C$ + $E·C$ to the score of the best non-null span $s_{\widehat{i},j}$ = $max_{j≥i}$$S·T_i + E·T_j$ .
* We predict a non-null answer when $s_{\widehat{i},j}$ > $s_{null}$ + $τ$ , where the threshold $τ$ is selected on the dev set to maximize F1.
## 4.4 SWAG
* SWAG dataset contains 113k sentence-pair completion examples that evaluate grounded commonsense inference.
* 