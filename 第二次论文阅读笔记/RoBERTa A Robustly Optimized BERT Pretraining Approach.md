# Abstract
* Author present a replication study of BERT pretraining  that carefully measures the impact of many key hyperparameters and training data size.
# 1 Introduction
* Self-training methods such as ELMo ([Peters et al.,2018]), GPT ([Radford et al., 2018]), BERT ([Devlin et al., 2019]), XLM ([Lample and Conneau,2019]), and XLNet ([Yang et al., 2019]) have brought significant performance gains, but it can be challenging to determine which aspects of the methods contribute the most.
* Training is computationally expensive, limiting the amount of tuning that can be done, and is often done with private training data of varying sizes, limiting our ability to measure the effects of the modeling advances.
* Author present a replication study of BERT pretraining,which includes a careful evaluation of the effects of hyperparmeter tuning and training set size.
* Author present RoBERTa, that can match or exceed the performance of all of the post-BERT methods.
* modifications:
	* (1) training the model longer, with bigger batches, over more data.
	* (2) removing the next sentence prediction objective.
	* (3) training on longer sequences
	* (4) dynamically changing the masking pattern applied to the training data.
* Author re-establish that BERT's masked language model training objective is competitive with other recently proposed training objectives such as perturbed autoregressive language modeling.
* In summary, the contributions of this paper are:
	* (1)Author present a set of important BERT design choices and training strategies and introduce alternatives that lead to better downstream task performance;
	* (2)Author use a novel dataset, CCNEWS, and confirm that using more data for pretraining further improves performance on downstream tasks.
	* (3) Author training improvements show that masked language model pretraining, under the right design choices, is competitive with all other recently published methods.
# 2 Background
## 2.1 Setup
* BERT takes as input a concatenation of two segments (sequences of tokens), $x_1, . . . , x_N$and $y_1, . . . , y_M$.
* The two segments are presented as a single input sequence to BERT with special tokens delimiting them:$[CLS ]$, $x_1, . . . , x_N$ , $[SEP ]$, $y_1, . . . , y_M$ , $[EOS ]$.
* $M$ and $N$ are constrained such that $M + N < T$ , where $T$ is a parameter that controls the maximum sequence length during training.
* The model is first pretrained on a large unlabeled text corpus and subsequently finetuned using end-task labeled data.
## 2.2 Architecture
* Author  use a transformer architecture with $L$ layers,each block uses $A$ self-attention heads and hidden dimension $H$.
## 2.3 Training Objectives
* During pretraining, BERT uses two objectives: masked language modeling and next sentence prediction.
* Masked Language Model (MLM):
	* A random sample of the tokens in the input sequence is selected and replaced with the special token$[MASK ]$
	* The MLM objective is a cross-entropy loss on predicting the masked tokens.
	* BERT uniformly selects 15% of the input tokens for possible replacement. Of the selected tokens, 80% are replaced with $[MASK ]$, 10% are left unchanged,and 10% are replaced by a randomly selected vocabulary token.
	* In the original implementation, random masking and replacement is performed once in the beginning and saved for the duration of training, although in practice, data is duplicated so the mask is not always the same for every training sentence.
* Next Sentence Prediction (NSP):
	* NSP is a binary classification loss for predicting whether two segments follow each other in the original text.
	* Positive examples are created by taking consecutive sentences from the text corpus. Negative examples are created by pairing segments from different documents.
## 2.4 Optimization
* BERT is optimized with Adam ([Kingma and Ba,2015]) using the following parameters: $β_1$ = 0.9,$β_2$ = 0.999, $\epsilon$ = 1e-6 and $L_2$ weight decay of 0.01.The learning rate is warmed up over the first 10,000 steps to a peak value of 1e-4, and then linearly decayed.BERT trains with a dropout of 0.1 on all layers and attention weights, and a GELU activation function.
* Models are pretrained for $S$ = 1,000,000 updates, with minibatches containing $B$ = 256 sequences of maximum length $T$ = 512 tokens.
## 2.5 Data
* BERT is trained on a combination of $BOOKCOR-PUS$  plus English $W_{IKIPEDIA}$, which totals 16GB of uncompressed text.
# 3 Experimental Setup
## 3.1 Implementation
* Author reimplement BERT in FAIRSEQ
* Author additionally found training to be very sensitive to the Adam epsilon term, and in some cases we obtained better performance or improved stability after tuning it.
* Author found setting $β_2$ = 0.98 to improve stability when training with large batch sizes.
* Author train with mixed precision floating point arithmetic on DGX-1 machines, each with 8 ×32GB Nvidia V100 GPUs interconnected by Infiniband.
## 3.2 Data
* BERT-style pretraining crucially relies on large quantities of text.
* Author use the following text corpora:
	* $BOOKCORPUS$  plus English $W_{IKIPEDIA}$.
	* CC-NEWS:collected from the English portion of the CommonCrawl News dataset.
	* OPENWEBTEXT:an open-source recreation of the WebText corpus.
	* STORIES:containing a subset of CommonCrawl data filtered to match the story-like style of Winograd schemas.
## 3.3 Evaluation
* Author evaluate our pretrained models on downstream tasks using the following three benchmarks:
*  GLUE:
	* The General Language Understanding Evaluation benchmark is a collection of 9 datasets for evaluating natural language understanding systems.
	* Tasks are framed as either single-sentence classification or sentence-pair classification tasks.
* SQuAD:
	* The Stanford Question Answering Dataset (SQuAD) provides a paragraph of context and a question.
	* The task is to answer the question by extracting the relevant span from the context.
	* For SQuAD V2.0, we add an additional binary classifier to predict whether the question is answerable, which Author train jointly by summing the classification and span loss terms.
*  RACE:
	* The ReAding Comprehension from Examinations task is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions.
	* In RACE, each passage is associated with multiple questions.For every question, the task is to select one correct answer from four options.RACE has significantly longer context than other popular reading comprehension datasets and the proportion of questions that requires reasoning is very large.
# 4 Training Procedure Analysis
* In the section,author begin begin by training BERT models with the same configuration as $BERT_{BASE}$ ($L$ = 12, $H$ = 768, $A$ = 12, 110M params).
## 4.1 Static vs. Dynamic Masking
* To avoid using the same mask for each training instance in every epoch, training data was duplicated 10 times so that each sequence is masked in 10 different ways over the 40 epochs of training.each training sequence was seen with the same mask four times during training.
* reimplementation with static masking performs similar to the original BERT model, and dynamic masking is comparable or slightly better than static masking.
## 4.2 Model Input Format and Next Sentence Prediction
* In the original BERT,the model observes two concatenated document segments, which are either sampled contiguously from the same document (with p = 0.5) or from distinct documents.the model is trained to predict whether the observed document segments come from the same or distinct documents via an auxiliary Next Sentence Prediction (NSP) loss.
* [Devlin et al. (2019]) observe that removing NSP hurts performance, with significant performance degradation on QNLI, MNLI, and SQuAD 1.1.
* To better understand this discrepancy, author compare several alternative training formats:
	* SEGMENT-PAIR+NSP:This follows the original input format used in BERT with the NSP loss.Each input has a pair of segments, which can each contain multiple natural sentences, but the total combined length must be less than 512 tokens.
	* SENTENCE-PAIR+NSP:Each input contains a pair of natural sentences, either sampled from a contiguous portion of one document or from separate documents.
	* FULL-SENTENCES: Each input is packed with full sentences sampled contiguously from one or more documents, such that the total length is at most 512 tokens.Inputs may cross document boundaries.
	* DOC-SENTENCES: Inputs are constructed similarly to FULL-SENTENCES, except that they may not cross document boundaries.
* Author first compare the originalSEGMENT-PAIR input format from [Devlin et al.(2019]) to the SENTENCE-PAIR format; both formats retain the NSP loss, but the latter uses single sentences. Author find that using individual sentences hurts performance on downstream tasks, which we hypothesize is because the model is not able to learn long-range dependencies.
* Author next compare training without the NSP loss and training with blocks of text from a single document (DOC-SENTENCES). We find that this setting outperforms the originally published BERTBASE results and that removing the NSP loss matches or slightly improves downstream task performance, in contrast to [Devlin et al. (2019)]. It is possible that the original BERT implementation may only have removed the loss term while still retaining the SEGMENT-PAIR input format.
* Finally author find that restricting sequences to come from a single document (DOC-SENTENCES) performs slightly better than packing sequences from multiple documents (FULL-SENTENCES).
## 4.3 Training with large batches
* BERT is also amenable to large batch training.
* Author observe that training with large batches improves perplexity for the masked language modeling objective, as well as end-task accuracy.
## 4.4 Text Encoding
* Byte-Pair Encoding (BPE) (Sennrich et al., 2016) is a hybrid between character- and word-level representations that allows handling the large vocabularies common in natural language corpora.Instead of full words, BPE relies on subwords units, which are extracted by performing statistical analysis of the training corpus.
* BPE vocabulary sizes typically range from 10K-100K subword units.
* unicode characters can account for a sizeable portion of this vocabulary when modeling large and diverse corpora
* Author instead consider training BERT with a larger byte-level BPE vocabulary containing 50K subword units, without any additional preprocessing or tokenization of the input.
* Author believe the advantages of a universal encoding scheme outweighs the minor degredation in performance and use this encoding in the remainder of our experiments.
# 5 RoBERTa(Robustly optimized BERT approach)
* RoBERTa is trained with dynamic masking , FULL-SENTENCES without NSP loss , large mini-batches  and a larger byte-level BPE 
* XLNet architecture is pretrained using nearly 10 times more data than the original BERT .
* RoBERTa following the BERTLARGE architecture ($L$ = 24,$H$ = 1024, $A$ = 16, 355M parameters).
* Results:
	* RoBERTa provides a large improvement over the originally reported BERTLARGE results.
	* further improvements in performance across all downstream tasks, validating the importance of data size and diversity in pretraining.
## 5.1 GLUE Results
* finetune RoBERTa separately for each of the GLUE tasks, using only the training data for the corresponding task.consider a limited hyperparameter sweep for each task, with batch sizes ∈ {16, 32}and learning rates ∈ {1e−5, 2e−5, 3e−5}, with a linear warmup for the first 6% of steps followed by a linear decay to 0
* Task-specific modifications:
	* QNLI:Author report development set results based on a pure classification approach.
	* WNLI:Author can only make use of the positive training examples, which excludes over half of the provided training examples
* RoBERTa achieves state-of-the-art results on all 9 of the GLUE task development sets.
* RoBERTa does not depend on multi-task finetuning, unlike most of the other top submissions.
## 5.2 SQuAD Results
* Author only finetune RoBERTa using the provided SQuAD training data
* On the SQuAD v2.0 development set, RoBERTa sets a new state-of-the-art, improving over XLNet by 0.4 points (EM) and 0.6 points (F1).
* single RoBERTa model outperforms all but one of the single model submissions, and is the top scoring system among those that do not rely on data augmentation.
## 5.3 RACE Results
* systems are provided with a passage of text, an associated question, and four candidate answers.Systems are required to classify  which of the four candidate answers is correct.
* Author modify RoBERTa for this task by concatenating each candidate answer with the corresponding question and passage.then encode each of these four sequences and pass the resulting $[CLS]$representations through a fully-connected layer, which is used to predict the correct answer.
# 6 Related Work
* Author goal was to replicate, simplify, and better tune the training of BERT, as a reference point for better understanding the relative performance of all of these methods.
# 7 Conclusion
* performance can be substantially improved by training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data.
* Author additionally use a novel dataset, CC-NEWS, and release our models and code for pretraining and finetuning https://github.com/pytorch/fairseq.