# ABSTRACT
* DeBERTaV3:improves the original DeBERTa model by replacing masked language modeling (MLM) with replaced token detection (RTD), a more sample-efficient pre-training task.
* vanilla embedding sharing in ELECTRA hurts training efficiency and model performance, because the training losses of the discriminator and the generator pull token embeddings in different directions, creating the "tugof-war" dynamics.
* thus propose a new gradient-disentangled embedding sharing method that avoids the tug-of-war dynamics, improving both training efficiency and the quality of the pre-trained model.
* https://github.com/microsoft/DeBERTa.
# 1 INTRODUCTION
* it is more important to explore more energy-efficient approaches to build PLMs with fewer parameters and less computation cost while retaining high model capacity.
* The second new pre-training approach to improve efficiency is Replaced Token Detection (RTD), proposed by ELECTRA.
* RTD uses a generator to generate ambiguous corruptions and a discriminator to distinguish the ambiguous tokens from the original inputs, similar to Generative Adversarial Networks (GAN).
* Author replace MLM in DeBERTa with RTD where the model is trained as a discriminator to predict whether a token in the corrupt input is either original or replaced by a generator.DeBERTa trained with RTD significantly outperforms the model trained using MLM.
* embedding sharing hurts training efficiency and model performance, since the training losses of the discriminator and the generator pull token embeddings into opposite directions.
* The MLM used for training the generator tries to pull the tokens that are semantically similar close to each other while the RTD of the discriminator tries to discriminate semantically similar tokens and pull their embeddings as far as possible to optimize the binary classification accuracy, causing a conflict between their training objectives.
* On the other hand, we show that using separated embeddings for the generator and the discriminator results in significant performance degradation when we fine-tune the discriminator on downstream tasks, indicating the merit of embedding sharing, e.g., the embeddings of the generator are beneficial to produce a better discriminator, as argued in [Clark et al. (2020)].
* gradient-disentangled embedding sharing (GDES):the generator shares its embeddings with the discriminator but stops the gradients from the discriminator to the generator embeddings.
# 2 BACKGROUND
## 2.1 TRANSFORMER
* relative position representations are more effective for natural language understanding and generation tasks .
## 2.2 DEBERTA
* DeBERTa improves BERT with two novel components:
	* DA (Disentangled Attention):the DA mechanism uses two separate vectors: one for the content and the other for the position.
	* DA (Disentangled Attention):DeBERTa uses an enhanced mask decoder to improve MLM by adding absolute position information of the context words at the MLM decoding layer.
##  2.3 ELECTRA
### 2.3.1 MASKED LANGUAGE MODEL
* 具体看DeBERTa
### 2.3.2 REPLACED TOKEN DETECTION
* ELECTRA was trained with two transformer encoders in GAN style.
* One is called generator trained with MLM; the other is called discriminator trained with a token-level binary classifier.The generator is used to generate ambiguous tokens to replace masked tokens in the input sequence. Then the modified input sequence is fed to the discriminator. The binary classifier in the discriminator needs to determine if a corresponding token is either an original token or a token replaced by the generator. 
# 3 DEBERTAV3
## 3.1 DEBERTA WITH RTD
* Wikipedia and the bookcorpus are used as training data
* The batch size is set to 2048, and the model is trained for 125,000 steps with a learning rate of 5e-4 and warmup steps of 10,000.use λ " 50 with the same optimization hyperparameters.
## 3.2 TOKEN EMBEDDING SHARING IN ELECTRA
* Embedding Sharing (ES), allows the generator to provide informative inputs for the discriminator and reduces the number of parameters to learn.
* generator's and the discriminator's objectives interfere with each other and slow down the training convergence.
* Let E and gE be the token embeddings and their gradients.
* MLM encourages the embeddings of semantically similar tokens to be close to each other, while RTD tries to separate them to make the classification easier.
* NES converges faster than ES, as expected, because it avoids the conflicting gradients between the two downstream tasks.
* NES produces two distinct embedding models, with $E_G$ being more semantically coherent than $E_D$.
* the embeddings learned by NES do not lead to any significant improvement on two representative downstream NLU tasks.
## 3.3 GRADIENT-DISENTANGLED EMBEDDING SHARING
* GDES can achieve the same converging speed as NES, but without sacrificing the quality of the embeddings.
* $E_D = sg(E_G) +E_∆$
	* initialize $E_∆$ as a zero matrix
	* generate the inputs for the discriminator using the generator and update both $E_G$ and $E_D$ with the MLM loss.
	* run the discriminator on the generated inputs and update $E_D$ with the RTD loss, but only through $E_∆$.
	* $E_∆$ to $E_G$ and save the resulting matrix as $E_D$ for the discriminator.
# 4 EXPERIMENT
## 4.1 MAIN RESULTS ON NLU TASKS
* train those models with 160GB datad 
* use the same SentencePiece vocabulary as DeBERTaV2
* All the models are trained for 500,000 steps with a batch size of 8192 and warming up steps of 10,000. The learning rate for base and small model is 5e-4, while the learning rate for large model is 3e-4.
* a fixed version of Adam with weight decay, and set $β_1$= 0.9, $β_2$ = 0.98 for the optimizer.
### 4.1.1 PERFORMANCE ON LARGE MODELS
* This indicates DeBERTaV3 is more data efficient and has a better generalization performance.
* DeBERTaV3large is evaluated on three categories of representative NLU benchmarks: 
	* Question Answering: SQuAD v2.0, RACE , ReCoRD , and SWAG ;
	* Natural Language Inference: MNLI
	* NER: CoNLL-2003
### 4.1.2 PERFORMANCE ON BASE AND SMALLER MODELS
* Surprisingly, even though DeBERTaV3xsmall has only half the parameters of DeBERTaV3small, it performs on par or even better than $DeBERTaV3_{small}$ on these two tasks.
## 4.2 MULTILINGUAL MODEL
* Zero-shot cross-lingual transfer is to fine-tune the model with English data only and evaluate it on multi-lingual test sets.
* translate-train-all is to fine-tune the model with English data and multi-lingual data translated from English data which is provided together with the XNLI dataset, and then evaluate the model on multi-lingual test sets.
* All this clearly demonstrates the efficiency of the DeBERTaV3 models. The consistent improvements over a large range of the downstream tasks also show the huge value of improving pre-trained language models.
# 5 CONCLUSIONS
* question:
	* simply combining these two models leads to pre-training instability and inefficiency, due to a critical interference issue between the generator and the discriminator in the RTD framework which is well known as the "tug-of-war" dynamics.
*  function:
	* Author introduce a novel embedding sharing paradigm called GDES, which is the main innovation and contribution of this work.GDES allows the discriminator to leverage the semantic information encoded in the generator's embedding layer without interfering with the generator's gradients and thus improves the pre-training efficiency.