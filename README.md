# TRADE: Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems
This is a Pytorch re-implementation of TRADE multi-domain dialogue state tracker. For more details of the research work, please refer to the [original repositry](https://github.com/jasonwu0731/trade-dst).

# Modifications (Experimental)
### Trainable word embedding + Non-trainable character embedding
With trainable word embedding, words can adapt to the trained task and hence give better representations for the task. However, the size of corpus and vocabulary is not large. This might cause a large train-test discrepancy because only a little portion of words are seen in train dataset. Therefore, I try to make a hybrid embedding composed of trainable word embeddings(GLOVE) and non-trainable character embeddings(Kazuma). The targets I want to achieve by this are as follows:
<br>
<br>
<b>1. Trainable word embeddings allow trained words to have a better representation for the trained task</b><br>
<b>2. Non-trainable char embeddings allow unseen words(OOV) and trained words to have consistent representations(from the same distribution), in favor of the copy mechanism of pointer generator network</b>

# Training
1. Modify cfg.py if necessary
> (e.g. which domains to train/test, use pretrained word embedding, resume from last checkpoint, etc.)<br>
2. Start training
```console
>>> python train.py
```

# Future Work
1. Implement parallel decoding to speed up the generator
2. Implement more metrics in order to compare to the original baseline

# Reference
```
@InProceedings{WuTradeDST2019,
  	author = "Wu, Chien-Sheng and Madotto, Andrea and Hosseini-Asl, Ehsan and Xiong, Caiming and Socher, Richard and Fung, Pascale",
  	title = 	"Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems",
  	booktitle = 	"Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  	year = 	"2019",
  	publisher = "Association for Computational Linguistics"
}
```
