import os
import json
import numpy as np
import torch
from tqdm import tqdm
from embeddings import GloveEmbedding, KazumaCharEmbedding, ConcatEmbedding

class Vocab():
	def __init__(self, pretrained_embed_path=None, corpus=None):
		'''
		if pretrained_embed_path is provided, corpus will be ignored.

		Input:
		- domain_dict_path (str): path to domain dict json file
		- pretrained_embed (str): path to pretrained embedding json file
		- corpus (list): list of long text(str)
		'''

		assert not (pretrained_embed_path == None and corpus == None), 'Either <pretrained_embed_path> or <corpus> must be provided'

		if pretrained_embed_path == None:
			self.word2id, self.embed_list = self._create_word2id_from_corpus(corpus)
		else:
			self.word2id, self.embed_list = self._load_from_pretrained_embed(pretrained_embed_path)


		self.id2word = {idx: word for word, idx in self.word2id.items()}

	def tokenize(self, text):
		'''
		Tokenize text into words.

		Input:
		- text (str)

		Output:
		- word_seq (list): list of words(str)
		'''
		word_seq = [w.strip() for w in text.split(' ')]
		return word_seq

	def words_to_index(self, word_seq):
		'''
		Convert word to word index. If word is unseen, it will be treated as unknown token ('UNK') 

		Input:
		- word_seq (list): list of words(str)

		Output:
		- word_id_seq (list): list of word indices
		'''
		word_id_seq = [self.word2id.get(word, self.word2id['[UNK]']) for word in word_seq]
		return word_id_seq

	def _create_word2id_from_corpus(self, corpus):
		'''
		tokenize the corpus into words and create word2id from them and save the embeddings to json file

		Input:
		- corpus (list): list of long text(str)

		Output:
		- word2id (dict)
		- embed_list (list): list of vectors(list)
		'''

		embed = [GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True), KazumaCharEmbedding()]
		embed_dim = sum([e.d_emb for e in embed])
		embed = ConcatEmbedding(embed, default='zero')

		word2id = {
			'[UNK]': 0,
			'[SEP]': 1,
			'[PAD]': 2,
			'[SOS]': 3,
			'[EOS]': 4
		}

		# Randomly initialize vectors for special tokens
		embed_list = [np.random.uniform(-1,1,embed_dim).tolist() for _ in range(len(word2id))]

		for passage in tqdm(corpus, desc='Creating vocabulary from data corpus: '):
			word_seq = self.tokenize(passage)
			for word in word_seq:
				if word2id.get(word) == None:
					word2id[word] = len(word2id)
					embed_list.append(embed.emb(word))

		# Save the embeddings json file
		save_path = './trained_vocab/'
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		with open(os.path.join(save_path, 'embeddings.json'), 'w') as f:
			embeddings_dict = {word: embed_list[idx] for word, idx in word2id.items()}
			json.dump(embeddings_dict, f)

		return word2id, embed_list

	def _load_from_pretrained_embed(self, pretrained_embed_path):
		'''
		Input:
		- pretrained_embed_path (str): path to pretrained embedding json file
		'''
		with open(pretrained_embed_path, 'r') as f:
			embeddings_dict = json.load(f)  # dict

		for idx, (word, vector) in enumerate(tqdm(embeddings_dict.items(), desc='Reading pretrained embeddings: ')):
			word2id[word] = idx
			embed_list.append(vector)

		return word2id, embed_list