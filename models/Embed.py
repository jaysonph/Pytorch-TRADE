import torch
import torch.nn as nn
from embeddings import KazumaCharEmbedding

class CombinedEmbedding(nn.Module):
	def __init__(self, vocab, input_size, use_pretrained_word_emb=False, train_word_embed=False):
		super().__init__()
		# Use pretrained embeddings or not
		self.vocab = vocab

		if use_pretrained_word_emb:
			self.word_emb = nn.Embedding.from_pretrained(torch.tensor(self.vocab.word_embed_list))
		else:
			vocab_size = len(self.vocab.word2id)
			self.word_emb = nn.Embedding(vocab_size, input_size)

		# Train embeddings or not
		self.word_emb.weight.requires_grad = train_word_embed  # shared between encoder and state generators

		# Character embedding is non-trainable and generated on the fly
		self.char_emb = KazumaCharEmbedding()

		# Non-trainable tensor
		self.char_emb_tensor = self.create_char_emb_tensor()
		self.char_emb_tensor.requires_grad = False

	def create_char_emb_tensor(self):
		# combine char embedding into the list and used later for extended embedding for indexing OOV words.
		char_emb_tensor = torch.zeros(len(self.vocab.word2id), self.char_emb.d_emb)  # [n_vocab, char_emb_dim]
		for word, idx in self.vocab.word2id.items():
			char_emb_tensor[idx, :] = torch.tensor(self.char_emb.emb(word))
		return char_emb_tensor


	def forward(self, word_id_seq, context_plain):
		'''
		Args:
		- word_id_seq (torch.tensor, int64): [bs, max_seq]
		- context_plain (list): batch of dialogue histories

		Returns:
		- embedded_seq (torch.tensor, float32): [bs, max_seq, 400]
		- extended_embeddings (torch.tensor, float32): [V+OOV, 400]
		- extended_word2id (dict): included the OOV words seen in sample
		- extended_word_id_seq (torch.tensor, int64): [bs, max_seq] OOV words are assigned an temporary id instead of '[UNK]' id
		'''
		device = self.word_emb.weight.device
		word_id_seq = word_id_seq.to(device)
		w_embedded_seq = self.word_emb(word_id_seq)  # [bs, max_seq, 300]

		# Get char embeddings, create extended embeddings & word2id & word_id_seq for OOV words
		c_embedded_seq = torch.zeros(w_embedded_seq.shape[0], w_embedded_seq.shape[1], self.char_emb.d_emb)
		c_embedded_seq[:, :, :] = torch.tensor(self.char_emb.emb('[PAD]'))
		extended_embeddings = torch.cat([self.word_emb.weight, self.char_emb_tensor.to(device)], dim=-1)  # [V, 400]
		oov_embeddings = []
		extended_word2id = self.vocab.word2id.copy()
		extended_word_id_seq = word_id_seq.clone().detach()
		for bi, context in enumerate(context_plain):
			for wi, word in enumerate(self.vocab.tokenize(context)):
				char_vec = torch.tensor(self.char_emb.emb(word)).unsqueeze(0).to(device)  # [1, 100]
				c_embedded_seq[bi, wi, :] = char_vec

				if extended_word2id.get(word) == None:  # create extended vocab for OOV words
					oov_word_vec = w_embedded_seq[bi, wi, :].unsqueeze(0)  # [1, 300]
					oov_embeddings.append(torch.cat([oov_word_vec, char_vec], dim=-1))
					extended_word_id_seq[bi, wi] = len(extended_word2id)
					extended_word2id[word] = len(extended_word2id)

		if oov_embeddings:
			oov_embeddings = torch.cat(oov_embeddings, dim=0)  # [OOV, 400]
			extended_embeddings = torch.cat([extended_embeddings, oov_embeddings], dim=0)  # [V+OOV, 400]

		c_embedded_seq = c_embedded_seq.to(device)
		embedded_seq = torch.cat([w_embedded_seq, c_embedded_seq], dim=-1)  # [bs, max_seq, 400]

		return embedded_seq, extended_embeddings, extended_word2id, extended_word_id_seq



