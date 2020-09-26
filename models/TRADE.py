import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from .Embed import CombinedEmbedding

class GRUEncoder(nn.Module):
	def __init__(self, cfg, embedding_layer, vocab):
		'''
		Args:
		- cfg (dict): from cfg.model
		- embedding_layer (class nn.Module)
		- vocab (class Vocab)
		'''
		super().__init__()
		encoder_cfg = cfg['encoder_gru']

		input_size = encoder_cfg['input_dim']
		self.hidden_size = encoder_cfg['hidden_dim']
		n_layers = encoder_cfg['n_layers']
		dropout = encoder_cfg['dropout']
		bidirectional = encoder_cfg['bidirectional']

		self.embedding_layer = embedding_layer

		self.encoder_model = nn.GRU(input_size, self.hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

	def forward(self, embedded_seq, context_lens):
		'''
		Args:
		- embedded_seq (torch.tensor, int64): Padded sequence with OOV [batch_size, batch_max_seq]
		- context_lens (list): indicating number of tokens for each sample in the batch

		Returns:
		- enc_seq (torch.tensor, float32): [bs, seq, h_dim]
		- enc_hidden (torch.tensor, float32): [1, bs, h_dim]
		'''
		packed_seq = pack_padded_sequence(embedded_seq, context_lens, batch_first=True)
		enc_seq, enc_hidden = self.encoder_model(packed_seq)  # enc_seq is a PackedSequence
		enc_seq, _ = pad_packed_sequence(enc_seq, batch_first=True)  # [bs, seq, h_dim*n_dir]

		enc_seq = enc_seq[:, :, :self.hidden_size] + enc_seq[:, :, self.hidden_size:]  # [bs, seq, h_dim*n_dir] --> [bs, seq, h_dim]
		enc_hidden = torch.sum(enc_hidden, dim=0).unsqueeze(0)  # [n_dir*n_layers,bs,h_dim] --> [1, bs, h_dim]

		return enc_seq, enc_hidden

class SlotGate(nn.Module):
	def __init__(self, cfg, gating_dict):
		'''
		Args:
		- cfg (dict): from cfg.model
		'''
		super().__init__()
		encoder_cfg = cfg['encoder_gru']
		slot_gate_cfg = cfg['slot_gate']

		n_dir = 2 if encoder_cfg['bidirectional'] else 1
		input_size = encoder_cfg['hidden_dim']
		n_layers = slot_gate_cfg['n_layers']

		module_list = nn.ModuleList()
		for n in range(n_layers):
			if n != n_layers-1:
				module_list += self.linear_block(input_size)
				input_size = input_size//2
			else:  # Last output layer
				module_list += nn.ModuleList([nn.Linear(input_size, len(gating_dict))])

		self.linears = nn.Sequential(*module_list)

	def linear_block(self, input_size):
		block = nn.ModuleList([
					nn.Linear(input_size, input_size//2),
					nn.BatchNorm1d(input_size//2),
					nn.ReLU()
				])
		return block

	def forward(self, context_vector):
		'''
		Args:
		- context_vector (torch.tensor, float32) [bs, h_dim]: weighted sum of encoded vectors by attention weights
		'''
		out = self.linears(context_vector)
		return out

class Attention(nn.Module):
	def __init__(self, dec_hidden_size, enc_input_size, enc_hidden_size):
		'''
		Args:
		- dec_hidden_size (int): decoder hidden size
		- enc_input_size (int): encoder input size
		- enc_hidden_size (int): encoder hidden size
		'''
		super().__init__()

		self.p_gen_W = nn.Linear(dec_hidden_size+enc_input_size+enc_hidden_size, 1)

	def forward(self, enc_seq, dec_inp_emb, dec_hidden, extended_embeddings, extended_word_id_seq, context_lens):
		'''
		Args:
		- enc_seq (torch.tensor, float32): [bs, max_seq, h_dim]
		- dec_inp_emb (torch.tensor, float32): [bs, 1, h_dim]
		- dec_hidden (torch.tensor, float32): [1, bs, h_dim]
		- extended_embeddings (torch.tensor, float32): [V+OOV, 400]
		- extended_word_id_seq (torch.tensor, int64): [bs, max_seq] OOV words are assigned an temporary id instead of '[UNK]' id
		- context_lens (list): indicating number of tokens for each sample in the batch

		Returns:
		- context_vec (torch.tensor, float32): [bs, h_dim]
		- p_final (torch.tensor float32): [bs, V+OOV]
		'''
		p_vocab = self.attend_vocab(dec_hidden, extended_embeddings)
		ctx_attn_scores, p_history = self.attend_context(enc_seq, 
														dec_hidden, 
														extended_embeddings, 
														extended_word_id_seq, 
														context_lens)

		context_vec = torch.bmm(ctx_attn_scores.unsqueeze(1), enc_seq).squeeze(1)  # [bs, 1, h_dim] --> [bs, h_dim]

		# Generating final distributions over Vocab and Out-of-Vocab
		p_gen = self.p_gen_W(torch.cat([dec_hidden.squeeze(0), dec_inp_emb.squeeze(1), context_vec], dim=-1))  # [bs, 1]
		p_gen = p_gen.expand_as(p_vocab)

		p_final = p_gen * p_vocab + (1-p_gen) * p_history

		return context_vec, p_final


	def attend_vocab(self, dec_hidden, extended_embeddings):
		'''
		Args:
		- dec_hidden (torch.tensor, float32): [1, bs, hidden_dim]
		- extended_embeddings (torch.tensor, float32): [V+OOV, 400]

		Returns:
		- p_vocab (torch.tensor, float32): [bs, V+OOV]
		'''
		p_vocab = torch.matmul(dec_hidden.squeeze(0), extended_embeddings.permute(1,0))  # [bs, emb_dim]*[emb_dim, V+OOV] --> [bs, V+OOV]
		p_vocab = nn.Softmax(dim=-1)(p_vocab)
		return p_vocab


	def attend_context(self, enc_seq, dec_hidden, extended_embeddings, extended_word_id_seq, context_lens):
		'''
		Args:
		- enc_seq (torch.tensor, float32): seq_out from GRUEncoder [bs, max_seq, h_dim] 
		- dec_hidden (torch.tensor, float32): hidden states from GRUEncoder [1, bs, h_dim] 
		- context_lens (list): indicating number of tokens for each sample in the batch
		- extended_embeddings (torch.tensor, float32): [V+OOV, 400]
		- extended_word_id_seq (torch.tensor, int64): [bs, max_seq] OOV words are assigned an temporary id instead of '[UNK]' id
		
		Returns:
		- ctx_attn_scores (torch.tensor, float32): [bs, max_seq]
		- p_history (torch.tensor, float32): [bs, V+OOV]
		'''
		ctx_attn_scores = torch.bmm(enc_seq, dec_hidden.permute(1,2,0)).squeeze(-1)  # [bs, max_seq]
		for i, l in enumerate(context_lens):
			ctx_attn_scores[i, l:] = -np.inf  # mask padding for scoring
		ctx_attn_scores = nn.Softmax(dim=-1)(ctx_attn_scores)

		p_history = torch.zeros(enc_seq.shape[0], extended_embeddings.shape[0]).to(ctx_attn_scores.device)  # [bs, V+OOV]
		p_history.scatter_add_(1, extended_word_id_seq, ctx_attn_scores)

		return ctx_attn_scores, p_history

class GRUStateGenerator(nn.Module):
	def __init__(self, cfg, embedding_layer, vocab, gating_dict):
		'''
		Args:
		- cfg (dict): from cfg.model
		- embedding_layer (class nn.Module)
		- vocab (class Vocab)
		'''
		super().__init__()
		encoder_cfg = cfg['encoder_gru']
		decoder_cfg = cfg['decoder_gru']

		enc_hidden_size = encoder_cfg['hidden_dim']
		dec_hidden_size = decoder_cfg['hidden_dim']
		dec_num_layers = decoder_cfg['n_layers']
		dropout = decoder_cfg['dropout']

		self.enc_input_size = encoder_cfg['input_dim']
		self.max_decode_len = decoder_cfg['max_decode_len']
		self.gating_dict = gating_dict

		self.decoder_model = nn.GRU(enc_hidden_size, dec_hidden_size, dec_num_layers, dropout=dropout, bidirectional=False, batch_first=True)
		self.attention_layer = Attention(dec_hidden_size, self.enc_input_size, enc_hidden_size)
		self.slot_gate_model = SlotGate(cfg, self.gating_dict)

		self.embedding_layer = embedding_layer
		self.vocab = vocab


	def forward(self, 
				enc_seq, 
				enc_hidden, 
				extended_embeddings, 
				extended_word2id, 
				extended_word_id_seq, 
				context_lens, 
				pred_slots, 
				teacher_forcing_rate=0., 
				target_inps=None, 
				domain_mask_prob=0., 
				slot_mask_prob=0.):
		'''
		For every slot, decode step by step.
		
		Args:
		- enc_seq (torch.tensor, float32): seq_out from GRUEncoder [bs, seq, h_dim]
		- enc_hidden (torch.tensor, float32): h_n from GRUEncoder [1, bs, h_dim]
		- extended_embeddings (torch.tensor, float32): [V+OOV, 400]
		- extended_word2id (dict): included the OOV words seen in batch
		- extended_word_id_seq (torch.tensor, int64): [bs, max_seq] OOV words are assigned an temporary id instead of '[UNK]' id
		- context_lens (list): indicating number of tokens for each sample in the batch
		- pred_slots (list): length = n_slots
		- target_inps (list): [generate_y, generate_y_plain]
			- generate_y (torch.tensor, int64): word id seq from dataloader [bs, n_slots, max_dec_len]
			- generate_y_plain (list): list of list of generate_y word [bs, n_slots, max_dec_len]

		Returns:
		- slot_gates (torch.tensor, float32): [bs, n_slots, n_gates]
		- p_finals (torch.tensor float32): [bs, n_slots, max_decode_len, V+OOV]
		'''
		assert (teacher_forcing_rate == 0.) or (teacher_forcing_rate != 0. and isinstance(target_inps, list)), 'target_inps (list) must be provided for teacher forcing.'

		device = self.embedding_layer.word_emb.weight.device
		batch_size = enc_hidden.shape[1]
		all_pred_word_id = torch.zeros(batch_size, len(pred_slots), self.max_decode_len).to(device)
		p_finals = torch.zeros(batch_size, len(pred_slots), self.max_decode_len, len(extended_word2id)).to(device)
		slot_gates = torch.zeros(batch_size, len(pred_slots), len(self.gating_dict)).to(device)

		for slot_i, slot in enumerate(pred_slots):	
			domain_text, slot_text = slot.split('-')
			domain_words, slot_words = list(map(lambda x: self.vocab.tokenize(x), [domain_text, slot_text]))

			domain_emb = torch.zeros(batch_size, self.enc_input_size)
			# Mask some domains for generalizing copy mechanism to zero-shot domains and few-shot domains
			if np.random.rand() < domain_mask_prob:
				domain_words_seq = torch.tensor([[self.vocab.word_to_index('[UNK]') for _ in domain_words]])
			else:
				domain_words_seq = torch.tensor([[self.vocab.word_to_index(w) for w in domain_words]])
			embedded_d_seq, _, _, _ = self.embedding_layer(domain_words_seq, [domain_text])
			domain_emb[:, :] = embedded_d_seq.sum(dim=1)  # [bs, seq, emb_dim] --> [bs, emb_dim]

			slot_emb = torch.zeros(batch_size, self.enc_input_size)
			# Mask some slots for generalizing copy mechanism to zero-shot slots and few-shot slots
			if np.random.rand() < slot_mask_prob:
				slot_words_seq = torch.tensor([[self.vocab.word_to_index('[UNK]') for w in slot_words]])
			else:
				slot_words_seq = torch.tensor([[self.vocab.word_to_index(w) for w in slot_words]])
			embedded_s_seq, _, _, _ = self.embedding_layer(slot_words_seq, [slot_text])
			slot_emb[:, :] = embedded_s_seq.sum(dim=1)  # [bs, seq, emb_dim] --> [bs, emb_dim]


			dec_inp_emb = domain_emb + slot_emb  # [bs, embed_dim]
			dec_inp_emb = dec_inp_emb.unsqueeze(1)
			dec_inp_emb = dec_inp_emb.to(device)

			dec_hidden = enc_hidden
			# Autoregression
			for dec_step in range(self.max_decode_len):
				dec_seq, dec_hidden = self.decoder_model(dec_inp_emb, dec_hidden)  # dec_hidden(out) [n_layers, bs, h_dim]
				dec_hidden = torch.sum(dec_hidden, dim=0).unsqueeze(0)  # [n_layers, bs, hidden_dim] --> [1, bs, h_dim]
				
				context_vec, p_final = self.attention_layer(enc_seq, 
															  dec_inp_emb, 
															  dec_hidden, 
															  extended_embeddings, 
															  extended_word_id_seq, 
															  context_lens)

				# slot gate
				if dec_step == 0:
					slot_gate = self.slot_gate_model(context_vec)  # [bs, n_gates]
					slot_gates[:, slot_i, :] = slot_gate

				p_finals[:, slot_i, dec_step, :] = p_final

				pred_y_id = p_final.argmax(dim=-1)  # [bs]
				all_pred_word_id[:, slot_i, dec_step] = pred_y_id

				if np.random.rand() < teacher_forcing_rate:
					generate_y, generate_y_plain = target_inps
					batch_y_id = generate_y[:, slot_i, dec_step].unsqueeze(-1)  # [bs, 1]
					batch_y_text = []
					for batch_y in generate_y_plain:
						word = self.vocab.tokenize(batch_y[slot_i])[dec_step]
						batch_y_text.append(word)
				else:
					batch_y_id = pred_y_id.unsqueeze(-1)  # [bs, 1]
					extended_id2word = {idx:word for word, idx in extended_word2id.items()}
					batch_y_text = [extended_id2word[idx.item()] for idx in pred_y_id]


				dec_inp_emb, _, _, _ = self.embedding_layer(batch_y_id, batch_y_text)

		return slot_gates, p_finals, all_pred_word_id

class TRADE(nn.Module):
	def __init__(self, cfg, vocab, gating_dict):
		'''
		Args:
		- cfg (dict): from cfg.model
		'''
		super().__init__()
		trade_cfg = cfg['TRADE']
		encoder_cfg = cfg['encoder_gru']

		enc_input_size = encoder_cfg['input_dim']
		use_pretrained_word_emb = trade_cfg['use_pretrained_word_embed']
		self.train_word_embed = trade_cfg['train_word_embed']

		self.vocab = vocab
		self.gating_dict = gating_dict

		# Embedding layer is shared between encoder and state generators
		self.embedding_layer = CombinedEmbedding(self.vocab, enc_input_size, use_pretrained_word_emb, self.train_word_embed)

		self.encoder = GRUEncoder(cfg, self.embedding_layer, self.vocab)
		self.state_generator = GRUStateGenerator(cfg, self.embedding_layer, self.vocab, self.gating_dict)

	def forward(self, 
				word_id_seq, 
				context_plain, 
				context_lens, 
				pred_slots, 
				teacher_forcing=False, 
				target_inps=None, 
				domain_mask_prob=0., 
				slot_mask_prob=0.):
		'''
		Args:
		- word_id_seq (torch.tensor, int64): [bs, batch_max_seq]
		- context_plain (list): batch of dialogue histories
		- context_lens (list): indicating number of tokens for each sample in the batch
		- pred_slots (list): length = n_slots
		- target_inps (list): [generate_y, generate_y_plain]
			- generate_y (torch.tensor, int64): word id seq from dataloader [bs, n_slots, max_dec_len]
			- generate_y_plain (list): list of list of generate_y word [bs, n_slots, max_dec_len]

		Returns:
		- slot_gates (torch.tensor, float32): [bs, n_slots, n_gates]
		- p_finals (torch.tensor float32): [bs, n_slots, max_decode_len, V+OOV]
		- extended_word2id (dict): included the OOV words seen in batch
		'''
		embedded_seq, extended_embeddings, extended_word2id, extended_word_id_seq = self.embedding_layer(word_id_seq, context_plain)
		enc_seq, enc_hidden = self.encoder(embedded_seq, context_lens)
		slot_gates, p_finals, all_pred_word_id = self.state_generator(enc_seq, 
																	  enc_hidden, 
																	  extended_embeddings, 
																	  extended_word2id, 
																	  extended_word_id_seq, 
																	  context_lens, 
																	  pred_slots, 
																	  teacher_forcing, 
																	  target_inps,
																	  domain_mask_prob, 
																	  slot_mask_prob)

		return slot_gates, p_finals, all_pred_word_id, extended_word2id
		