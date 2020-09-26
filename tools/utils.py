import torch
import numpy as np
import os
from .metrics import compute_acc

def mask_words_as_unk(word_id_seq, context_lens, mask_ratio=0.1, unk_id=0):
	'''
	Args:
	- word_id_seq (torch.tensor, int64): [bs, batch_max_seq]
	- context_lens (list): indicating number of tokens for each sample in the batch
	- pad_id (int): pad token id
	- unk_id (int): unknown token id

	Returns:
	- masked_seq (torch.tensor, int64): [bs, batch_max_seq]
	'''
	mask = torch.zeros_like(word_id_seq)

	for bi, seq in enumerate(mask):
	    seq_len = context_lens[bi]
	    mask_size = int(mask_ratio * seq_len)
	    for mask_idx in np.random.choice(seq_len, mask_size):
	        mask[bi, mask_idx] = 1

	mask = mask.bool()

	masked_seq = word_id_seq.masked_fill(mask, unk_id)

	return masked_seq

def extend_labels_with_oov(generate_y_plain, extended_word2id, vocab):
	'''
	Args:
	# - generate_y (torch.tensor, int64): word id seq from dataloader [bs, n_slots, max_dec_len]
	- generate_y_plain (list): list of list of generate_y word [bs, n_slots, max_dec_len]
	- extended_word2id (dict): included the OOV words seen in batch
	
	Returns:
	- extended_generate_y (torch.tensor, int64): y labels with indexed OOV
	'''
	extended_generate_y = []
	for y_plain in generate_y_plain:
	    batch_y = []
	    for y in y_plain:
	        words = vocab.tokenize(y)
	        batch_y.append([extended_word2id.get(w, extended_word2id['[UNK]']) for w in words])
	    extended_generate_y.append(batch_y)

	return torch.tensor(extended_generate_y, dtype=torch.int64)

def evaluate(model, criterion, dataloader, datatype):
	'''
	Args:
	- datatype (str): 'val'/'test'
	'''
	assert datatype in ['val', 'test'], "datatype can only be either 'val' or 'test'"

	model.eval()
	device = next(trade_model.parameters()).device

	if datatype == 'val':
		action = 'Validation'
	elif datatype == 'test':
		action = 'Test'

	mean_loss, mean_gen_acc, mean_gate_acc = 0, 0, 0
	pred_slots = dataloader.dataset.slots
	with torch.no_grad():
		for batch_i, data in enumerate(tqdm(dataloader, desc=f'{action}: ')):
			context_plain = data['context_plain']
			context_lens = data['context_lens']
			generate_y_plain = data['generate_y_plain']
			y_lengths = data['y_lengths']
			context = data['context'].to(device)
			gating_label = data['gating_label'].to(device)

			# Forward
			slot_gates, p_finals, all_pred_word_id, extended_word2id = model(context, 
																		     context_plain, 
																		     context_lens, 
																		     pred_slots)

			# indexing V & OOV for loss calculation
			generate_y = extend_labels_with_oov(generate_y_plain, extended_word2id, model.vocab).to(device)

			loss = criterion(slot_gates, p_finals, gating_label, generate_y, y_lengths)

			gate_acc, gen_acc = compute_acc(slot_gates.detach(), 
											all_pred_word_id.detach(), 
											gating_label, 
											generate_y, 
											y_lengths, 
											model.gating_dict, 
											pad_id=model.vocab.word2id['[PAD]'])

			mean_loss += loss.detach().item()
			mean_gate_acc += gate_acc
			mean_gen_acc += gen_acc

	mean_loss /= (batch_i+1)
	mean_gate_acc /= (batch_i+1)
	mean_gen_acc /= (batch_i+1)

	return mean_loss, mean_gate_acc, mean_gen_acc

def save_ckpt_for_resume_training(epoch, model, optimizer, val_loss, save_dir):
	'''
	Args:
	- epoch (int)
	- model (class nn.Module)
	- optimizer (class torch.optim)
	- loss (torch.tensor, float32)
	- save_dir (str): directory to save the checkpoint files
	'''
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
            }, os.path.join(save_dir, f'ckpt_ep{epoch}.tar'))

def load_ckpt_for_resume_training(ckpt_path, model, optimizer):
	'''
	Args:
	- ckpt_path (str): path to checkpoint file
	- model (class nn.Module)
	- optimizer (class torch.optim)
	'''
	ckpt = torch.load(ckpt_path)
	model.load_state_dict(ckpt['model_state_dict'])
	optimizer.load_state_dict(ckpt['optimizer_state_dict'])
	epoch = ckpt['epoch']
	val_loss = ckpt['val_loss']
	return epoch, val_loss