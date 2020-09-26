import torch
import numpy as np

def compute_acc(slot_gates, all_pred_word_id, gating_label, generate_y, y_lengths, gating_dict, pad_id=2):
	'''
	Args:
	- slot_gates (torch.tensor, float32): [bs, n_slots, n_gates]
	- all_pred_word_id (torch.tensor, int64): [bs, n_slots, max_dec_len]
	- gating_label (torch.tensor, int64): [bs, n_slots]
	- generate_y (torch.tensor, int64): word id seq from dataloader [bs, n_slots, max_dec_len]
	- y_lengths (torch.tensor, int64): [bs, n_slots] for masking pad
	- gating_dict (dict): gate to id map
	- pad_id (int)

	Returns:
	- gate_acc (float): batch accuracy of slot gate classifications
	- gen_acc (float): batch accuracy of slot-value generation
	'''
	eps = 1e-16  # avoid zero division

	pred_gates = slot_gates.argmax(dim=-1)  # [bs, n_slots]
	gate_correct_count = (pred_gates == gating_label).sum(dim=-1)
	gate_acc = gate_correct_count / (torch.ones_like(gate_correct_count) * gating_label.shape[1]).float()
	gate_acc = gate_acc.mean()


	# Mask all pad_token & only count those with gating labels = 'ptr' (i.e. 0)
	total_ptr_nums = (gating_label == 0).sum(dim=-1) + eps # [bs]
	ptr_mask = (gating_label == 0)
	true_mask = ptr_mask.unsqueeze(-1).expand_as(all_pred_word_id) * (generate_y != pad_id)

	gen_scores = (generate_y == all_pred_word_id) * true_mask
	gen_scores = (gen_scores.sum(dim=-1) == true_mask.sum(dim=-1))  # [bs, n_slots]
	gen_scores = (ptr_mask * gen_scores).sum(-1)  # [bs]
	gen_acc = (gen_scores/total_ptr_nums).mean()

	return gate_acc.item(), gen_acc.item()


def compute_f1_score():
	pass
