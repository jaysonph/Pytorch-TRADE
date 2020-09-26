import torch
import torch.nn as nn

class TRADELoss(nn.Module):
	def __init__(self, cfg):
		'''
		Args:
		- cfg: from cfg.training['hyperparams']
		'''
		super().__init__()

		self.alpha = cfg['alpha']
		self.beta = cfg['beta']
		self.cross_entropy = nn.CrossEntropyLoss()

	def forward(self, slot_gates, p_finals, gating_label, generate_y, y_lengths):
		'''
		Args:
		- slot_gates (torch.tensor, float32): [bs, n_slots, n_gates]
		- p_finals (torch.tensor float32): [bs, n_slots, max_dec_len, V+OOV]
		- gating_label (torch.tensor, int64): [bs, n_slots]
		- generate_y (torch.tensor, int64): [bs, n_slots, max_dec_len]
		- y_lengths (torch.tensor, int64): [bs, n_slots] for masking pad when calculate loss

		Returns:
		- combined_loss (torch.tensor, float32)
		'''
		slot_gates = slot_gates.reshape(-1, 3)  # [bs, n_slots, n_gates] --> [bs*n_slots, n_gates]
		gating_label = gating_label.reshape(-1)  # [bs, n_slots] --> [bs*n_slots]
		loss_g = self.cross_entropy(slot_gates, gating_label)

		loss_v = torch.tensor(0., requires_grad=True).to(slot_gates.device)
		for bi, y_length in enumerate(y_lengths):
			for slot_i, len_y in enumerate(y_length):
				pred_y = p_finals[bi, :, :len_y, :].reshape(-1, p_finals.shape[-1])  # [n_slots, len_y, V+OOV] --> [n_slots*len_y, V+OOV]
				true_y = generate_y[bi, :, :len_y].reshape(-1)  # [bs, n_slots, len_y] --> [bs*n_slots*len_y]
				loss_v = loss_v + self.cross_entropy(pred_y, true_y)

		return self.alpha*loss_g + self.beta*loss_v