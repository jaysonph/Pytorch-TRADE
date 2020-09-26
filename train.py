from tqdm import tqdm
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import cfg
from mwoz_dataset import MultiWOZDataset
from tools.utils import mask_words_as_unk, extend_labels_with_oov, save_ckpt_for_resume_training, load_ckpt_for_resume_training
from tools.metrics import compute_acc
from models.TRADE import TRADE
from loss import TRADELoss

hyperparams = cfg.training['hyperparams']
train_settings = cfg.training['settings']
filepaths = cfg.training['filepaths']

if train_settings['use_cuda']:
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
	device = 'cpu'

# ================================== Create Dataset ==================================
print('Creating datasets...')

train_set = MultiWOZDataset(cfg,
                            dataset_type = 'train',
                            training = True)

dev_set = MultiWOZDataset(cfg,
                          dataset_type = 'dev',
                          training = True,
                          vocab = train_set.vocab)

test_set = MultiWOZDataset(cfg,
                           dataset_type = 'test',
                           training = True,
                           vocab = train_set.vocab)

train_loader = DataLoader(dataset=train_set,
                          batch_size=hyperparams['batch_size'],
                          shuffle=True,
                          num_workers=8,
                          collate_fn=train_set.collate_fn)

dev_loader = DataLoader(dataset=dev_set,
                          batch_size=hyperparams['batch_size'],
                          shuffle=False,
                          num_workers=8,
                          collate_fn=dev_set.collate_fn)

test_loader = DataLoader(dataset=test_set,
                          batch_size=hyperparams['batch_size'],
                          shuffle=False,
                          num_workers=8,
                          collate_fn=test_set.collate_fn)

# ================================== Create models ==================================
print('Creating models...')

model = TRADE(cfg.model, train_set.vocab, train_set.gating_dict).to(device)

# ================================== Create loss functions & optimizer ==================================
print('Creating loss functions & optimizer...')

criterion = TRADELoss(hyperparams).to(device)

optimizer = torch.optim.Adam(model.parameters(), hyperparams['learning_rate'], hyperparams['betas'])

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

# ================================== Training loop ==================================
print('Training starting...')
for dataset in [train_set, dev_set, test_set]:
	print(f'''
=========================== {dataset.dataset_type} dataset ===========================
{'data: ':>4} {len(dataset):>4}
{'# slots: ':>4} {len(dataset.slots)}
{'domains: ':>4} {dataset.interest_domains}
		''')
for name, param in hyperparams.items():
	print(f"{name+':':<25} {json.dumps(param):>8}")
print(f"{'device:':<25} {device:>8}")
print('\n')

ckpt_save_dir = filepaths['ckpt_save_dir']
ckpt_path = filepaths['training_ckpt']

n_epochs = hyperparams['n_epochs']
context_mask_prob = hyperparams['context_mask_prob']
context_mask_ratio = hyperparams['context_mask_ratio']
domain_mask_prob = hyperparams['domain_mask_prob'] 
slot_mask_prob = hyperparams['slot_mask_prob']
teacher_forcing_rate = hyperparams['teacher_forcing_rate']
val_interval = train_settings['val_interval']

if ckpt_path != None:
	prev_epoch, prev_loss = load_ckpt_for_resume_training(ckpt_path, model, optimizer)
else:
	prev_epoch, prev_loss = 0, 0

for epoch in range(prev_epoch+1, n_epochs):
	train_loss = 0
	train_gate_acc = 0
	train_gen_acc = 0
	for batch_i, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1:>3}/{n_epochs}: ')):
		model.train()
		context_plain = data['context_plain']
		context_lens = data['context_lens']
		generate_y_plain = data['generate_y_plain']
		y_lengths = data['y_lengths']
		context = data['context']
		generate_y = data['generate_y'].to(device)
		gating_label = data['gating_label'].to(device)

		if np.random.rand() < context_mask_prob:
			context = mask_words_as_unk(context, context_lens, context_mask_ratio, train_set.vocab.word2id['[UNK]'])
		context = context.to(device)

		# Clear out all gradients
		optimizer.zero_grad()

		# Forward
		slot_gates, p_finals, all_pred_word_id, extended_word2id = model(context, 
																	     context_plain, 
																	     context_lens, 
																	     train_set.slots, 
																	     teacher_forcing_rate, 
																	     [generate_y, generate_y_plain],
																	     domain_mask_prob,
																	     slot_mask_prob)

		# indexing V & OOV for loss calculation
		generate_y = extend_labels_with_oov(generate_y_plain, extended_word2id, model.vocab).to(device)

		# Backward & updating
		loss = criterion(slot_gates, p_finals, gating_label, generate_y, y_lengths)
		loss.backward()
		optimizer.step()

		train_loss += loss.detach().item()


	# Metrics
	train_gate_acc, train_gen_acc = compute_acc(slot_gates.detach(), 
												all_pred_word_id.detach(), 
												gating_label, 
												generate_y, 
												y_lengths, 
												model.gating_dict, 
												pad_id=model.vocab.word2id['[PAD]'])

	train_loss /= (batch_i+1)

	print(f"Train loss: {train_loss} | Train gate acc: {train_gate_acc} | Train gen acc: {train_gen_acc}")

	if epoch % val_interval == 0:
		val_loss, val_gate_acc, val_gen_acc = evaluate(model, dev_loader, criterion, datatype='val')
		print(f"Val loss: {val_loss} | Val gate acc: {val_gate_acc} | Val gen acc: {val_gen_acc}")

	scheduler.step(val_loss)
	save_ckpt_for_resume_training(epoch, model, optimizer, val_loss, ckpt_save_dir)




