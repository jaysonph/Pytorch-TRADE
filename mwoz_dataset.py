import os
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import json
import random
import numpy as np
from data.multiwoz.fix_label import fix_general_label_error
from collections import Counter
from collections import OrderedDict
from tqdm import tqdm
from tools.vocab import Vocab


class MultiWOZDataset(data.Dataset):
	def __init__(self, cfg, dataset_type, training, vocab=None):
		'''
		Args:
		- cfg
		- dials_path (str): path to dials jsons (train_dials.json, dev_dials.json, test_dials.json)
		- gating_dict_path (str): path to gating_dict.json
		- vocab (class Vocab)
		'''
		training_cfg = cfg.training
		model_cfg = cfg.model

		filepaths = training_cfg['filepaths']
		settings = training_cfg['settings']

		# filepaths
		dials_path = filepaths[f'{dataset_type}_dials_path']
		gating_dict_path = filepaths['gating_dict_path']
		ontology_path = filepaths['ontology_path']
		domain_dict_path = filepaths['domain_dict_path']
		pretrained_word_embed_path = filepaths['pretrained_word_embed_path']

		# Settings
		load_type = 'train' if dataset_type == 'dev' else dataset_type
		self.data_ratio = settings['train_data_ratio']
		self.only_domains = settings[f'{load_type}_only_domains']
		self.exclude_domains = settings[f'{load_type}_except_domains']
		assert not (self.only_domains != None and self.exclude_domains != None), "Either 'only_domains' or 'exclude_domains' could be specified, not both."

		self.dataset_type = dataset_type
		with open(gating_dict_path, 'r') as f:
			self.gating_dict = json.load(f)  # dict
		self.interest_domains, self.slots = self.get_filtered_domains_slots(ontology_path)
		self.max_decode_len = model_cfg['decoder_gru']['max_decode_len']
		self.domain_counter = None
		self.max_seq_len = 0

		with open(domain_dict_path, 'r') as f:
			self.domain2id = json.load(f)

		# Read data json
		with open(dials_path, 'r') as f:
			raw_data = json.load(f)  # List of dicts

		self.data = self._filter_and_organize_data(raw_data)

		if training and dataset_type == 'train' and self.data_ratio != 1.0:
			random.Random(8).shuffle(self.data)
			self.data = self.data[:round(len(self.data)*self.data_ratio)]

		if dataset_type == 'train':
			corpus = [turn_data['turn_uttr'] for turn_data in self.data]
			# if pretrained_word_embed_path is provided, corpus will be ignored.
			self.vocab = Vocab(pretrained_word_embed_path, corpus)
		else:
			self.vocab = vocab

	def __getitem__(self, idx):
		data = self.preprocess_data(self.data[idx])
		return data

	def __len__(self):
		return len(self.data)


	def get_filtered_domains_slots(self, ontology_path):
		with open(ontology_path, 'r') as f:
			ontology = json.load(f)
		ontology_domains = {slot: val for slot, val in ontology.items()}
		filtered_slots = [k.replace(' ','').lower() if ('book' not in k) else k.lower() for k in ontology_domains.keys()]
	    # Filter slots w.r.t. exclude_domains/only_domains for few/zero-shot task (allow multiple few/zero-shot domains)
		if self.only_domains != None:
			filtered_slots = [slot for slot in filtered_slots if slot.split('-')[0] in self.only_domains]
		if self.exclude_domains != None:
			filtered_slots = [slot for slot in filtered_slots if slot.split('-')[0] not in self.exclude_domains]

		filtered_domains = set([slot.split('-')[0] for slot in filtered_slots])

		return filtered_domains, filtered_slots

	def _filter_and_organize_data(self, raw_data):
		organized_data = []
		self.domain_counter = {}
		for dial_dict in tqdm(raw_data, desc='Reading data: '):
			ID = dial_dict['dialogue_idx']
			domains = dial_dict['domains']
			dialog_history = ''
			for domain in domains:
				if domain in self.interest_domains:
					self.domain_counter[domain] = self.domain_counter.get(domain, 0) + 1

			# Skip those dial_dicts that do not have any interest_domains at all
			if not set(self.interest_domains).intersection(set(dial_dict['domains'])):
				continue

			for ti, turn in enumerate(dial_dict['dialogue']):
				turn_id = turn['turn_idx']
				turn_domain = turn['domain']
				turn_uttr = turn['system_transcript'] + ' [SEP] ' + turn['transcript']
				dialog_history += (turn_uttr + ' [SEP] ')
				turn_belief_dict = fix_general_label_error(turn['belief_state'], False, self.slots)

				# Filter turn dialogues w.r.t. interest_domains for few/zero-shot task (allow multiple few/zero-shot domains)
				turn_belief_dict = OrderedDict([(slot, val) for slot, val in turn_belief_dict.items() if slot.split('-')[0] in self.interest_domains])

				turn_belief_list = ['-'.join(map(str, slot_val_pair)) for slot_val_pair in turn_belief_dict.items()]

				generate_y, gating_label = [], []
				for slot in self.slots:
					val = turn_belief_dict.get(slot, 'none')
					generate_y.append(val)
					gating_label.append(val if val in self.gating_dict else 'ptr')  # val == 'ptr' if not ('none' or 'dontcare')


				turn_data_detail = {
					'ID': ID,
					'domains': domains,
					'turn_domain': turn_domain,
					'turn_id': turn_id,
					'dialog_history': dialog_history.strip(),
					'turn_belief': turn_belief_list,
					'gating_label': gating_label,
					'turn_uttr': turn_uttr.strip(),
					'generate_y': generate_y
				}

				organized_data.append(turn_data_detail)

				if self.max_seq_len < len(dialog_history.strip().split(' ')):
					self.max_seq_len = len(dialog_history.strip().split(' ')) + 1

		return organized_data

	def preprocess_data(self, single_turn_data):
		'''
		Args:
		- single_turn_data (dict): including keys 
			'ID'
			'domains'
			'turn_domain'
			'turn_id'
			'dialog_history'
			'turn_belief'
			'gating_label'
			'turn_uttr', 
			'generate_y'

		Returns:
		- preprocessed_data (dict): indexed data for training
		'''
		gating_label = [self.gating_dict[label] for label in single_turn_data['gating_label']]
		context = [self.vocab.word_to_index(w) for w in self.vocab.tokenize(single_turn_data['dialog_history'])]
		context = torch.tensor(context, dtype=torch.float32)
		turn_domain = self.domain2id[single_turn_data['turn_domain']]

		generate_y, generate_y_plain = [], []
		for y in single_turn_data['generate_y']:
			y = y + ' [EOS]'
			generate_y_plain.append(y)
			y = [self.vocab.word_to_index(w) for w in self.vocab.tokenize(y)]
			generate_y.append(y)


		preprocessed_data = {
			'ID': single_turn_data['ID'],
			'turn_id': single_turn_data['turn_id'],
			'turn_belief': single_turn_data['turn_belief'],
			'gating_label': gating_label,
			'context': context,
			'context_plain': single_turn_data['dialog_history'],
			'turn_uttr_plain': single_turn_data['turn_uttr'],
			'turn_domain': turn_domain,
			'generate_y': generate_y,
			'generate_y_plain': generate_y_plain
		}

		return preprocessed_data

	def collate_fn(self, batch_data_list):
		'''
		Collate function for dataloader

		Args:
		- batch_data_list (list): list of __getitem__ outputs, each output is a dict (ref. preprocess_data)
		'''
		bs = len(batch_data_list)
		batch_data_list.sort(key=lambda x: x['context'].shape[-1], reverse=True)

		merged_batch_data = {}
		for key in batch_data_list[0].keys():
			merged_batch_data[key] = [data[key] for data in batch_data_list]

		# Pad the context tensors w.r.t. the longest one and merge
		batch_context_len = [data['context'].shape[-1] for data in batch_data_list]
		# batch_context_max_len = max(batch_context_len)
		# batch_context_tensor = self.vocab.word2id['[PAD]']*torch.ones((bs, batch_context_max_len), dtype=torch.float32)
		# for bi, data in enumerate(batch_data_list):
		# 	context_tensor = data['context']
		# 	batch_context_tensor[bi, :context_tensor.shape[-1]] = context_tensor
		context_tensor_list = [data['context'] for data in batch_data_list]
		batch_context_tensor = pad_sequence(context_tensor_list, padding_value=self.vocab.word2id['[PAD]'], batch_first=True)

		# Pack and pad the responses into tensor of shape [bs, n_slots, longest_len]
		batch_gen_y_len = [[len(slot) for slot in y] for y in merged_batch_data['generate_y']]
		# batch_gen_y_max_len = max([max(gen_y_len) for gen_y_len in batch_gen_y_len])
		n_slots = len(self.slots)
		batch_gen_y_tensor = self.vocab.word2id['[PAD]']*torch.ones((bs, n_slots, self.max_decode_len))
		for bi, data in enumerate(batch_data_list):
			for slot_i, y in enumerate(data['generate_y']):
				batch_gen_y_tensor[bi, slot_i, :len(y)] = torch.tensor(y)

		batch_gen_y_plain = []
		for bi, gen_y_plain in enumerate(merged_batch_data['generate_y_plain']):
			padded_gen_y_plain = []
			for slot_i, y_plain in enumerate(gen_y_plain):
				pad_string = ' [PAD]' * (self.max_decode_len - batch_gen_y_len[bi][slot_i])
				padded_gen_y_plain.append(y_plain + pad_string)
			batch_gen_y_plain.append(padded_gen_y_plain)

		merged_batch_data['gating_label'] = torch.tensor(merged_batch_data['gating_label'])
		merged_batch_data['turn_domain'] = torch.tensor(merged_batch_data['turn_domain'])
		merged_batch_data['context'] = batch_context_tensor.long()
		merged_batch_data['generate_y'] = batch_gen_y_tensor.long()
		merged_batch_data['generate_y_plain'] = batch_gen_y_plain
		merged_batch_data['context_lens'] = batch_context_len
		merged_batch_data['y_lengths'] = torch.tensor(batch_gen_y_len)

		return merged_batch_data



################################################################
## The class is from https://github.com/jasonwu0731/trade-dst ##
################################################################
class ImbalancedDatasetSampler(data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.turn_domain[idx]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples




if __name__ == '__main__':
	train_set = MultiWOZDataset('./dataset/multiwoz/data/train_dials.json', 
	                                   './dataset/multiwoz/gating_dict.json', 
	                                   './dataset/multiwoz/data/multi-woz/MULTIWOZ2 2/ontology.json',
	                                   './dataset/multiwoz/domain_dict.json',
	                                   dataset_type = 'train',
	                                   training = True
	                                  )

	dev_set = MultiWOZDataset('./dataset/multiwoz/data/dev_dials.json', 
	                                   './dataset/multiwoz/gating_dict.json', 
	                                   './dataset/multiwoz/data/multi-woz/MULTIWOZ2 2/ontology.json',
	                                   './dataset/multiwoz/domain_dict.json',
	                                   dataset_type = 'dev',
	                                   training = True,
	                                   vocab = train_set.vocab
	                                  )

	test_set = MultiWOZDataset('./dataset/multiwoz/data/test_dials.json', 
	                                   './dataset/multiwoz/gating_dict.json', 
	                                   './dataset/multiwoz/data/multi-woz/MULTIWOZ2 2/ontology.json',
	                                   './dataset/multiwoz/domain_dict.json',
	                                   dataset_type = 'test',
	                                   training = True,
	                                   vocab = train_set.vocab
	                                  )

	print(train_set[0])