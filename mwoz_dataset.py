import os
import torch
import torch.utils.data as data
import json
import random
import numpy as np
from dataset.multiwoz.fix_label import fix_general_label_error
from collections import Counter
from collections import OrderedDict
from tqdm import tqdm
from tools.vocab import Vocab


class MultiWOZDataset(data.Dataset):
	def __init__(self, 
				 dials_path, 
				 gating_dict_path, 
				 ontology_path,
				 domain_dict_path, 
				 dataset_type,
				 training, 
				 vocab = None,
				 **kwargs):
		'''
		Input:
		- dials_path (str): path to dials jsons (train_dials.json, dev_dials.json, test_dials.json)
		- gating_dict_path (str): path to gating_dict.json
		- vocab (class Vocab)
		'''
		self.data_ratio = kwargs.get('data_ratio', 1.0)
		self.exp_domains = kwargs.get('exp_domains', ["hotel", "train", "restaurant", "attraction", "taxi"])
		self.only_domains = kwargs.get('only_domains', None)
		self.exclude_domains = kwargs.get('exclude_domains', None)
		pretrained_embed_path = kwargs.get('pretrained_embed_path', None)
		self.dataset_type = dataset_type
		with open(gating_dict_path, 'r') as f:
			self.gating_dict = json.load(f)  # dict
		self.slots = self.get_filtered_exp_slots(ontology_path)
		self.domain_counter = None
		self.data = None
		self.max_seq_len = 0
		assert not (self.only_domains != None and self.exclude_domains != None), "Either 'only_domains' or 'exclude_domains' could be specified, not both."

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
			# if pretrained_embed_path is provided, corpus will be ignored.
			self.vocab = Vocab(pretrained_embed_path, corpus)
		else:
			self.vocab = vocab

	def __getitem__(self, idx):
		data = self.preprocess_data(self.data[idx])
		return data

	def __len__(self):
		return len(self.data)


	def get_filtered_exp_slots(self, ontology_path):
		with open(ontology_path, 'r') as f:
			ontology = json.load(f)
		ontology_domains = {k: v for k, v in ontology.items() if k.split('-')[0] in self.exp_domains}
		all_slots = [k.replace(' ','').lower() if ('book' not in k) else k.lower() for k in ontology_domains.keys()]
	    # Filter slots w.r.t. exclude_domains/only_domains for few/zero-shot task (allow multiple few/zero-shot domains)
		if self.only_domains != None:  # For few-shot finetuning
			all_slots = [slot for slot in all_slots if slot.split('-')[0] in self.only_domains]
		if self.exclude_domains != None:  # For zero-shot testing
			if self.dataset_type != 'test':  # Train & dev sets
				all_slots = [slot for slot in all_slots if slot.split('-')[0] not in self.exclude_domains]
			else:  # Test set
				all_slots = [slot for slot in all_slots if slot.split('-')[0] in self.exclude_domains]

		return all_slots

	def _filter_and_organize_data(self, raw_data):
		organized_data = []
		self.domain_counter = {}
		for dial_dict in tqdm(raw_data, desc='Reading data: '):
			ID = dial_dict['dialogue_idx']
			domains = dial_dict['domains']
			dialog_history = ''
			for domain in dial_dict['domains']:
				if domain in self.exp_domains:
					self.domain_counter[domain] = self.domain_counter.get(domain, 0) + 1

			# Skip those dial_dicts that do not have any only_domains at all
			if self.only_domains != None and not set(self.only_domains).intersection(set(dial_dict['domains'])):
				continue

			# Skip test-set dial dicts that do not have any exclude_domains at all & non-test-set dial dicts that have EXACT exclude_domains
			if (self.exclude_domains != None and self.dataset_type == 'test' and not set(self.exclude_domains).intersection(set(dial_dict['domains']))) or \
			   (self.exclude_domains != None and self.dataset_type != 'test' and set(self.exclude_domains) == set(dial_dict['domains'])):
			   continue

			for ti, turn in enumerate(dial_dict['dialogue']):
				turn_id = turn['turn_idx']
				turn_domain = turn['domain']
				turn_uttr = turn['system_transcript'] + ' [SEP] ' + turn['transcript']
				dialog_history += (turn_uttr + ' [SEP] ')
				turn_belief_dict = fix_general_label_error(turn['belief_state'], False, self.slots)

				# Filter turn dialogues w.r.t. exclude_domains/only_domains for few/zero-shot task (allow multiple few/zero-shot domains)
				if self.only_domains != None:  # For few-shot finetuning
					turn_belief_dict = OrderedDict([(slot, val) for slot, val in turn_belief_dict.items() if slot.split('-')[0] in self.only_domains])

				if self.exclude_domains != None:  # For zero-shot testing
					if self.dataset_type != 'test':  # Train & dev sets
						turn_belief_dict = OrderedDict([(slot, val) for slot, val in turn_belief_dict.items() if slot.split('-')[0] not in self.exclude_domains])
					else:  # Test set
						turn_belief_dict = OrderedDict([(slot, val) for slot, val in turn_belief_dict.items() if slot.split('-')[0] in self.exclude_domains])

				turn_belief_list = ['-'.join(map(str, slot_val_pair)) for slot_val_pair in turn_belief_dict.items()]

				generate_y, gating_label = [], []
				for slot in self.slots:
					val = turn_belief_dict.get(slot, 'none')
					generate_y.append(val)
					gating_label.append(val if val in self.gating_dict else 'ptr')  # val == 'ptr' if not 'none' or 'dontcare'


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
		Input:
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

		Output:
		- preprocessed_data (dict): indexed data for training
		'''
		gating_label = [self.gating_dict[label] for label in single_turn_data['gating_label']]
		context = [self.vocab.words_to_index(self.vocab.tokenize(single_turn_data['dialog_history']))]
		context = torch.tensor(context, dtype=torch.float32)
		turn_domain = self.domain2id[single_turn_data['turn_domain']]

		generate_y = []
		for y in single_turn_data['generate_y']:
			y = y + ' [EOS]'
			y = self.vocab.words_to_index(self.vocab.tokenize(y))
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
		}

		return preprocessed_data

	def collate_fn(self, batch_data_list):
		'''
		Collate function for dataloader

		Input:
		- batch_data_list (list): list of __getitem__ outputs, each output is a dict (ref. preprocess_data)
		'''
		bs = len(batch_data_list)
		merged_batch_data = {}
		for key in batch_data_list[0].keys():
			merged_batch_data[key] = [data[key] for data in batch_data_list]

		# Pad the context tensors w.r.t. the longest one and merge
		batch_context_len = [data['context'].shape[-1] for data in batch_data_list]
		batch_context_max_len = max(batch_context_len)
		batch_context_tensor = self.vocab.word2id['[PAD]']*torch.ones((bs, batch_context_max_len), dtype=torch.int64)
		for bi, data in enumerate(batch_data_list):
			context_tensor = data['context']
			batch_context_tensor[bi, :context_tensor.shape[-1]] = context_tensor

		# Pack and pad the responses into tensor of shape [bs, n_slots, longest_len]
		batch_gen_y_len = [[len(slot) for slot in y] for y in merged_batch_data['generate_y']]
		batch_gen_y_max_len = max([max(gen_y_len) for gen_y_len in batch_gen_y_len])
		n_slots = len(data['generate_y'])
		batch_gen_y_tensor = self.vocab.word2id['[PAD]']*torch.ones((bs, n_slots, batch_gen_y_max_len), dtype=torch.int64)
		for bi, data in enumerate(batch_data_list):
			for slot_i, y in enumerate(data['generate_y']):
				batch_gen_y_tensor[bi, slot_i, :len(y)] = torch.tensor(y)

		merged_batch_data['gating_label'] = torch.tensor(merged_batch_data['gating_label'])
		merged_batch_data['turn_domain'] = torch.tensor(merged_batch_data['turn_domain'])
		merged_batch_data['context'] = batch_context_tensor
		merged_batch_data['generate_y'] = batch_gen_y_tensor
		merged_batch_data['context_len'] = batch_context_len
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