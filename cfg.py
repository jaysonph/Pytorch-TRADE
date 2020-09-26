# Training Setting
training = {

	'filepaths': {
		'train_dials_path': './data/multiwoz/data/train_dials.json',
		'dev_dials_path': './data/multiwoz/data/dev_dials.json',
		'test_dials_path': './data/multiwoz/data/test_dials.json',
		'gating_dict_path': './tools/gating_dict.json',
		'ontology_path': './data/multiwoz/data/multi-woz/MULTIWOZ2 2/ontology.json',
		'domain_dict_path': './data/multiwoz/domain_dict.json',
		'pretrained_word_embed_path': None,  # If None, will create from dataset and saved to trained_vocab/word_embeddings.json
		'ckpt_save_dir': './checkpoints',  # directory to save the checkpoints
		'training_ckpt': None  # path to load training checkpoint to resume training
	},

	'settings': {
		'use_cuda': True,  # Use cuda if available
		'train_data_ratio': 1.,  # Portion of data to use for train set, between [0., 1.]
		'train_only_domains': ["hotel", "train", "restaurant", "attraction", "taxi"], # Only these domains will be used for train set
		'train_except_domains': None,  # All domains except these will be used for train set
		'test_only_domains': ["hotel", "train", "restaurant", "attraction", "taxi"], # Only these domains will be used for test set
		'test_except_domains': None,  # All domains except these will be used for test set
		'val_interval': 1  # in terms of epoch
	},

	'hyperparams': {
		'n_epochs': 500,
		'batch_size': 32,
		'learning_rate': 0.001,
		'betas': (0.9, 0.999),  # params for optimizer
		'alpha': 1.,  # weight for gate loss function
		'beta': 1.,  # weight for slot-value generation loss function
		'context_mask_prob': 0.5,  # probability of applying masking on context for replacing unk token
		'context_mask_ratio': 0.02,  # Ratio of tokens to mask as unk
		'domain_mask_prob': 0.1,  # probability of applying masking on domain for training unk token
		'slot_mask_prob': 0.1,  # probability of applying masking on slots for training unk token
		'teacher_forcing_rate': 0.7
	}

}


model = {

	'TRADE': {
		'use_pretrained_word_embed': True,
		'train_word_embed': True,
	},

	'decoder_gru': {
		'n_layers': 1,
		'hidden_dim': 400,
		'dropout': 0.,  # only effective when n_layers > 1
		'max_decode_len': 10,  # max-length for auto-regression steps
		
	},

	'encoder_gru': {
		'input_dim': 400,
		'hidden_dim': 400,
		'n_layers': 2,
		'dropout': 0.1,
		'bidirectional': True,

	},

	'slot_gate': {
		'n_layers': 2   # total number of layers including final output layer
	}
}
