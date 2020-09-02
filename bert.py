# global
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import tensorflow
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from transformers import BertModel, BertTokenizer

from itertools import islice
import numpy as np

# local
from dataset import CustomDataset


def build_sentence_list(start_token, sentences):
	text = [start_token]
	for sentence in sentences:
		text += sentence + ['SEP']
	return text


sample_dataset = CustomDataset('sample10000.txt')

dataloader = DataLoader(sample_dataset, batch_size = 2, num_workers = 0)

class MaskLMDataset:
	def __init__(self, dataset, dataloader):
		self.dataset = dataset
		self.dataloader = dataloader


class PretrainedModel(nn.Module):
	def __init__(self):
		super(PretrainedModel, self).__init__()
		self.model = BertModel.from_pretrained(
			'bert-base-uncased',
			output_hidden_states=True,
			output_attentions=True
		)
		self.model.eval()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	def forward(self, text = None, tokenized_text = None, attention_mask = None):
		if text is not None:
			tokenized_text = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])
		if attention_mask is None:
			attention_mask = torch.tensor([[1]*len(tokenized_text)])
		all_hidden_states, all_attentions = self.model(tokenized_text, attention_mask = attention_mask)[-2:]
		return all_hidden_states
		
if 1:
	model = PretrainedModel()
	hidden_states = model("Here is some text to encode")

	len(hidden_states)

class TinyBert(nn.Module):
	def __init__(self):
		super(TinyBert, self).__init__()
		
	def forward(self, text):
		return


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.pretrained_model = PretrainedModel()
		self.tinybert = TinyBert()
		self.tokenizer = self.pretrained_model.tokenizer
		self.y = []

	def forward(self, text):
		if isinstance(text[0], list):
			return self.forward_sentence(text)
		elif isinstance(text[0], str):
			return self.forward_maskLM(text)
		else:
			raise ValueError('wtf is this text?' + text + type(text[0]))
	def forward_maskLM(self, text):
		self.y = []
		sentences = [build_sentence_list(
			'CLS', [self.tokenizer.tokenize(line)]) for line in text]
		
		lengths = [len(sentence) - 2 for sentence in sentences]
		mask_idxes = [np.random.randint(0, length) for length in lengths]
		
		masks = [np.ones(length + 2) for length in lengths]
		for mask_idx, mask, sentence in zip(mask_idxes, masks, sentences):
			mask[mask_idx + 1] = 0
			self.y.append(sentence[mask_idx + 1])
			sentence[mask_idx + 1] = '[MASK]'
		attention_mask = torch.tensor(pad_sequences(masks, padding='post'))
		tokenized_text = torch.tensor(
			pad_sequences(
				[self.tokenizer.convert_tokens_to_ids(sentence) for sentence in sentences]
			).tolist()
		)
		print(tokenized_text.shape)
		pretrained_hidden = self.pretrained_model(
			tokenized_text = tokenized_text, attention_mask = attention_mask)
		
		return pretrained_hidden
		

if 1:
	mdl = Model()

	pretrained_hidden = mdl(['hi there', 'how are you'])

	print(len(pretrained_hidden))
	print(pretrained_hidden[4].shape)
