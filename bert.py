import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import tensorflow
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from transformers import BertModel, BertTokenizer

from itertools import islice
import numpy as np

CUDA_ENABLED  = 0

def to_cuda(tensor):
	if CUDA_ENABLED:
		tensor = tensor.cuda()
	return tensor

def build_sentence_list(start_token, sentences):
	text = [start_token]
	for sentence in sentences:
		text += sentence + ['SEP']
	return text


class CustomDataset(Dataset):
	def __init__(self, filename, num_bunches = 100):
		self.num_bunches = num_bunches
		self.num_lines = 114180969
		self.bunch_width = self.num_lines // num_bunches
		self.filename = filename
		self.set_bunch(0)

	def set_bunch(self, bunch_idx):
		start = bunch_idx * self.bunch_width
		end = (bunch_idx + 1) * self.bunch_width
		with open(self.filename, encoding='iso-8859-1') as f:
			lines = [line[:-1] for line in islice(f, start, end)]
		self.X = lines
	
	def preprocess(self, text):
		return text

	def __len__(self):
		return len(self.X)
	def __getitem__(self, index):
		return self.X[index]


if 0:
	dataset = CustomDataset('sample100000.txt')

	#Wrap it around a dataloader
	dataloader = DataLoader(dataset, batch_size = 2, num_workers = 0)

class MaskLMDataset:
	def __init__(self, dataset, dataloader):
		self.dataset = dataset
		self.dataloader = dataloader
if 0:
	itr = 0
	for text, lengths_and_masks in dataloader:
		print(len(text))
		itr += 1
		if itr > 2:
			break

class PretrainedModel(nn.Module):
	def __init__(self):
		super(PretrainedModel, self).__init__()
		self.model = to_cuda(BertModel.from_pretrained(
			'bert-base-uncased',
			output_hidden_states=True,
			output_attentions=True
		))
		self.model.eval()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	def forward(self, text = None, tokenized_text = None, attention_mask = None):
		if text is not None:
			tokenized_text = to_cuda(torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)]))
		if attention_mask is None:
			attention_mask = to_cuda(torch.tensor([[1]*len(tokenized_text)]))
		all_hidden_states, all_attentions = self.model(tokenized_text, attention_mask = attention_mask)[-2:]
		return all_hidden_states
		
if 0:
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
		attention_mask = to_cuda(torch.tensor(pad_sequences(masks, padding='post')))
		tokenized_text = to_cuda(torch.tensor(pad_sequences([
			self.tokenizer.convert_tokens_to_ids(sentence) for sentence in sentences]).tolist()))
		print(tokenized_text.shape)
		pretrained_hidden = self.pretrained_model(
			tokenized_text = tokenized_text, attention_mask = attention_mask)
		
		return pretrained_hidden
		

mdl = Model()

pretrained_hidden = mdl(['hi there', 'how are you'])

print(len(pretrained_hidden))
print(pretrained_hidden[4].shape)

import matplotlib.pyplot as plt
plt.hist(pretrained_hidden[0].reshape(-1).detach().numpy(), bins = 100)
