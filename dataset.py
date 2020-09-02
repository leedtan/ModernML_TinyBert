from torch.utils.data import Dataset
from itertools import islice



class CustomDataset(Dataset):
	'''
	import a custom dataset from a filename
	'''
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
