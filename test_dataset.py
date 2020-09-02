from dataset import CustomDataset
from torch.utils.data import DataLoader

def test_exists():
	assert CustomDataset, 'Custom Dataset exists' 

def test_can_import_sample_file():
	sample_dataset = CustomDataset('sample10000.txt')
	assert sample_dataset, 'can create an instance with '

def test_can_make_dataloader():
	sample_dataset = CustomDataset('sample10000.txt')
	dataloader = DataLoader(sample_dataset, batch_size = 2, num_workers = 0)
	assert dataloader, 'can make a dataloader model from sample dataset'

def test_first_lengths_in_data():
	sample_dataset = CustomDataset('sample10000.txt')
	dataloader = DataLoader(sample_dataset, batch_size = 2, num_workers = 0)

	itr = 0
	for text, lengths_and_masks in dataloader:
		print(text)		
		if itr == 0:
			assert text == 'Anarchism '
		if itr == 1: 
			assert text == 'It calls for the abolition of the state which it holds to be undesirable, unnecessary, and harmful.'
			break
		itr = itr + 1
