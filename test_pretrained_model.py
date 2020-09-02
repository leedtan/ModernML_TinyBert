
from pretrained_model import PretrainedModel		

def test_exists():
	assert PretrainedModel

def test_can_create_a_model():
	model = PretrainedModel()
	assert model

def test_hidden_states():
	model = PretrainedModel()

	hidden_states = model("Here is some text to encode")

	assert len(hidden_states) == 13, '13 hidden states, just pinned. really should test > 0'
