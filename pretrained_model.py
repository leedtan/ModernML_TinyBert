from utils import *
class PretrainedModel(nn.Module):  # We could delete this and just use to_cuda of Bert whatever. We don't need the class. Just use BERT model
    def __init__(self):
        super(PretrainedModel, self).__init__()
        self.model = to_cuda(BertForMaskedLM.from_pretrained(
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
        loss, output, all_hidden_states, all_attentions = self.model(
            tokenized_text, attention_mask = attention_mask, labels = tokenized_text)
        return loss, output, all_hidden_states, all_attentions