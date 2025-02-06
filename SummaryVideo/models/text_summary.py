import torch.nn as nn
from transformers import BertModel


class TextSummarizer(nn.Module):
    def __init__(self):
        super(TextSummarizer, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

    def forward(self, summary_embedding):
        text_embedding = self.bert.embeddings.word_embeddings(summary_embedding)
        return self.fc(text_embedding)
