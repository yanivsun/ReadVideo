import torch.nn as nn


class VideoSummarizer(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, num_layers=2):
        super(VideoSummarizer, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 256)

    def forward(self, features):
        _, (h_n, _) = self.lstm(features)
        return self.fc(h_n[-1])  # Take the last hidden state as video summary embedding
