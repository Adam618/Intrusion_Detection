import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=2, padding=3),
            nn.BatchNorm1d(num_features=6),
            nn.ReLU(),
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=6),
            nn.ReLU(),
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=12),
            nn.ReLU(),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(num_features=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm_layer = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=False)
        
        #self.dropout = nn.Dropout(p=0.1)

        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
        x1 = self.encoder1(x)
        #print("Encoder 1 output shape:", x.shape)
        x2 = self.encoder2(x)
        #print("Encoder 2 output shape:", x.shape)
        x = torch.cat((x1, x2), dim=1)
        # x = x2
        # print("x.shape", x.shape)
        x, _ = self.lstm_layer(x)
        output = x[:, -1, :]
        feature = output.reshape(output.shape[0], -1)
        output = self.fc(output)

        return feature, output

#Best Epoch 49, Train Accuracy: 0.9970072952140538 , Train Loss: 0.011474510859374168 , Test Accuracy: 0.810903122782115 , Test Loss: 0.9772600117433655