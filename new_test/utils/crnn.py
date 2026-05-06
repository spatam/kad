import torch
import torch.nn as nn
from utils.Conv import ResidualBlock


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes,
        n_mels,
        num_filters=128,
        num_blocks=3,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.2,
    ):
        super().__init__()

        kernel_size = 3
        layers = []
        for i in range(num_blocks):
            dilation_size = 2**i

            # First block takes n_mels as input channels, subsequent blocks take num_filters
            in_ch = n_mels if i == 0 else num_filters
            out_ch = num_filters

            layers.append(
                ResidualBlock(in_ch, out_ch, kernel_size, dilation_size, dropout)
            )

        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.ln = nn.LayerNorm(lstm_hidden * 2)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
            if x.dim() == 4:
                x = x.squeeze(1)
            x = self.cnn(x)
            x = x.permute(0, 2, 1)
            out, (hn, cn) = self.lstm(x)

            pooled_out, _= torch.max(out, dim=1)
            
            norm_out = self.ln(pooled_out) 
            
            logits = self.fc(norm_out)

            return logits
