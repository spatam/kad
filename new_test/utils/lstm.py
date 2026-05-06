import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size=128, layers=2, hidden_size=128, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout # Applies between layer 1 and 2
        )
        
        # Normalizes the features for each time step. 
        # Size is hidden_size * 2 because it's bidirectional.
        self.ln = nn.LayerNorm(hidden_size * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # Permute to: [Batch, Time (49), Mels (128)]
        x = x.permute(0, 2, 1)
        
        # LSTM output shape: [Batch, 49, hidden_size * 2]
        out, _ = self.lstm(x)
        
        # Apply LayerNorm across the feature dimension
        out = self.ln(out)
        
        pooled_out, _ = torch.max(out, dim=1)
        
        logits = self.classifier(pooled_out)
        return logits
