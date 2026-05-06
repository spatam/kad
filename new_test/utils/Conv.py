import torch.nn as nn

class TemporalConv1D(nn.Module):
    """
    Implementation a Causal 1D Convolution.
    
    Standard Conv1d is "centered," meaning the kernel at time 't' sees data 
    from both t-1 and t+1. The t+1 represents future signals making the convolution invalid for real-time causal modeling.
    
    This modified Conv1d layer solves that by:
    1. Padding the left side (the past) of the input.
    2. Shifting the input so the convolution kernel ends exactly at time 't',
       seeing only t, t-1, t-2, etc.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        
        # Calculate the exact left-padding needed to preserve sequence length while keeping the operation causal.
        # Formula ensures the kernel reaches exactly to the oldest past input.
        self.pad_len = (kernel_size - 1) * dilation 
        
        self.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=0,  # Padding turned off. We will work on it manually, padding only the left side 
            dilation=dilation
        ))

    def forward(self, x):
        x_padded = nn.functional.pad(x, (self.pad_len, 0)) # Putting pad_len, 0, means that we pad only on the left side
        
        # Apply the convolution to the newly padded input.
        return self.conv(x_padded)


class ResidualBlock(nn.Module):
    '''
    Manual Implementation of Residual Block Found in 'A New Pipeline for Snooping Keystroke Based on Deep Learning Algorithm'

    The blocks are made of Causal layers where the main block is the TemporalConv1D defined above.
    After applying the convolution, we pass the results into ReLU and then apply dropout as stated in the paper.

    We reapeat this twice per block, then downsample through 1x1 convolution if necessary (when input and output channels differ) 
    and apply a final relu
    '''


    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        # First Causal Layer
        self.first_layer=nn.Sequential(
            TemporalConv1D(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Second Causal Layer
        self.second_layer = nn.Sequential(
            TemporalConv1D(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.final_out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.first_layer(x)
        out = self.second_layer(out)

        res = x if self.downsample is None else self.downsample(x)
        final = res + out
        return self.final_out(final)