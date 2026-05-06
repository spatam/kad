import torch.nn as nn
from torch.utils.data import DataLoader
from utils.Conv import ResidualBlock



class TCN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        # Architecture of the Net
        num_blocks = 4      # 4 repeating Residual blocks
        num_filters = 128   # 128 Number of filters per Layer
        kernel_size = 3     # Kernel of size 3 (1x3 vector)
        dropout = 0.2       # Percentage of Neurons to turn off at each iteraton
        
        input_channels = input_shape[0]

        layers = []
        for i in range(num_blocks):
            dilation_size = 2 ** i # As stated in the paper, dilation is 2^i-1, since here we start at 0 we can write 2^i
            
            in_ch = input_channels if i == 0 else num_filters   # If we are in the first block, 
                                                                # the input channels are given by the shape, else by the number of filters 
                                                                # (which is the output each channel)
            out_ch = num_filters
            
            layers.append(
                ResidualBlock(in_ch, out_ch, kernel_size, dilation_size, dropout)
            )

        # We apply all the 4 blocks sequentially
        self.network = nn.Sequential(*layers)
        # Then with avg pool we normalze the sizes
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(num_filters, num_classes) # FC for classification

    def forward(self, x):

        x = x.squeeze(1)
        x = self.network(x)
        x = self.avg_pool(x) 
        x = x.squeeze(2)     
        x = self.fc(x)
        return x
