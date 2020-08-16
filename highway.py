#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, embed_size):
        """
        @param in_features : embedding size
        """
        super(Highway, self).__init__()
        self.linear1 = nn.Linear(embed_size, embed_size, bias=True)
        self.linear2 = nn.Linear(embed_size, embed_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        """
        @param x : tensor conv_out with shape (batch_size, embed)
        """
        x_proj = self.linear1(x)
        x_proj = self.relu(x_proj)
        x_gate = self.linear2(x)
        x_gate = self.sigmoid(x_gate)
        x_highway = x_gate * x_proj + (1 - x_gate) * x
        return x_highway

# sanity checks
if __name__ == "__main__":
    net = Highway(5, 5)
    test_tensor = torch.zeros([10, 5])
    out = net(test_tensor)
    print(net)
    print(out)


### END YOUR CODE

