#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, embed_char, max_word_length, n_filters, kernel_size=5):
        """
        @param embed_char : input size, embedding dim
               max_word_length : length of word, dim that is convolved (time)
               n_filters : output size, number of filters f
               kernel_size : number of kernels (5)
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(embed_char, n_filters, kernel_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x : torch.Tensor):
        """
        @param input tensor x (batch, embed_char, max_word_length)
        output : tensor (batch, embed_char)
        """
        conv = self.conv(x)
        conv = self.relu(conv)
        conv_out = self.maxpool(conv) # leftover 1 dim
        conv_out = conv_out.squeeze(2) # remove leftover dim
        return conv_out


if __name__ == '__main__':
    net = CNN(50, 10, 5)
    inp = torch.ones([100, 50, 10])
    out = net(inp)
    print(out.shape)

### END YOUR CODE

