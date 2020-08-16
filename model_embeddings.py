#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway
import vocab

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        self.vocab = vocab
        n_filters = embed_size
        max_word_length = 21
        kernel_size = 5
        p_drop = 0.3

        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), embed_size, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(p_drop)
        self.cnn = CNN(embed_size, max_word_length, n_filters, kernel_size=kernel_size)
        self.highway = Highway(n_filters)

        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        sents_embeds = self.embeddings(input_tensor)
        sentence_length, batch_size, max_word_length, embed_size = list(sents_embeds.size())
        embeds = sents_embeds.permute(0, 1, 3, 2) # (s_length, b, embed_size, max_word_len)
        embeds = sents_embeds.reshape(sentence_length * batch_size, embed_size, max_word_length)
        # embeds now can be fed into cnn (b*s_l, embed_size, max_word_len)
        cnn_out = self.cnn(embeds) # output shape : (b*s_l, n_filters)
        highway_out = self.highway(cnn_out) # output shape : (b*s_l, n_filters)

        x_word_emb = self.dropout(highway_out)
        x_word_emb = x_word_emb.reshape(sentence_length, batch_size, -1) # (s_l, b, n_filters)

        return x_word_emb

        ### END YOUR CODE

if __name__ == "__main__":
    #embed = ModelEmbeddings(100, vocab)
    pass

