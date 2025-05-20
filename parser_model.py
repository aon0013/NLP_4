#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class ParserModel(torch.nn.Module):
    """
    Feedforward neural network with an embedding layer and single hidden layer.
    This network predicts which transition should be applied to a given partial parse.
    """

    def __init__(self, embeddings, n_features=36,
                 hidden_size=200, n_classes=3, dropout_prob=0.5):
        """Initialize the parser model.

        Args:
            embeddings: Word embeddings
            n_features: Number of input features
            hidden_size: Number of hidden units
            n_classes: Number of output classes
            dropout_prob: Dropout probability
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = torch.nn.Parameter(torch.tensor(embeddings, dtype=torch.float32))

        # Initialize weights and bias for the embed_to_hidden layer (W, b1)
        self.embed_to_hidden_weight = torch.nn.Parameter(
            torch.zeros(n_features * self.embed_size, hidden_size)
        )
        self.embed_to_hidden_bias = torch.nn.Parameter(
            torch.zeros(hidden_size)
        )

        # Initialize weights and bias for the hidden_to_logits layer (U, b2)
        self.hidden_to_logits_weight = torch.nn.Parameter(
            torch.zeros(hidden_size, n_classes)
        )
        self.hidden_to_logits_bias = torch.nn.Parameter(
            torch.zeros(n_classes)
        )

        # Initialize weights using Xavier initialization
        torch.nn.init.xavier_uniform_(self.embed_to_hidden_weight)
        torch.nn.init.xavier_uniform_(self.hidden_to_logits_weight)
        torch.nn.init.uniform_(self.embed_to_hidden_bias, -0.1, 0.1)
        torch.nn.init.uniform_(self.hidden_to_logits_bias, -0.1, 0.1)

    def embedding_lookup(self, w):
        """Look up embeddings for a batch of input words.

        Args:
            w: Input tensor of word indices of shape (batch_size, n_features)

        Returns:
            x: Tensor of embeddings with shape (batch_size, n_features * embed_size)
        """
        # Get the embeddings for each word index
        batch_size = w.shape[0]
        embeds = self.embeddings[w]  # Shape: (batch_size, n_features, embed_size)

        # Reshape to (batch_size, n_features * embed_size)
        x = embeds.reshape(batch_size, -1)

        return x

    def forward(self, w):
        """Compute logits for predicting transitions.

        Args:
            w: Input tensor of word indices of shape (batch_size, n_features)

        Returns:
            logits: Tensor of logits of shape (batch_size, n_classes)
            pred: Tensor of softmax probabilities of shape (batch_size, n_classes)
        """
        # Get embeddings for the words
        x = self.embedding_lookup(w)

        # Apply dropout to embeddings during training
        if self.training:
            x = F.dropout(x, p=self.dropout_prob)

        # Compute hidden layer: h = ReLU(xW + b1)
        h = F.relu(x @ self.embed_to_hidden_weight + self.embed_to_hidden_bias)

        # Apply dropout to hidden layer during training
        if self.training:
            h = F.dropout(h, p=self.dropout_prob)

        # Compute logits: l = hU + b2
        logits = h @ self.hidden_to_logits_weight + self.hidden_to_logits_bias

        # Compute predictions using softmax
        pred = F.softmax(logits, dim=1)

        return logits, pred