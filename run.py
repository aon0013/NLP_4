#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os

from parser_model import ParserModel
from parser_transitions import minibatch_parse
from utils.parser_utils import load_and_preprocess_data, AverageMeter, minibatches


def train_for_epoch(model, train_data, optimizer, loss_func, batch_size):
    """Train the neural network for a single epoch.

    Args:
        model: The neural network model
        train_data: Training data
        optimizer: The optimizer to use for training
        loss_func: Loss function
        batch_size: Size of minibatches

    Returns:
        Average loss over the epoch
    """
    model.train()  # Set model to training mode
    loss_meter = AverageMeter()

    # Create mini-batches from the training data
    train_minibatches = minibatches(train_data, batch_size)

    for batch in tqdm(train_minibatches):
        # Get inputs and labels
        inputs = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[2] for x in batch])

        # Forward pass
        logits, _ = model(inputs)

        # Compute loss
        loss = loss_func(logits, labels)
        loss_meter.update(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_meter.avg


def train(model, train_data, dev_data, optimizer, loss_func, batch_size, n_epochs, model_path):
    """Train the neural network.

    Args:
        model: The neural network model
        train_data: Training data
        dev_data: Development data
        optimizer: The optimizer to use for training
        loss_func: Loss function
        batch_size: Size of minibatches
        n_epochs: Number of training epochs
        model_path: Path to save the best model

    Returns:
        Best UAS on dev set
    """
    best_dev_UAS = 0.0

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")

        # Train for one epoch
        train_loss = train_for_epoch(model, train_data, optimizer, loss_func, batch_size)
        print(f"Average train loss: {train_loss:.4f}")

        # Evaluate on dev set
        dev_UAS = evaluate(model, dev_data, batch_size)
        print(f"Dev UAS: {dev_UAS:.4f}")

        # Save the model if it's the best so far
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with UAS: {best_dev_UAS:.4f}")

    return best_dev_UAS


def evaluate(model, dev_data, batch_size):
    """Evaluate the model on the dev set.

    Args:
        model: The neural network model
        dev_data: Development data
        batch_size: Size of minibatches

    Returns:
        UAS: Unlabeled Attachment Score
    """
    model.eval()  # Set model to evaluation mode

    # Extract sentences from dev data
    sentences = [x[0] for x in dev_data]
    gold_deps = [x[2] for x in dev_data]

    # Create a wrapper class for the model to use in minibatch_parse
    class ModelWrapper:
        def predict(self, partial_parses):
            inputs = []
            for p in partial_parses:
                # Extract features from the partial parse
                # This assumes there's a function extract_features in utils
                inputs.append(extract_features(p))

            inputs = torch.stack(inputs)
            with torch.no_grad():
                _, preds = model(inputs)

            # Convert predictions to transitions
            transitions = []
            for pred in preds:
                # Assuming transitions are ["S", "LA", "RA"] in that order
                transitions.append(["S", "LA", "RA"][pred.argmax().item()])

            return transitions

    # Get predicted dependencies
    model_wrapper = ModelWrapper()
    predicted_deps = minibatch_parse(sentences, model_wrapper, batch_size)

    # Calculate UAS (Unlabeled Attachment Score)
    correct = 0
    total = 0
    for i in range(len(gold_deps)):
        for h, d in predicted_deps[i]:
            if (h, d) in gold_deps[i]:
                correct += 1
            total += 1

    UAS = correct / total if total > 0 else 0.0
    return UAS


def main(debug=False):
    """Main function.

    Args:
        debug: If True, use a subset of data for faster debugging
    """
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")

    # Set random seed for reproducibility
    torch.manual_seed(1234)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_data, dev_data, test_data, embeddings, _ = load_and_preprocess_data(debug)

    # Initialize the model
    model = ParserModel(embeddings)

    # Define optimizer, loss function, and training parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    batch_size = 1024 if not debug else 32
    n_epochs = 10 if not debug else 2
    model_path = "model.weights"

    print(f"Train data: {len(train_data)} examples")
    print(f"Dev data: {len(dev_data)} examples")
    print(f"Test data: {len(test_data)} examples")

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")

    # Train the model
    best_dev_UAS = train(model, train_data, dev_data, optimizer, loss_func,
                         batch_size, n_epochs, model_path)

    print(80 * "=")
    print("TESTING")
    print(80 * "=")

    # Load the best model
    model.load_state_dict(torch.load(model_path))

    # Evaluate on the test set
    test_UAS = evaluate(model, test_data, batch_size)

    print(f"Best dev UAS: {best_dev_UAS:.4f}")
    print(f"Test UAS: {test_UAS:.4f}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Use a small subset of data for debugging")
    args = parser.parse_args()
    main(args.debug)