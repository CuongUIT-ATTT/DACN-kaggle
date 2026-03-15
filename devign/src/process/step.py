import torch
from ..utils.objects import stats
import numpy as np


def softmax_accuracy(probs, all_labels):
    acc = (torch.argmax(probs) == all_labels).sum()
    acc = torch.div(acc, len(all_labels) + 0.0)
    return acc

def binary_accuracy(probs, all_labels):
    """
    Calculates the accuracy for binary classification given sigmoid probabilities and true labels.

    Args:
    - probs (torch.Tensor): Model's output probabilities, shape (n_samples,).
    - all_labels (torch.Tensor): True labels, shape (n_samples,).

    Returns:
    - float: The accuracy as a percentage.
    """
    # Round each probability to get binary predictions (0 or 1)
    predicted_classes = torch.round(probs)
    
    # Calculate accuracy
    correct_predictions = (predicted_classes == all_labels).sum()
    accuracy = torch.div(correct_predictions, len(all_labels) + 0.0)
    
    return accuracy

class Step:
    # Performs a step on the loader and returns the result
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer

    def __call__(self, i, x, y):
        out = self.model(x)
        loss = self.criterion(out, y.float())
        # acc = softmax_accuracy(out, y.float())
        acc = binary_accuracy(out, y.float())

        if self.model.training:
            # calculates the gradient
            loss.backward()
            # and performs a parameter update based on it
            self.optimizer.step()
            # clears old gradients from the last step
            self.optimizer.zero_grad()

        # print(f"\tBatch: {i}; Loss: {round(loss.item(), 4)}", end="")
        return stats.Stat(out.tolist(), loss.item(), acc.item(), y.tolist())

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
