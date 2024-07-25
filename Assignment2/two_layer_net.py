import torch
import statistics
from typing import Dict, List, Callable, Optional

def sample_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    num_train: int,
    batch_size: int
):
    random_samples = torch.randint(0, num_train, (batch_size, ))
    X_batch = X[random_samples]
    y_batch = y[random_samples]

    return X_batch, y_batch


def nn_forward_pass(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor
):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    num_input, dim = X.shape

    y1 = torch.mm(X, W1) + b1
    zeros = torch.zeros_like(y1)
    hidden = torch.maximum(y1, zeros)

    scores = torch.mm(hidden, W2) + b2

    return scores, hidden

def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):

    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    num_input, dim = X.shape

    scores, hidden = nn_forward_pass(params, X)

    # If targets are not given, return scores
    if y is None:
        return scores

    # Compute the loss
    # Shift scores to prevent numerical instability
    scores -= torch.max(scores, dim=1, keepdim=True).values


    correct_scores = scores[range(num_input), y]
    sum_exp = torch.sum(torch.exp(scores), dim=1)
    loss = -torch.log(torch.exp(correct_scores) / sum_exp)
    loss = loss.sum() / num_input + 0.5 * reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

    # Backward pass: compute gradients
    grads = {}

    # Gradient of scores
    grad_scores = torch.exp(scores) / sum_exp.reshape(-1, 1)
    grad_scores[torch.arange(num_input), y] -= 1
    grad_scores /= num_input

    # Gradient of second layer weights and biases
    grads["W2"] = hidden.t().mm(grad_scores) + reg * W2
    grads["b2"] = torch.sum(grad_scores, dim=0)

    # Gradient of first layer weights and biases
    grad_hidden = grad_scores.mm(W2.t())
    grad_hidden[hidden <= 0] = 0  # ReLU backpropagation
    grads["W1"] = X.t().mm(grad_hidden) + reg * W1
    grads["b1"] = torch.sum(grad_hidden, dim=0)


    return loss, grads
