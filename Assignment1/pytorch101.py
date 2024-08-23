import torch
from torch import Tensor
from typing import List, Tuple

def create_sample_tensor() -> Tensor:
    x = torch.zeros((3, 2))
    x[0, 1] = 10
    x[1, 0] = 100

    return x

def mutate_tensor(
        x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:

    for idx, val in zip(indices, values):
        x[idx] = val

    return x

def count_tensor_elements(x: Tensor) -> int:
    num_elements = 1
    for i in x.shape:
        num_elements *= i

    return num_elements

def create_tensor_of_pi(M: int, N: int) -> Tensor:
    x = torch.full((M, N), 3.14)

    return x

def multiples_of_ten(start: int, stop: int) -> Tensor:
    assert start <= stop

    flag = False
    list = []
    for i in range(start, stop):
        if i % 10 == 0:
            list.append(i)
            flag = True

    if flag:
        x = torch.Tensor(list).to(dtype=torch.float64)
    else:
        x = torch.empty(0, dtype=torch.float64)

    return x


def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5

    last_row = x[-1, :]
    third_col = x[:, 2:3]
    first_two_rows_three_cols = x[0:2, 0:3]
    even_rows_odd_cols = x[0::2, 1::2]

    out = (
        last_row,
        third_col,
        first_two_rows_three_cols,
        even_rows_odd_cols
    )

    return out


def slice_assignment_practice(x: Tensor) -> Tensor:
    x[0:2, 0:1] = 0
    x[0:2, 1:2] = 1
    x[0:2, 2:6] = 2
    x[2:4, 0:4:2] = 3
    x[2:4, 1:4:2] = 4
    x[2:4, 4:6] = 5

    return x


def shuffle_cols(x: Tensor) -> Tensor:
    y = x[:, [0, 0, 2, 1]]

    return y

def reverse_rows(x: Tensor) -> Tensor:
    H, W = x.shape
    list_reverse = list(range(H))[::-1]
    y = x[list_reverse, :]

    return y

def take_one_elem_per_col(x: Tensor) -> Tensor:
    y = x[[1, 0, 3], [0, 1, 2]]

    return y