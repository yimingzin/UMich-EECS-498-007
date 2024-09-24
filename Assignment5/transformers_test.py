import os

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

import torch

from torch import nn
import torch.nn.functional as F

from eecs598.utils import (
    reset_seed,
    tensor_to_image,
    attention_visualizer,
)
from eecs598.grad import rel_error, compute_numeric_gradient
from transformers import *
import matplotlib.pyplot as plt
import time
from IPython.display import Image


# for plotting
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

to_float = torch.float
to_long = torch.long

if torch.cuda.is_available():
    print("Good to go!")
    DEVICE = torch.device("cuda")
else:
    print("Please set GPU via Edit -> Notebook Settings.")
    DEVICE = torch.device("cpu")

from a5_helper import get_toy_data

# load the data using helper function
data = get_toy_data(os.path.join('./', "two_digit_op.json"))

# Create vocab
SPECIAL_TOKENS = ["POSITIVE", "NEGATIVE", "add", "subtract", "BOS", "EOS"]
vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] + SPECIAL_TOKENS

# ---------------------------------------------------------------------------------------------------------
#
# # BOS: Beginning of Sequence EOS: End of Sequence
# num_examples = 4
# for q, a in zip(
#     data["inp_expression"][:num_examples],
#     data["out_expression"][:num_examples]
#     ):
#   print("Expression: " + q + " Output: " + a)

# ---------------------------------------------------------------------------------------------------------
#
# from transformers import generate_token_dict
#
# convert_str_to_tokens = generate_token_dict(vocab)
#
# try:
#     assert convert_str_to_tokens["0"] == 0
# except:
#     print("The first element does not map to 0. Please check the implementation")
#
# try:
#     assert convert_str_to_tokens["EOS"] == 15
# except:
#     print("The last element does not map to 2004. Please check the implementation")
#
# print("Dictionary created successfully!")
# ---------------------------------------------------------------------------------------------------------
# from transformers import prepocess_input_sequence
#
# convert_str_to_tokens = generate_token_dict(vocab)
#
# ex1_in = "BOS POSITIVE 0333 add POSITIVE 0696 EOS"
# ex2_in = "BOS POSITIVE 0673 add POSITIVE 0675 EOS"
# ex3_in = "BOS NEGATIVE 0286 subtract NEGATIVE 0044 EOS"
# ex4_in = "BOS NEGATIVE 0420 add POSITIVE 0342 EOS"
#
# ex1_out = "BOS POSITIVE 1029 EOS"
# ex2_out = "BOS POSITIVE 1348 EOS"
# ex3_out = "BOS NEGATIVE 0242 EOS"
# ex4_out = "BOS NEGATIVE 0078 EOS"
#
# ex1_inp_preprocessed = torch.tensor(
#     prepocess_input_sequence(ex1_in, convert_str_to_tokens, SPECIAL_TOKENS)
# )
# ex2_inp_preprocessed = torch.tensor(
#     prepocess_input_sequence(ex2_in, convert_str_to_tokens, SPECIAL_TOKENS)
# )
# ex3_inp_preprocessed = torch.tensor(
#     prepocess_input_sequence(ex3_in, convert_str_to_tokens, SPECIAL_TOKENS)
# )
# ex4_inp_preprocessed = torch.tensor(
#     prepocess_input_sequence(ex4_in, convert_str_to_tokens, SPECIAL_TOKENS)
# )
#
# ex1_processed_expected = torch.tensor([14, 10, 0, 3, 3, 3, 12, 10, 0, 6, 9, 6, 15])
# ex2_processed_expected = torch.tensor([14, 10, 0, 6, 7, 3, 12, 10, 0, 6, 7, 5, 15])
# ex3_processed_expected = torch.tensor([14, 11, 0, 2, 8, 6, 13, 11, 0, 0, 4, 4, 15])
# ex4_processed_expected = torch.tensor([14, 11, 0, 4, 2, 0, 12, 10, 0, 3, 4, 2, 15])
#
# ex1_out = torch.tensor(
#     prepocess_input_sequence(ex1_out, convert_str_to_tokens, SPECIAL_TOKENS)
# )
# ex2_out = torch.tensor(
#     prepocess_input_sequence(ex2_out, convert_str_to_tokens, SPECIAL_TOKENS)
# )
# ex3_out = torch.tensor(
#     prepocess_input_sequence(ex3_out, convert_str_to_tokens, SPECIAL_TOKENS)
# )
# ex4_out = torch.tensor(
#     prepocess_input_sequence(ex4_out, convert_str_to_tokens, SPECIAL_TOKENS)
# )
#
# ex1_out_expected = torch.tensor([14, 10, 1, 0, 2, 9, 15])
# ex2_out_expected = torch.tensor([14, 10, 1, 3, 4, 8, 15])
# ex3_out_expected = torch.tensor([14, 11, 0, 2, 4, 2, 15])
# ex4_out_expected = torch.tensor([14, 11, 0, 0, 7, 8, 15])
#
# print(
#     "preprocess input token error 1: ",
#     rel_error(ex1_processed_expected, ex1_inp_preprocessed),
# )
# print(
#     "preprocess input token error 2: ",
#     rel_error(ex2_processed_expected, ex2_inp_preprocessed),
# )
# print(
#     "preprocess input token error 3: ",
#     rel_error(ex3_processed_expected, ex3_inp_preprocessed),
# )
# print(
#     "preprocess input token error 4: ",
#     rel_error(ex4_processed_expected, ex4_inp_preprocessed),
# )
# print("\n")
# print("preprocess output token error 1: ", rel_error(ex1_out_expected, ex1_out))
# print("preprocess output token error 2: ", rel_error(ex2_out_expected, ex2_out))
# print("preprocess output token error 3: ", rel_error(ex3_out_expected, ex3_out))
# print("preprocess output token error 4: ", rel_error(ex4_out_expected, ex4_out))
# ---------------------------------------------------------------------------------------------------------
# N = 2  # Number of sentences
# K = 5  # Number of words in a sentence
# M = 4  # feature dimension of each word embedding
#
# query = torch.linspace(-0.4, 0.6, steps=K * M).reshape(K, M)  # **to_double_cuda
# key = torch.linspace(-0.8, 0.5, steps=K * M).reshape(K, M)  # **to_double_cuda
# value = torch.linspace(-0.3, 0.8, steps=K * M).reshape(K, M)  # *to_double_cuda
#
# y = scaled_dot_product_two_loop_single(query, key, value)
# y_expected = torch.tensor(
#     [
#         [0.08283, 0.14073, 0.19862, 0.25652],
#         [0.13518, 0.19308, 0.25097, 0.30887],
#         [0.18848, 0.24637, 0.30427, 0.36216],
#         [0.24091, 0.29881, 0.35670, 0.41460],
#         [0.29081, 0.34871, 0.40660, 0.46450],
#     ]
# ).to(torch.float32)
# print("sacled_dot_product_two_loop_single error: ", rel_error(y_expected, y))
# ---------------------------------------------------------------------------------------------------------
# N = 2  # Number of sentences
# K = 5  # Number of words in a sentence
# M = 4  # feature dimension of each word embedding
#
# query = torch.linspace(-0.4, 0.6, steps=N * K * M).reshape(N, K, M)  # **to_double_cuda
# key = torch.linspace(-0.8, 0.5, steps=N * K * M).reshape(N, K, M)  # **to_double_cuda
# value = torch.linspace(-0.3, 0.8, steps=N * K * M).reshape(N, K, M)  # *to_double_cuda
#
# y = scaled_dot_product_two_loop_batch(query, key, value)
# y_expected = torch.tensor(
#     [
#         [
#             [-0.09603, -0.06782, -0.03962, -0.01141],
#             [-0.08991, -0.06170, -0.03350, -0.00529],
#             [-0.08376, -0.05556, -0.02735, 0.00085],
#             [-0.07760, -0.04939, -0.02119, 0.00702],
#             [-0.07143, -0.04322, -0.01502, 0.01319],
#         ],
#         [
#             [0.49884, 0.52705, 0.55525, 0.58346],
#             [0.50499, 0.53319, 0.56140, 0.58960],
#             [0.51111, 0.53931, 0.56752, 0.59572],
#             [0.51718, 0.54539, 0.57359, 0.60180],
#             [0.52321, 0.55141, 0.57962, 0.60782],
#         ],
#     ]
# ).to(torch.float32)
# print("scaled_dot_product_two_loop_batch error: ", rel_error(y_expected, y))
# ---------------------------------------------------------------------------------------------------------
