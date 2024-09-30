import os
import timeit

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

import torch
import torch.utils.data
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
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

to_float = torch.float
to_long = torch.long


from a5_helper import get_toy_data

# load the data using helper function
data = get_toy_data(os.path.join('./', "two_digit_op.json"))

# Create vocab
SPECIAL_TOKENS = ["POSITIVE", "NEGATIVE", "add", "subtract", "BOS", "EOS"]
vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] + SPECIAL_TOKENS

convert_str_to_tokens = generate_token_dict(vocab)

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
# N = 2  # Number of sentences
# K = 5  # Number of words in a sentence
# M = 4  # feature dimension of each word embedding
#
# query = torch.linspace(-0.4, 0.6, steps=N * K * M).reshape(N, K, M)  # **to_double_cuda
# key = torch.linspace(-0.8, 0.5, steps=N * K * M).reshape(N, K, M)  # **to_double_cuda
# value = torch.linspace(-0.3, 0.8, steps=N * K * M).reshape(N, K, M)  # *to_double_cuda
#
#
# y, _ = scaled_dot_product_no_loop_batch(query, key, value)
#
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
#
# print("scaled_dot_product_no_loop_batch error: ", rel_error(y_expected, y))
# ---------------------------------------------------------------------------------------------------------
# N = 64
# K = 256  # defines the input sequence length
# M = emb_size = 2048
# dim_q = dim_k = 2048
# query = torch.linspace(-0.4, 0.6, steps=N * K * M).reshape(N, K, M)  # **to_double_cuda
# key = torch.linspace(-0.8, 0.5, steps=N * K * M).reshape(N, K, M)  # **to_double_cuda
# value = torch.linspace(-0.3, 0.8, steps=N * K * M).reshape(N, K, M)  # *to_double_cuda
#
# # 测试的函数
# def test_function():
#     y, weights_softmax = scaled_dot_product_no_loop_batch(query, key, value)
#
# # 使用 timeit 模块，重复执行 5 次，测量执行时间
# execution_time = timeit.timeit(test_function, number=5)
# print(f"Average Execution Time over 5 runs: {execution_time / 5} seconds")
#
# N = 64
# K = 512  # defines the input requence length
# M = emb_size = 2048
# dim_q = dim_k = 2048
# query = torch.linspace(-0.4, 0.6, steps=N * K * M).reshape(N, K, M)  # **to_double_cuda
# key = torch.linspace(-0.8, 0.5, steps=N * K * M).reshape(N, K, M)  # **to_double_cuda
# value = torch.linspace(-0.3, 0.8, steps=N * K * M).reshape(N, K, M)  # *to_double_cuda
#
# # 测试的函数
# def test_function():
#     y, weights_softmax = scaled_dot_product_no_loop_batch(query, key, value)
#
# # 使用 timeit 模块，重复执行 5 次，测量执行时间
# execution_time = timeit.timeit(test_function, number=5)
# print(f"Average Execution Time over 5 runs: {execution_time / 5} seconds")
# ---------------------------------------------------------------------------------------------------------
# reset_seed(0)
# N = 2
# K = 4
# M = emb_size = 4
# dim_q = dim_k = 4
# atten_single = SelfAttention(emb_size, dim_q, dim_k)
#
# for k, v in atten_single.named_parameters():
#     # print(k, v.shape) # uncomment this to see the weight shape
#     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).reshape(*v.shape))
#
# query = torch.linspace(-0.4, 0.6, steps=N * K * M, requires_grad=True).reshape(
#     N, K, M
# )  # **to_double_cuda
# key = torch.linspace(-0.8, 0.5, steps=N * K * M, requires_grad=True).reshape(
#     N, K, M
# )  # **to_double_cuda
# value = torch.linspace(-0.3, 0.8, steps=N * K * M, requires_grad=True).reshape(
#     N, K, M
# )  # *to_double_cuda
#
# query.retain_grad()
# key.retain_grad()
# value.retain_grad()
#
# y_expected = torch.tensor(
#     [
#         [
#             [-1.10382, -0.37219, 0.35944, 1.09108],
#             [-1.45792, -0.50067, 0.45658, 1.41384],
#             [-1.74349, -0.60428, 0.53493, 1.67414],
#             [-1.92584, -0.67044, 0.58495, 1.84035],
#         ],
#         [
#             [-4.59671, -1.63952, 1.31767, 4.27486],
#             [-4.65586, -1.66098, 1.33390, 4.32877],
#             [-4.69005, -1.67339, 1.34328, 4.35994],
#             [-4.71039, -1.68077, 1.34886, 4.37848],
#         ],
#     ]
# )
#
# dy_expected = torch.tensor(
#     [
#         [
#             [-0.09084, -0.08961, -0.08838, -0.08715],
#             [0.69305, 0.68366, 0.67426, 0.66487],
#             [-0.88989, -0.87783, -0.86576, -0.85370],
#             [0.25859, 0.25509, 0.25158, 0.24808],
#         ],
#         [
#             [-0.05360, -0.05287, -0.05214, -0.05142],
#             [0.11627, 0.11470, 0.11312, 0.11154],
#             [-0.01048, -0.01034, -0.01019, -0.01005],
#             [-0.03908, -0.03855, -0.03802, -0.03749],
#         ],
#     ]
# )
#
# y = atten_single(query, key, value)
# dy = torch.randn(*y.shape)  # , **to_double_cuda
#
# y.backward(dy)
# query_grad = query.grad
#
# print("SelfAttention error: ", rel_error(y_expected, y))
# print("SelfAttention error: ", rel_error(dy_expected, query_grad))
# ---------------------------------------------------------------------------------------------------------
# reset_seed(0)
# N = 2
# num_heads = 2
# K = 4
# M = inp_emb_size = 4
# out_emb_size = 8
# atten_multihead = MultiHeadAttention(num_heads, inp_emb_size, out_emb_size)
#
# for k, v in atten_multihead.named_parameters():
#     # print(k, v.shape) # uncomment this to see the weight shape
#     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).reshape(*v.shape))
#
# query = torch.linspace(-0.4, 0.6, steps=N * K * M, requires_grad=True).reshape(
#     N, K, M
# )  # **to_double_cuda
# key = torch.linspace(-0.8, 0.5, steps=N * K * M, requires_grad=True).reshape(
#     N, K, M
# )  # **to_double_cuda
# value = torch.linspace(-0.3, 0.8, steps=N * K * M, requires_grad=True).reshape(
#     N, K, M
# )  # *to_double_cuda
#
# query.retain_grad()
# key.retain_grad()
# value.retain_grad()
#
# y_expected = torch.tensor(
#     [
#         [
#             [-0.23104, 0.50132, 1.23367, 1.96603],
#             [0.68324, 1.17869, 1.67413, 2.16958],
#             [1.40236, 1.71147, 2.02058, 2.32969],
#             [1.77330, 1.98629, 2.19928, 2.41227],
#         ],
#         [
#             [6.74946, 5.67302, 4.59659, 3.52015],
#             [6.82813, 5.73131, 4.63449, 3.53767],
#             [6.86686, 5.76001, 4.65315, 3.54630],
#             [6.88665, 5.77466, 4.66268, 3.55070],
#         ],
#     ]
# )
# dy_expected = torch.tensor(
#     [[[ 0.56268,  0.55889,  0.55510,  0.55131],
#          [ 0.43286,  0.42994,  0.42702,  0.42411],
#          [ 2.29865,  2.28316,  2.26767,  2.25218],
#          [ 0.49172,  0.48841,  0.48509,  0.48178]],
#
#         [[ 0.25083,  0.24914,  0.24745,  0.24576],
#          [ 0.14949,  0.14849,  0.14748,  0.14647],
#          [-0.03105, -0.03084, -0.03063, -0.03043],
#          [-0.02082, -0.02068, -0.02054, -0.02040]]]
# )
#
# y = atten_multihead(query, key, value)
# dy = torch.randn(*y.shape)  # , **to_double_cuda
#
# y.backward(dy)
# query_grad = query.grad
# print("MultiHeadAttention error: ", rel_error(y_expected, y))
# print("MultiHeadAttention error: ", rel_error(dy_expected, query_grad))
# ---------------------------------------------------------------------------------------------------------
# reset_seed(0)
# N = 2
# K = 4
# norm = LayerNormalization(K)
# inp = torch.linspace(-0.4, 0.6, steps=N * K, requires_grad=True).reshape(N, K)
#
# inp.retain_grad()
# y = norm(inp)
#
# y_expected = torch.tensor(
#     [[-1.34164, -0.44721, 0.44721, 1.34164], [-1.34164, -0.44721, 0.44721, 1.34164]]
# )
#
# dy_expected = torch.tensor(
#     [[  5.70524,  -2.77289, -11.56993,   8.63758],
#         [  2.26242,  -4.44330,   2.09933,   0.08154]]
# )
#
# dy = torch.randn(*y.shape)
# y.backward(dy)
# inp_grad = inp.grad
#
# print("LayerNormalization error: ", rel_error(y_expected, y))
# print("LayerNormalization grad error: ", rel_error(dy_expected, inp_grad))
# ---------------------------------------------------------------------------------------------------------
# reset_seed(0)
# N = 2
# K = 4
# M = emb_size = 4
#
# ff_block = FeedForwardBlock(emb_size, 2 * emb_size)
#
# for k, v in ff_block.named_parameters():
#     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).reshape(*v.shape))
#
# inp = torch.linspace(-0.4, 0.6, steps=N * K, requires_grad=True).reshape(
#     N, K
# )
# inp.retain_grad()
# y = ff_block(inp)
#
# y_expected = torch.tensor(
#     [[-2.46161, -0.71662, 1.02838, 2.77337], [-7.56084, -1.69557, 4.16970, 10.03497]]
# )
#
# dy_expected = torch.tensor(
#     [[0.55105, 0.68884, 0.82662, 0.96441], [0.30734, 0.31821, 0.32908, 0.33996]]
# )
#
# dy = torch.randn(*y.shape)
# y.backward(dy)
# inp_grad = inp.grad
#
# print("FeedForwardBlock error: ", rel_error(y_expected, y))
# print("FeedForwardBlock error: ", rel_error(dy_expected, inp_grad))
# ---------------------------------------------------------------------------------------------------------
# reset_seed(0)
# N = 2
# num_heads = 2
# emb_dim = K = 4
# feedforward_dim = 8
# M = inp_emb_size = 4
# out_emb_size = 8
# dropout = 0.2
#
# enc_seq_inp = torch.linspace(-0.4, 0.6, steps=N * K * M, requires_grad=True).reshape(
#     N, K, M
# )  # **to_double_cuda
#
# enc_block = EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
#
# for k, v in enc_block.named_parameters():
#     # print(k, v.shape) # uncomment this to see the weight shape
#     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).reshape(*v.shape))
#
# encoder_out1_expected = torch.tensor(
#     [[[ 0.00000, -0.31357,  0.69126,  0.00000],
#          [ 0.42630, -0.25859,  0.72412,  3.87013],
#          [ 0.00000, -0.31357,  0.69126,  3.89884],
#          [ 0.47986, -0.30568,  0.69082,  3.90563]],
#
#         [[ 0.00000, -0.31641,  0.69000,  3.89921],
#          [ 0.47986, -0.30568,  0.69082,  3.90563],
#          [ 0.47986, -0.30568,  0.69082,  3.90563],
#          [ 0.51781, -0.30853,  0.71598,  3.85171]]]
# )
# encoder_out1 = enc_block(enc_seq_inp)
# print("EncoderBlock error 1: ", rel_error(encoder_out1, encoder_out1_expected))
#
#
# N = 2
# num_heads = 1
# emb_dim = K = 4
# feedforward_dim = 8
# M = inp_emb_size = 4
# out_emb_size = 8
# dropout = 0.2
#
# enc_seq_inp = torch.linspace(-0.4, 0.6, steps=N * K * M, requires_grad=True).reshape(
#     N, K, M
# )  # **to_double_cuda
#
# enc_block = EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
#
# for k, v in enc_block.named_parameters():
#     # print(k, v.shape) # uncomment this to see the weight shape
#     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).reshape(*v.shape))
#
# encoder_out2_expected = torch.tensor(
#     [[[ 0.42630, -0.00000,  0.72412,  3.87013],
#          [ 0.49614, -0.31357,  0.00000,  3.89884],
#          [ 0.47986, -0.30568,  0.69082,  0.00000],
#          [ 0.51654, -0.32455,  0.69035,  3.89216]],
#
#         [[ 0.47986, -0.30568,  0.69082,  0.00000],
#          [ 0.49614, -0.31357,  0.69126,  3.89884],
#          [ 0.00000, -0.30354,  0.76272,  3.75311],
#          [ 0.49614, -0.31357,  0.69126,  3.89884]]]
# )
# encoder_out2 = enc_block(enc_seq_inp)
# print("EncoderBlock error 2: ", rel_error(encoder_out2, encoder_out2_expected))
# ---------------------------------------------------------------------------------------------------------
# from transformers import get_subsequent_mask
#
# reset_seed(0)
# seq_len_enc = K = 4
# M = inp_emb_size = 3
#
# inp_sequence = torch.linspace(-0.4, 0.6, steps=K * M, requires_grad=True).reshape(
#     K, M
# )  # **to_double_cuda
#
# mask_expected = torch.tensor(
#     [
#         [[False, True, True], [False, False, True], [False, False, False]],
#         [[False, True, True], [False, False, True], [False, False, False]],
#         [[False, True, True], [False, False, True], [False, False, False]],
#         [[False, True, True], [False, False, True], [False, False, False]],
#     ]
# )
# mask_predicted = get_subsequent_mask(inp_sequence)
# print(
#     "get_subsequent_mask error: ", rel_error(mask_predicted.int(), mask_expected.int())
# )
#
# reset_seed(0)
# N = 4
# K = 3
# M = 3
#
# query = torch.linspace(-0.4, 0.6, steps=K * M * N, requires_grad=True).reshape(N, K, M)
# key = torch.linspace(-0.1, 0.2, steps=K * M * N, requires_grad=True).reshape(N, K, M)
# value = torch.linspace(0.4, 0.8, steps=K * M * N, requires_grad=True).reshape(N, K, M)
#
# y_expected = torch.tensor(
#     [
#         [
#             [0.40000, 0.41143, 0.42286],
#             [0.41703, 0.42846, 0.43989],
#             [0.43408, 0.44551, 0.45694],
#         ],
#         [
#             [0.50286, 0.51429, 0.52571],
#             [0.51999, 0.53142, 0.54285],
#             [0.53720, 0.54863, 0.56006],
#         ],
#         [
#             [0.60571, 0.61714, 0.62857],
#             [0.62294, 0.63437, 0.64580],
#             [0.64032, 0.65175, 0.66318],
#         ],
#         [
#             [0.70857, 0.72000, 0.73143],
#             [0.72590, 0.73733, 0.74876],
#             [0.74344, 0.75487, 0.76630],
#         ],
#     ]
# )
# y_predicted, _ = scaled_dot_product_no_loop_batch(query, key, value, mask_expected)
#
# print("scaled_dot_product_no_loop_batch error: ", rel_error(y_expected, y_predicted))
# ---------------------------------------------------------------------------------------------------------
# reset_seed(0)
# N = 2
# num_heads = 2
# seq_len_enc = K1 = 4
# seq_len_dec = K2 = 2
# feedforward_dim = 8
# M = emb_dim = 4
# out_emb_size = 8
# dropout = 0.2
#
# dec_inp = torch.linspace(-0.4, 0.6, steps=N * K1 * M, requires_grad=True).reshape(
#     N, K1, M
# )
# enc_out = torch.linspace(-0.4, 0.6, steps=N * K2 * M, requires_grad=True).reshape(
#     N, K2, M
# )
# dec_block = DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
#
# for k, v in dec_block.named_parameters():
#     # print(k, v.shape) # uncomment this to see the weight shape
#     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).reshape(*v.shape))
#
#
# dec_out_expected = torch.tensor(
#     [[[ 0.50623, -0.32496,  0.00000,  0.00000],
#          [ 0.00000, -0.31690,  0.76956,  3.72647],
#          [ 0.49014, -0.32809,  0.66595,  3.93773],
#          [ 0.00000, -0.00000,  0.68203,  3.90856]],
#
#         [[ 0.51042, -0.32787,  0.68093,  3.90848],
#          [ 0.00000, -0.31637,  0.72275,  3.83122],
#          [ 0.64868, -0.00000,  0.77715,  0.00000],
#          [ 0.00000, -0.33105,  0.66565,  3.93602]]]
# )
# dec_out1 = dec_block(dec_inp, enc_out)
# print("DecoderBlock error: ", rel_error(dec_out1, dec_out_expected))
#
# N = 2
# num_heads = 2
# seq_len_enc = K1 = 4
# seq_len_dec = K2 = 4
# feedforward_dim = 4
# M = emb_dim = 4
# out_emb_size = 8
# dropout = 0.2
#
# dec_inp = torch.linspace(-0.4, 0.6, steps=N * K1 * M, requires_grad=True).reshape(
#     N, K1, M
# )
# enc_out = torch.linspace(-0.4, 0.6, steps=N * K2 * M, requires_grad=True).reshape(
#     N, K2, M
# )
# dec_block = DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
#
# for k, v in dec_block.named_parameters():
#     # print(k, v.shape) # uncomment this to see the weight shape
#     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).reshape(*v.shape))
#
#
# dec_out_expected = torch.tensor(
#     [[[ 0.46707, -0.31916,  0.66218,  3.95182],
#          [ 0.00000, -0.31116,  0.66325,  0.00000],
#          [ 0.44538, -0.32419,  0.64068,  3.98847],
#          [ 0.49012, -0.31276,  0.68795,  3.90610]],
#
#         [[ 0.45800, -0.33023,  0.64106,  3.98324],
#          [ 0.45829, -0.31487,  0.66203,  3.95529],
#          [ 0.59787, -0.00000,  0.72361,  0.00000],
#          [ 0.70958, -0.37051,  0.78886,  3.63179]]]
# )
# dec_out2 = dec_block(dec_inp, enc_out)
# print("DecoderBlock error: ", rel_error(dec_out2, dec_out_expected))
# ---------------------------------------------------------------------------------------------------------
# from transformers import position_encoding_simple
#
# reset_seed(0)
# K = 4
# M = emb_size = 4
#
# y = position_encoding_simple(K, M)
# y_expected = torch.tensor(
#     [
#         [
#             [0.00000, 0.00000, 0.00000, 0.00000],
#             [0.25000, 0.25000, 0.25000, 0.25000],
#             [0.50000, 0.50000, 0.50000, 0.50000],
#             [0.75000, 0.75000, 0.75000, 0.75000],
#         ]
#     ]
# )
#
# print("position_encoding_simple error: ", rel_error(y, y_expected))
#
# K = 5
# M = emb_size = 3
#
#
# y = position_encoding_simple(K, M)
# y_expected = torch.tensor(
#     [
#         [
#             [0.00000, 0.00000, 0.00000],
#             [0.20000, 0.20000, 0.20000],
#             [0.40000, 0.40000, 0.40000],
#             [0.60000, 0.60000, 0.60000],
#             [0.80000, 0.80000, 0.80000],
#         ]
#     ]
# )
# print("position_encoding_simple error: ", rel_error(y, y_expected))
# ---------------------------------------------------------------------------------------------------------
# from transformers import position_encoding_sinusoid
#
# reset_seed(0)
# K = 4
# M = emb_size = 4
#
# y1 = position_encoding_sinusoid(K, M)
# y_expected = torch.tensor(
#     [
#         [
#             [0.00000, 1.00000, 0.00000, 1.00000],
#             [0.84147, 0.54030, 0.84147, 0.54030],
#             [0.90930, -0.41615, 0.90930, -0.41615],
#             [0.14112, -0.98999, 0.14112, -0.98999],
#         ]
#     ]
# )
#
# print("position_encoding error: ", rel_error(y1, y_expected))
#
# K = 5
# M = emb_size = 3
#
#
# y2 = position_encoding_sinusoid(K, M)
# y_expected = torch.tensor(
#     [
#         [
#             [0.00000, 1.00000, 0.00000],
#             [0.84147, 0.54030, 0.84147],
#             [0.90930, -0.41615, 0.90930],
#             [0.14112, -0.98999, 0.14112],
#             [-0.75680, -0.65364, -0.75680],
#         ]
#     ]
# )
# print("position_encoding error: ", rel_error(y2, y_expected))
# ---------------------------------------------------------------------------------------------------------
# from sklearn.model_selection import train_test_split
# from transformers import AddSubDataset
#
# BATCH_SIZE = 16
#
# X, y = data["inp_expression"], data["out_expression"]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#
# train_data = AddSubDataset(
#     X_train,
#     y_train,
#     convert_str_to_tokens,
#     SPECIAL_TOKENS,
#     32,
#     position_encoding_simple,
# )
# valid_data = AddSubDataset(
#     X_test, y_test, convert_str_to_tokens, SPECIAL_TOKENS, 32, position_encoding_simple
# )
#
# train_loader = torch.utils.data.DataLoader(
#     train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
# )
# valid_loader = torch.utils.data.DataLoader(
#     valid_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
# )
#
# import torch.optim as optim
# from transformers import Transformer
#
# inp_seq_len = 9
# out_seq_len = 5
# num_heads = 4
# emb_dim = 32
# dim_feedforward = 64
# dropout = 0.2
# num_enc_layers = 4
# num_dec_layers = 4
# vocab_len = len(vocab)
#
# model = Transformer(
#     num_heads,
#     emb_dim,
#     dim_feedforward,
#     dropout,
#     num_enc_layers,
#     num_dec_layers,
#     vocab_len,
# )
# for it in train_loader:
#   it
#   break
# inp, inp_pos, out, out_pos = it
# device = DEVICE
# model = model.to(device)
# inp_pos = inp_pos.to(device)
# out_pos = out_pos.to(device)
# out = out.to(device)
# inp = inp.to(device)
#
#
# model_out = model(inp.long(), inp_pos, out.long(), out_pos)
# assert model_out.size(0) == BATCH_SIZE * (out_seq_len - 1)
# assert model_out.size(1) == vocab_len
# ---------------------------------------------------------------------------------------------------------
# from transformers import LabelSmoothingLoss, CrossEntropyLoss
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from transformers import Transformer
# from a5_helper import train as train_transformer
# from a5_helper import val as val_transformer
#
# X, y = data["inp_expression"], data["out_expression"]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#
# inp_seq_len = 9
# out_seq_len = 5
# num_heads = 4
# emb_dim = 32
# dim_feedforward = 32
# dropout = 0.2
# num_enc_layers = 1
# num_dec_layers = 1
# vocab_len = len(vocab)
# BATCH_SIZE = 4
# num_epochs=200 #number of epochs
# lr=1e-3 #learning rate after warmup
# loss_func = CrossEntropyLoss
# warmup_interval = None #number of iterations for warmup
#
# model = Transformer(
#     num_heads,
#     emb_dim,
#     dim_feedforward,
#     dropout,
#     num_enc_layers,
#     num_dec_layers,
#     vocab_len,
# )
# train_data = AddSubDataset(
#     X_train,
#     y_train,
#     convert_str_to_tokens,
#     SPECIAL_TOKENS,
#     emb_dim,
#     position_encoding_simple,
# )
# valid_data = AddSubDataset(
#     X_test,
#     y_test,
#     convert_str_to_tokens,
#     SPECIAL_TOKENS,
#     emb_dim,
#     position_encoding_simple,
# )
#
# train_loader = torch.utils.data.DataLoader(
#     train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
# )
# valid_loader = torch.utils.data.DataLoader(
#     valid_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
# )
#
# small_dataset = torch.utils.data.Subset(
#     train_data, torch.linspace(0, len(train_data) - 1, steps=4).long()
# )
# small_train_loader = torch.utils.data.DataLoader(
#     small_dataset, batch_size=4, pin_memory=True, num_workers=1, shuffle=False
# )
#
# if __name__ == '__main__':
#     #Overfitting the model
#     trained_model = train_transformer(
#         model,
#         small_train_loader,
#         small_train_loader,
#         loss_func,
#         num_epochs=num_epochs,
#         lr=lr,
#         batch_size=BATCH_SIZE,
#         warmup_interval=warmup_interval,
#         device=DEVICE,
#     )
#
#     #Overfitted accuracy
#     print(
#         "Overfitted accuracy: ",
#         "{:.4f}".format(
#             val_transformer(
#                 trained_model,
#                 small_train_loader,
#                 CrossEntropyLoss,
#                 batch_size=4,
#                 device=DEVICE,
#             )[1]
#         ),
#     )
# ---------------------------------------------------------------------------------------------------------
import torch.optim as optim
from transformers import Transformer
from sklearn.model_selection import train_test_split
from a5_helper import train as train_transformer
from a5_helper import val as val_transformer

X, y = data["inp_expression"], data["out_expression"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

inp_seq_len = 9
out_seq_len = 5
BATCH_SIZE = 256

#You should change these!

num_heads = 8
emb_dim = 256
dim_feedforward = 512
dropout = 0.1
num_enc_layers = 4
num_dec_layers = 4
vocab_len = len(vocab)
loss_func = CrossEntropyLoss
poss_enc = position_encoding_simple
num_epochs = 200
warmup_interval = 500
lr = 5e-4

model = Transformer(
    num_heads,
    emb_dim,
    dim_feedforward,
    dropout,
    num_enc_layers,
    num_dec_layers,
    vocab_len,
)


train_data = AddSubDataset(
    X_train,
    y_train,
    convert_str_to_tokens,
    SPECIAL_TOKENS,
    emb_dim,
    position_encoding_sinusoid,
)
valid_data = AddSubDataset(
    X_test,
    y_test,
    convert_str_to_tokens,
    SPECIAL_TOKENS,
    emb_dim,
    position_encoding_sinusoid,
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)

if __name__ == '__main__':
    # Training the model with complete data
    trained_model = train_transformer(
        model,
        train_loader,
        valid_loader,
        loss_func,
        num_epochs,
        lr=lr,
        batch_size=BATCH_SIZE,
        warmup_interval=warmup_interval,
        device=DEVICE
    )
    weights_path = os.path.join('D:/PythonProject/UMichLearn/Assignment5', "transformer.pt")
    torch.save(trained_model.state_dict(), weights_path)

    # Final validation accuracy
    print(
        "Final Model accuracy: ",
        "{:.4f}".format(
            val_transformer(
                trained_model, valid_loader, LabelSmoothingLoss, 4, device=DEVICE
            )[1]
        ),
    )