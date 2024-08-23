import torch

# from pytorch101 import create_sample_tensor, mutate_tensor, count_tensor_elements
#
# # Create a sample tensor
# x = create_sample_tensor()
# print('Here is the sample tensor:')
# print(x)
#
# # Mutate the tensor by setting a few elements
# indices = [(0, 0), (1, 0), (1, 1)]
# values = [4, 5, 6]
# mutate_tensor(x, indices, values)
# print('\nAfter mutating:')
# print(x)
# print('\nCorrect shape: ', x.shape == (3, 2))
# print('x[0, 0] correct: ', x[0, 0].item() == 4)
# print('x[1, 0] correct: ', x[1, 0].item() == 5)
# print('x[1, 1] correct: ', x[1, 1].item() == 6)
#
# # Check the number of elements in the sample tensor
# num = count_tensor_elements(x)
# print('\nNumber of elements in x: ', num)
# print('Correctly counted: ', num == 6)
# -------------------------------------------------------------------
# from pytorch101 import create_tensor_of_pi
#
# x = create_tensor_of_pi(4, 5)
#
# print('x is a tensor:', torch.is_tensor(x))
# print('x has correct shape: ', x.shape == (4, 5))
# print('x is filled with pi: ', (x == 3.14).all().item() == 1)
# -------------------------------------------------------------------
# from pytorch101 import multiples_of_ten
#
# start = 5
# stop = 25
# x = multiples_of_ten(start, stop)
# print('Correct dtype: ', x.dtype == torch.float64)
# print('Correct shape: ', x.shape == (2,))
# print('Correct values: ', x.tolist() == [10, 20])
#
# # If there are no multiples of ten in the given range you should return an empty tensor
# start = 5
# stop = 7
# x = multiples_of_ten(start, stop)
# print('\nCorrect dtype: ', x.dtype == torch.float64)
# print('Correct shape: ', x.shape == (0,))
# -------------------------------------------------------------------
# # We will use this helper function to check your results
# def check(orig, actual, expected):
#     if not torch.is_tensor(actual):
#         return False
#     expected = torch.tensor(expected)
#     same_elements = (actual == expected).all().item()
#     same_storage = (orig.storage().data_ptr() == actual.storage().data_ptr())
#     return same_elements and same_storage
#
# from pytorch101 import slice_indexing_practice
#
# # Create the following rank 2 tensor of shape (3, 5)
# # [[ 1  2  3  4  5]
# #  [ 6  7  8  9 10]
# #  [11 12 13 14 15]]
# x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 8, 10], [11, 12, 13, 14, 15]])
# out = slice_indexing_practice(x)
#
# last_row = out[0]
# print('last_row:')
# print(last_row)
# correct = check(x, last_row, [11, 12, 13, 14, 15])
# print('Correct: %r\n' % correct)
#
# third_col = out[1]
# print('third_col:')
# print(third_col)
# correct = check(x, third_col, [[3], [8], [13]])
# print('Correct: %r\n' % correct)
#
# first_two_rows_three_cols = out[2]
# print('first_two_rows_three_cols:')
# print(first_two_rows_three_cols)
# correct = check(x, first_two_rows_three_cols, [[1, 2, 3], [6, 7, 8]])
# print('Correct: %r\n' % correct)
#
# even_rows_odd_cols = out[3]
# print('even_rows_odd_cols:')
# print(even_rows_odd_cols)
# correct = check(x, even_rows_odd_cols, [[2, 4], [12, 14]])
# print('Correct: %r\n' % correct)
# -------------------------------------------------------------------
# from pytorch101 import slice_assignment_practice
#
# # note: this "x" has one extra row, intentionally
# x = torch.zeros(5, 7, dtype=torch.int64)
# print('Here is x before calling slice_assignment_practice:')
# print(x)
# slice_assignment_practice(x)
# print('Here is x after calling slice assignment practice:')
# print(x)
#
# expected = [
#     [0, 1, 2, 2, 2, 2, 0],
#     [0, 1, 2, 2, 2, 2, 0],
#     [3, 4, 3, 4, 5, 5, 0],
#     [3, 4, 3, 4, 5, 5, 0],
#     [0, 0, 0, 0, 0, 0, 0],
# ]
# print('Correct: ', x.tolist() == expected)
# -------------------------------------------------------------------
from pytorch101 import shuffle_cols, reverse_rows, take_one_elem_per_col

# Build a tensor of shape (4, 3):
# [[ 1,  2,  3],
#  [ 4,  5,  6],
#  [ 7,  8,  9],
#  [10, 11, 12]]
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print('Here is x:')
print(x)

y1 = shuffle_cols(x)
print('\nHere is shuffle_cols(x):')
print(y1)
expected = [[1, 1, 3, 2], [4, 4, 6, 5], [7, 7, 9, 8], [10, 10, 12, 11]]
y1_correct = torch.is_tensor(y1) and y1.tolist() == expected
print('Correct: %r\n' % y1_correct)

y2 = reverse_rows(x)
print('Here is reverse_rows(x):')
print(y2)
expected = [[10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3]]
y2_correct = torch.is_tensor(y2) and y2.tolist() == expected
print('Correct: %r\n' % y2_correct)

y3 = take_one_elem_per_col(x)
print('Here is take_one_elem_per_col(x):')
print(y3)
expected = [4, 2, 12]
y3_correct = torch.is_tensor(y3) and y3.tolist() == expected
print('Correct: %r' % y3_correct)