#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch


def split_chessboard(x, num_split):
    """
    x: b * c * h * w
    Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    h, w = H // num_split, W // num_split
    x_split = torch.cat(
        [
            x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w]
            for i in range(num_split)
            for j in range(num_split)
        ],
        dim=0,
    )
    return x_split


def merge_chessboard(x, num_split):
    """
    x: b * c * h * w
    Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
    (inverse of split_chessboard)
    """
    B, C, H, W = x.shape
    assert B % (num_split**2) == 0
    b = B // (num_split**2)
    x_merge = torch.cat(
        [
            torch.cat(
                [
                    x[(i * num_split + j) * b : (i * num_split + j + 1) * b]
                    for j in range(num_split)
                ],
                dim=-1,
            )
            for i in range(num_split)
        ],
        dim=-2,
    )
    return x_merge


def batched_forward(model, x, batch_size=-1):
    if batch_size == -1:
        return model(x)
    else:
        x_batched = x.split(batch_size)
        outs = [model(x) for x in x_batched]
        return torch.cat(outs, dim=0)
