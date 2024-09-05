import numpy as np
import torch
import json
import math


class GlobalCMVN(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, istd: torch.Tensor, norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


def load_cmvn_json(json_cmvn_file):
    with open(json_cmvn_file) as f:
        cmvn_json = json.load(f)

    avg = cmvn_json["mean_stat"]
    var = cmvn_json["var_stat"]
    count = cmvn_json["frame_num"]
    for i in range(len(avg)):
        avg[i] /= count
        var[i] = var[i] / count - avg[i] * avg[i]
        if var[i] < 1.0e-20:
            var[i] = 1.0e-20
        var[i] = 1.0 / math.sqrt(var[i])
    cmvn = np.array([avg, var])
    return cmvn


def load_cmvn_kaldi(kaldi_cmvn_file):
    avg = []
    var = []
    with open(kaldi_cmvn_file, "r") as file:
        # kaldi binary file start with '\0B'
        if file.read(2) == "\0B":
            logging.error(
                "kaldi cmvn binary file is not supported, please "
            )
            sys.exit(1)
        file.seek(0)
        arr = file.read().split()
        assert arr[0] == "["
        assert arr[-2] == "0"
        assert arr[-1] == "]"
        feat_dim = int((len(arr) - 2 - 2) / 2)
        for i in range(1, feat_dim + 1):
            avg.append(float(arr[i]))
        count = float(arr[feat_dim + 1])
        for i in range(feat_dim + 2, 2 * feat_dim + 2):
            var.append(float(arr[i]))

    for i in range(len(avg)):
        avg[i] /= count
        var[i] = var[i] / count - avg[i] * avg[i]
        if var[i] < 1.0e-20:
            var[i] = 1.0e-20
        var[i] = 1.0 / math.sqrt(var[i])
    cmvn = np.array([avg, var])
    return cmvn


def load_cmvn(filename, is_json):
    if is_json:
        file = load_cmvn_json(filename)
    else:
        file = load_cmvn_kaldi(filename)
    return file[0], file[1]
