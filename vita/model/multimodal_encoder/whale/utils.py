import argparse
import importlib
import json
import os
from distutils.util import strtobool as dist_strtobool

import torch
import yaml

IGNORE_ID = -1


def assign_args_from_yaml(args, yaml_path, prefix_key=None):
    with open(yaml_path) as f:
        ydict = yaml.load(f, Loader=yaml.FullLoader)
    if prefix_key is not None:
        ydict = ydict[prefix_key]
    for k, v in ydict.items():
        k_args = k.replace("-", "_")
        if hasattr(args, k_args):
            setattr(args, k_args, ydict[k])
    return args


def get_model_conf(model_path):
    model_conf = os.path.dirname(model_path) + "/model.json"
    with open(model_conf, "rb") as f:
        print("reading a config file from " + model_conf)
        confs = json.load(f)
    # for asr, tts, mt
    idim, odim, args = confs
    return argparse.Namespace(**args)


def strtobool(x):
    return bool(dist_strtobool(x))


def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'espnet.transform.add_deltas:AddDeltas'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ":" not in import_path:
        raise ValueError(
            "import_path should be one of {} or "
            'include ":", e.g. "espnet.transform.add_deltas:AddDeltas" : '
            "{}".format(set(alias), import_path)
        )
    if ":" not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(":")
    m = importlib.import_module(module_name)
    return getattr(m, objname)


def set_deterministic_pytorch(args):
    # seed setting
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
    return pad


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def subsequent_chunk_mask(
    size: int,
    ck_size: int,
    num_l_cks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_l_cks < 0:
            start = 0
        else:
            start = max((i // ck_size - num_l_cks) * ck_size, 0)
        ending = min((i // ck_size + 1) * ck_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(
    xs: torch.Tensor,
    masks: torch.Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
):
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_l_cks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_l_cks = num_decoding_left_chunks
        else:
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_l_cks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_l_cks = torch.randint(0, max_left_chunks, (1,)).item()
        ck_masks = subsequent_chunk_mask(
            xs.size(1), chunk_size, num_l_cks, xs.device
        )  # (L, L)
        ck_masks = ck_masks.unsqueeze(0)  # (1, L, L)
        ck_masks = masks & ck_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_l_cks = num_decoding_left_chunks
        ck_masks = subsequent_chunk_mask(
            xs.size(1), static_chunk_size, num_l_cks, xs.device
        )  # (L, L)
        ck_masks = ck_masks.unsqueeze(0)  # (1, L, L)
        ck_masks = masks & ck_masks  # (B, L, L)
    else:
        ck_masks = masks
    return ck_masks
