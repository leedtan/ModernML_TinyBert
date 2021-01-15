import pdb
import copy
import argparse
import glob
import json
import logging
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.optim import Adam
from options import get_parser
from model_utils import ElectraForSequenceClassification
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from transformers import glue_compute_metrics as compute_metrics
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from prior_wd_optim import PriorWD


from transformers.modeling_bert import BertLayerNorm
from mixout import MixLinear, mixout_layer


def manipulate_model(
    model,
    encoder_temp,
    first_frozen_index,
    first_mixout_index,
    first_finetune_index,
    first_reinit_index,
    args,
):
    layer_itr = 0
    mix_counter = 0
    for sup_module in list(model.modules()):
        for name, module in sup_module.named_children():
            if isinstance(module, nn.Linear):
                if layer_itr == 0:
                    module.weight.data.normal_(
                        mean=0.0, std=encoder_temp.config.initializer_range
                    )
                # Frozen
                if layer_itr >= first_frozen_index and layer_itr < first_mixout_index:
                    for param in module.parameters():
                        param.requires_grad = False
                # Mixout
                if layer_itr >= first_mixout_index and layer_itr < first_finetune_index:
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    mix_depth = (layer_itr - 1) // (args.mixout_layers)
                    mix_depth = mix_depth / float(args.mixout_layers - 1)
                    mix_percent = (
                        mix_depth * args.mixout + (1 - mix_depth) * args.mixout_decay
                    )
                    if 1:
                        new_module = mixout_layer(
                            module,
                            mix_percent if args.mixout_decay > 0.0 else args.mixout,
                            args.device,
                            layer_mixout=args.layer_mixout,
                            frozen=True,
                            norm_flag=args.normalize,
                            name=str(mix_counter) + "_mix_layer",
                        )
                        mix_counter += 1
                    else:
                        new_module = MixLinear(
                            module.in_features,
                            module.out_features,
                            bias,
                            target_state_dict["weight"],
                            args.mixout,
                        )
                        new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)

                # Reinit (do nothing):

                layer_itr += 1
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    if args.reinit_layers > 0:
        for layer in encoder_temp.encoder.layer[-args.reinit_layers :]:
            for module in layer.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(
                        mean=0.0, std=encoder_temp.config.initializer_range
                    )
                    module.weight.data.zero_()
                elif isinstance(module, BertLayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
