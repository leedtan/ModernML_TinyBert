import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)

from transformers_local import glue_output_modes as output_modes
from transformers_local import glue_processors as processors

try:
    pass
except ImportError:
    pass


def get_optimizer_grouped_parameters(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    if args.layerwise_learning_rate_decay == 1.0:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": args.learning_rate,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "classifier" in n or "pooler" in n
                ],
                "weight_decay": 0.0,
                "lr": args.learning_rate,
            },
        ]

        if args.model_type in ["bert", "roberta", "electra"]:
            model.config.num_hidden_layers
            layers = [getattr(model, args.model_type).embeddings] + list(
                getattr(model, args.model_type).encoder.layer
            )
            layers.reverse()
            lr = args.learning_rate
            for layer in layers:
                lr *= args.layerwise_learning_rate_decay
                optimizer_grouped_parameters += [
                    {
                        "params": [
                            p
                            for n, p in layer.named_parameters()
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                        "lr": lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in layer.named_parameters()
                            if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]
        else:
            raise NotImplementedError
    return optimizer_grouped_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_cache_examples(args, task, tokenizer, evaluate=False, logger=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if (evaluate and args.resplit_val <= 0) else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir)
            if evaluate
            else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if args.downsample_trainset > 0 and not evaluate:
        assert (args.downsample_trainset + args.resplit_val) <= len(features)

    if args.downsample_trainset > 0 or args.resplit_val > 0:
        set_seed(args.seed)  # use the same seed for downsample
        if output_mode == "classification":
            label_to_idx = defaultdict(list)
            for i, f in enumerate(features):
                label_to_idx[f.label].append(i)

            samples_per_class = (
                args.resplit_val if evaluate else args.downsample_trainset
            )
            samples_per_class = samples_per_class // len(label_to_idx)

            for k in label_to_idx:
                label_to_idx[k] = np.array(label_to_idx[k])
                np.random.shuffle(label_to_idx[k])
                if evaluate:
                    if args.resplit_val > 0:
                        label_to_idx[k] = label_to_idx[k][-samples_per_class:]
                    else:
                        pass
                else:
                    if args.resplit_val > 0 and args.downsample_trainset <= 0:
                        samples_per_class = len(
                            label_to_idx[k]
                        ) - args.resplit_val // len(label_to_idx)
                    label_to_idx[k] = label_to_idx[k][:samples_per_class]

            sampled_idx = np.concatenate(list(label_to_idx.values()))
        else:
            if args.downsample_trainset > 0:
                sampled_idx = torch.randperm(len(features))[: args.downsample_trainset]
            else:
                raise NotImplementedError
        set_seed(args.seed)
        features = [features[i] for i in sampled_idx]

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    import pdb

    pdb.set_trace()
    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )
    return dataset
