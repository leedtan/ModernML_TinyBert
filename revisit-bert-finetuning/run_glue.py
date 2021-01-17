import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from glue_utils import load_and_cache_examples, set_seed
from manipulate_model import manipulate_model
from model_utils import check_config
from options import get_parser
from train import run_train

logger = logging.getLogger(__name__)


def main(args):
    args = check_config(args, logger)
    set_seed(args.seed)

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    num_labels_old = AutoConfig.from_pretrained(args.model_name_or_path).num_labels
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels_old,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if num_labels != num_labels_old:
        config.num_labels = num_labels
        model.num_labels = num_labels
        model.classifier = nn.Linear(config.hidden_size, config.num_labels)
        for module in model.classifier.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    if args.reinit_pooler:
        encoder_temp = getattr(model, args.model_type)
        encoder_temp.pooler.dense.weight.data.normal_(
            mean=0.0, std=encoder_temp.config.initializer_range
        )
        encoder_temp.pooler.dense.bias.data.zero_()
        for p in encoder_temp.pooler.parameters():
            p.requires_grad = True

    if args.reinit_layers > 0 or args.mixout_layers > 0:

        manipulate_model(
            model,
            args,
        )

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, evaluate=False, logger=logger
        )
        global_step, tr_loss, result = run_train(
            args, train_dataset, model, tokenizer, logger
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can
    # reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)

    return result


if __name__ == "__main__":
    parser = get_parser()
    cmdargs = parser.parse_args()
    main(cmdargs)
