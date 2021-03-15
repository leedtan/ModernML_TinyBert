import logging
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import glue_output_modes as output_modes

from glue_utils import load_and_cache_examples, set_seed
from manipulate_model import manipulate_model
from model_utils import ElectraForSequenceClassification
from options import get_parser
from train import run_train
from transformers_local import glue_processors as processors

try:
    pass
except ImportError:
    pass


logger = logging.getLogger(__name__)


def main(args):
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args.seed)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
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
    if args.model_type == "electra":
        model = ElectraForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    if num_labels != num_labels_old:
        config.num_labels = num_labels
        model.num_labels = num_labels
        if args.model_type in ["roberta", "bert", "electra"]:
            from transformers.modeling_roberta import RobertaClassificationHead

            model.classifier = (
                RobertaClassificationHead(config)
                if args.model_type == "roberta"
                else nn.Linear(config.hidden_size, config.num_labels)
            )
            for module in model.classifier.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
        elif args.model_type == "bart":
            from transformers.modeling_bart import BartClassificationHead

            model.classification_head = BartClassificationHead(
                config.d_model,
                config.d_model,
                config.num_labels,
                config.classif_dropout,
            )
            model.model._init_weights(model.classification_head.dense)
            model.model._init_weights(model.classification_head.out_proj)
        elif args.model_type == "xlnet":
            model.logits_proj = nn.Linear(config.d_model, config.num_labels)
            model.transformer._init_weights(model.logits_proj)
        else:
            raise NotImplementedError

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    if args.reinit_pooler:
        if args.model_type in ["bert", "roberta"]:
            encoder_temp = getattr(model, args.model_type)
            encoder_temp.pooler.dense.weight.data.normal_(
                mean=0.0, std=encoder_temp.config.initializer_range
            )
            encoder_temp.pooler.dense.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True
        elif args.model_type in ["xlnet", "bart", "electra"]:
            raise ValueError(f"{args.model_type} does not have a pooler at the end")
        else:
            raise NotImplementedError

    if args.reinit_layers > 0 or args.mixout_layers > 0:
        if args.model_type in ["bert", "roberta", "electra"]:

            encoder_temp = getattr(model, args.model_type)
            frozen_layers = args.frozen_layers
            mix_layers = args.mixout_layers
            finetune_layers = args.finetune_layers
            re_layers = args.reinit_layers
            first_frozen_index = 1
            first_mixout_index = 1 + frozen_layers * 12
            first_finetune_index = 1 + (frozen_layers + mix_layers) * 12
            first_reinit_index = 12 * (12 - re_layers) + 1
            assert (12 * (12 - re_layers) + 1) == (
                1 + (frozen_layers + mix_layers + finetune_layers) * 12
            )
            manipulate_model(
                model,
                encoder_temp,
                first_frozen_index,
                first_mixout_index,
                first_finetune_index,
                first_reinit_index,
                args,
            )

        elif args.model_type == "xlnet":
            from transformers.modeling_xlnet import (
                XLNetLayerNorm,
                XLNetRelativeAttention,
            )

            for layer in model.transformer.layer[-args.reinit_layers :]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        # Slightly different from the TF version which uses truncated_normal for initialization
                        # cf https://github.com/pytorch/pytorch/pull/5617
                        module.weight.data.normal_(
                            mean=0.0, std=model.transformer.config.initializer_range
                        )
                        if isinstance(module, nn.Linear) and module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, XLNetLayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    elif isinstance(module, XLNetRelativeAttention):
                        for param in [
                            module.q,
                            module.k,
                            module.v,
                            module.o,
                            module.r,
                            module.r_r_bias,
                            module.r_s_bias,
                            module.r_w_bias,
                            module.seg_embed,
                        ]:
                            param.data.normal_(
                                mean=0.0, std=model.transformer.config.initializer_range
                            )
        elif args.model_type == "bart":
            for layer in model.model.decoder.layers[-args.reinit_layers :]:
                for module in layer.modules():
                    model.model._init_weights(module)

        else:
            raise NotImplementedError

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, evaluate=False, logger=logger
        )
        global_step, tr_loss, result = run_train(
            args, train_dataset, model, tokenizer, logger
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    return result


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
