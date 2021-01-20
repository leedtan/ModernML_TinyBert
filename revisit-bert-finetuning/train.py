import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from glue_utils import (
    get_optimizer_grouped_parameters,
    load_and_cache_examples,
    set_seed,
)
from validate import evaluate

try:
    pass
except ImportError:
    pass

from prior_wd_optim import PriorWD


def run_train(args, train_dataset, model, tokenizer, logger):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        with open(f"{args.output_dir}/raw_log.txt", "w") as f:
            pass  # create a new file

    if args.train_batch_size == 0:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    eval_task_names = (args.task_name,)
    eval_datasets = [
        load_and_cache_examples(args, task, tokenizer, evaluate=True, logger=logger)
        for task in eval_task_names
    ]
    if args.test_val_split:
        assert len(eval_datasets) == 1
        val_test_indices = []
        for i, eval_dataset in enumerate(eval_datasets):
            class2idx = defaultdict(list)
            for i, sample in enumerate(eval_dataset):
                class2idx[sample[-1].item()].append(i)
            val_indices = []
            test_indices = []
            for class_num, indices in class2idx.items():
                state = np.random.RandomState(1)
                state.shuffle(indices)
                class_val_indices, class_test_indices = (
                    indices[: len(indices) // 2],
                    indices[len(indices) // 2 :],
                )
                val_indices += class_val_indices
                test_indices += class_test_indices
            val_indices = torch.tensor(val_indices).long()
            test_indices = torch.tensor(test_indices).long()
            val_test_indices.append((val_indices, test_indices))
            eval_dataset.tensors = [t[val_indices] for t in eval_dataset.tensors]

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # assert args.logging_steps == 0 or args.num_loggings == 0, "Can only use 1 logging option"
    if args.logging_steps == 0:
        assert args.num_loggings > 0
        args.logging_steps = t_total // args.num_loggings

    if args.warmup_ratio > 0:
        # assert args.warmup_steps == 0
        args.warmup_steps = int(args.warmup_ratio * t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model)

    if args.use_torch_adamw:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            correct_bias=not args.use_bertadam,
        )

    optimizer = PriorWD(optimizer, use_prior_wd=args.prior_weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        print("USING FP16")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    best_val_acc = -100.0
    best_model = None
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args.seed)  # Added here for reproductibility

    # pretrained_model = copy.deepcopy(model)
    # pretrained_model.eval()
    for module in list(model.modules()):
        if hasattr(module, "is_our_mixout"):
            break
    # print(module)
    epoch_counter = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="iteration", disable=args.local_rank not in [-1, 0]
        )
        if args.unfreeze_after_epoch == epoch_counter:
            print("\n unfreezing mixout layers \n")
        for module in list(model.modules()):
            if hasattr(module, "is_our_mixout"):
                if args.unfreeze_after_epoch == epoch_counter:
                    module.frozen = False
        for step, batch in enumerate(epoch_iterator):
            # skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type not in {"distilbert", "bart"}:
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else none
                )  # xlm, distilbert, roberta, and xlm-roberta don't use segment_ids
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            l2_reg = 0
            layer_itr = 0
            total_mix_layers = args.mixout_layers * 12
            for sup_module in list(model.modules()):
                for name, module in sup_module.named_children():
                    if hasattr(module, "is_our_mixout"):
                        if args.l2_scaling:
                            mix_depth = layer_itr / total_mix_layers
                            l2_reg_layer = args.l2_reg_mult * (
                                args.l2_reg_decay ** mix_depth
                            )
                            l2_reg += module.regularize(l2_reg_layer)
                            layer_itr += 1
                        else:
                            l2_reg += module.regularize(args.l2_reg_mult)

            loss += l2_reg
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and (
                    (args.logging_steps > 0 and global_step % args.logging_steps == 0)
                    or (global_step == t_total)
                ):
                    logs = {}
                    if args.local_rank == -1:
                        results = evaluate(
                            args,
                            model,
                            tokenizer,
                            eval_datasets=eval_datasets,
                            logger=logger,
                        )
                        for key, value in results.items():
                            eval_key = "val_{}".format(key)
                            logs[eval_key] = value

                    if (
                        args.local_rank in [-1, 0]
                        and args.save_best
                        and logs["val_acc"] > best_val_acc
                    ):
                        output_dir = os.path.join(args.output_dir, "checkpoint-best")
                        os.makedirs(output_dir, exist_ok=True)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s", output_dir
                        )

                    if "val_acc" in logs:
                        if logs["val_acc"] > best_val_acc:
                            best_val_acc = logs["val_acc"]
                            best_model = {
                                k: v.cpu().detach()
                                for k, v in model.state_dict().items()
                            }
                        logs["best_val_acc"] = best_val_acc
                    elif "val_mcc" in logs:
                        if logs["val_mcc"] > best_val_acc:
                            best_val_acc = logs["val_mcc"]
                            best_model = {
                                k: v.cpu().detach()
                                for k, v in model.state_dict().items()
                            }
                        logs["best_val_mcc"] = best_val_acc
                    elif "val_spearmanr":
                        if logs["val_spearmanr"] > best_val_acc:
                            best_val_acc = logs["val_spearmanr"]
                            best_model = {
                                k: v.cpu().detach()
                                for k, v in model.state_dict().items()
                            }
                        logs["best_val_spearmanr"] = best_val_acc
                    else:
                        raise ValueError(f"logs:{logs}")

                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar

                    if args.logging_steps > 0:
                        if global_step % args.logging_steps == 0:
                            loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        else:
                            loss_scalar = (tr_loss - logging_loss) / (
                                global_step % args.logging_steps
                            )
                    else:
                        loss_scalar = (tr_loss - logging_loss) / global_step
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    logs["step"] = global_step
                    with open(f"{args.output_dir}/raw_log.txt", "a") as f:
                        if os.stat(f"{args.output_dir}/raw_log.txt").st_size == 0:
                            for k in logs:
                                f.write(f"{k},")
                            f.write("\n")
                        for v in logs.values():
                            f.write(f"{v:.6f},")
                        f.write("\n")

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-last".format(global_step)
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        epoch_counter += 1
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    args.resplit_val = 0  # test on the original test_set
    eval_task_names = (args.task_name,)

    # test the last checkpoint on the second half
    eval_datasets = [
        load_and_cache_examples(args, task, tokenizer, evaluate=True, logger=logger)
        for task in eval_task_names
    ]
    if args.test_val_split:
        for i, eval_dataset in enumerate(eval_datasets):
            test_indices = val_test_indices[i][1]
            eval_dataset.tensors = [t[test_indices] for t in eval_dataset.tensors]

    result = evaluate(
        args, model, tokenizer, eval_datasets=eval_datasets, logger=logger
    )
    result["step"] = t_total
    # overwriting validation results
    with open(f"{args.output_dir}/test_last_log.txt", "w") as f:
        f.write(",".join(["test_" + k for k in result.keys()]) + "\n")
        f.write(",".join([f"{v:.4f}" for v in result.values()]))

    if best_model is not None:
        model.load_state_dict(best_model)

    # test on the second half
    eval_datasets = [
        load_and_cache_examples(args, task, tokenizer, evaluate=True, logger=logger)
        for task in eval_task_names
    ]
    if args.test_val_split:
        for i, eval_dataset in enumerate(eval_datasets):
            test_indices = val_test_indices[i][1]
            eval_dataset.tensors = [t[test_indices] for t in eval_dataset.tensors]

    result = evaluate(
        args, model, tokenizer, eval_datasets=eval_datasets, logger=logger
    )
    result["step"] = t_total
    # overwriting validation results
    with open(f"{args.output_dir}/test_best_log.txt", "w") as f:
        f.write(",".join(["test_" + k for k in result.keys()]) + "\n")
        f.write(",".join([f"{v:.4f}" for v in result.values()]))

    return global_step, tr_loss / global_step, result
