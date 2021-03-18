import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import glue_compute_metrics as compute_metrics

from glue_utils import load_and_cache_examples

try:
    pass
except ImportError:
    pass


def evaluate(args, model, tokenizer, prefix="", eval_datasets=None, logger=None):
    eval_task_names = [args.task_name]
    eval_outputs_dirs = [args.output_dir]

    results = {}
    for i, (eval_task, eval_output_dir) in enumerate(
        zip(eval_task_names, eval_outputs_dirs)
    ):
        if eval_datasets is None:
            eval_dataset = load_and_cache_examples(
                args, eval_task, tokenizer, evaluate=True, logger=logger
            )
            print("eval4 size", eval_dataset.tensors[0].shape)
        elif isinstance(eval_datasets, list):
            eval_dataset = eval_datasets[i]
        else:
            raise ValueError("Wrong Pre-fetched Eval Set")

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                if args.model_type not in {"distilbert", "bart"}:
                    inputs["token_type_ids"] = (
                        batch[2]
                        if args.model_type in ["bert", "xlnet", "albert"]
                        else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        if eval_task == "mnlihans":
            eval_task = "hans"
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

    results["loss"] = eval_loss

    return results
