import json
import math
import os
import pathlib
import random

import deepspeed
import fire
import loguru
import numpy as np
import torch
from datasets import load_dataset
from datasets import load_metric
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
    set_seed
)

logger = loguru.logger
world_size = int(os.getenv('WORLD_SIZE', '1'))
logger.info(f"world_size: {world_size}")


def create_experiment_dir(
    checkpoint_dir: pathlib.Path, all_arguments
):
    """ Create an experiment directory and save all arguments in it."""
    hparams_file = checkpoint_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)

    # Create the Tensorboard Dir
    tb_dir = checkpoint_dir / "tb_dir"
    tb_dir.mkdir()


def main(
    checkpoint_dir: str = None,
    # Dataset Params
    train_file: str = None,
    validation_file: str = None,
    # Model Params
    model_name_or_path: str = None,
    # Training Params
    epoch: int = 3,
    batch_size: int = 32,
    checkpoint_every: int = 1000,
    weight_decay: float = 0.0,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 1,
    lr_scheduler_type: SchedulerType = "linear",
    num_warmup_steps: int = 0,
    log_every: int = 10,
    seed=666,
    local_rank: int = -1
):
    """ Train a BERT style model for classification """
    # local_rank will be automatically decided, don't need us to manually specify

    # First init process group
    deepspeed.init_distributed()

    device = (
        torch.device("cuda", local_rank)
        if (local_rank > -1) and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"local_rank: {local_rank}")
    assert checkpoint_dir is not None
    checkpoint_dir = pathlib.Path(checkpoint_dir)

    ########## Creating Experiment Directory ###########
    if local_rank != 0:
        torch.distributed.barrier()

    # Only allow rank 0 to create directory
    if local_rank == 0:
        logger.info("Creating Experiment Directory")
        checkpoint_dir.mkdir(exist_ok=True)
        all_arguments = {
            "task": "classification",
            # Dataset Params
            "train_file": train_file,
            "validation_file": validation_file,
            # Training Params
            "batch_size": batch_size,
            "checkpoint_every": checkpoint_every,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lr_scheduler_type": lr_scheduler_type,
            "num_warmup_steps": num_warmup_steps,
            "seed": seed
        }
        create_experiment_dir(checkpoint_dir, all_arguments)
        logger.info(f"Experiment Directory created at {checkpoint_dir}")
        tb_dir = checkpoint_dir / "tb_dir"
        assert tb_dir.exists()
        summary_writer = SummaryWriter(log_dir=tb_dir)

    if local_rank == 0:
        # other ranks can proceed
        torch.distributed.barrier()

    set_seed(seed)

    ######### Create Dataset #########
    data_files = {}
    if train_file is None or validation_file is None:
        logger.error("Need to specify both train_file and validation_file")
        return
    if (not train_file.endswith(".json")) or (not validation_file.endswith(".json")):
        logger.error("train_file and val_file should all be json files")

    data_files["train"] = train_file
    data_files["validation"] = validation_file
    raw_datasets = load_dataset("json", data_files=data_files)

    # Get the label list
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = 2

    assert model_name_or_path is not None, "You need to specify model_name_or_path"
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    max_seq_length = tokenizer.model_max_length

    # Preprocessing the datasets
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (examples["sentence"],)
        result = tokenizer(*texts, padding=False, max_length=max_seq_length, truncation=True)
        result["labels"] = [label_to_id[l] for l in examples["label"]]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        seed=seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=data_collator
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=batch_size
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = epoch * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    metric = load_metric("accuracy")

    ####### DeepSpeed Engine ########
    logger.info("Creating DeepSpeed Engine")
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "fp16": {
            "enabled": True
        },
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu"
            }
        }
    }

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        lr_scheduler=lr_scheduler,
        config=ds_config
    )
    logger.info("DeepSpeed Engine Created")

    # Get the initial zero-shot accuracy
    model.eval()
    total_num = 0
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            total_num += len(labels)
            metric.add_batch(
                predictions=predictions,
                references=labels,
            )

    eval_metric = metric.compute()
    logger.info(f"total_num: {total_num}, zero-shot accuracy: {eval_metric}")
    if local_rank == 0:
        summary_writer.add_text("Accuracy", f"Zero-shot accuracy: {eval_metric}")

    # Train !
    total_batch_size = batch_size * gradient_accumulation_steps * world_size

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {epoch}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=(local_rank != 0))

    losses = []

    for e in range(epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Forward pass
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass, don't need to aggregate from other GPUs
            model.backward(loss)
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # Optimizer Step
                model.step()
                progress_bar.update(1)

            losses.append(loss.item())
            if step % log_every == 0:
                logger.info("Loss: {0:.4f}".format(np.mean(losses)))
                if local_rank == 0:
                    summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)
            if step % checkpoint_every == 0:
                model.save_checkpoint(save_dir=checkpoint_dir, client_state={"checkpoint_step": step})
                logger.info(f"Saved model to {checkpoint_dir}")

        model.eval()
        total_num = 0
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                total_num += len(labels)
                metric.add_batch(
                    predictions=predictions,
                    references=labels,
                )
        eval_metric = metric.compute()
        logger.info(f"total_num: {total_num}, on epoch {e} accuracy: {eval_metric}")
        if local_rank == 0:
            summary_writer.add_text("Accuracy", f"total_num: {total_num}, on epoch {e} accuracy: {eval_metric}")


if __name__ == "__main__":
    fire.Fire(main)
"""
deepspeed --include localhost:0,1,2,3 train.py --checkpoint_dir trained --model_name_or_path roberta-large --train_file SST-2/train.json --validation_file SST-2/dev.json --batch_size 16

zero-shot accuracy: {'accuracy': 0.4908256880733945}
On epoch 0
On epoch 1
On epoch 2 accuracy:  {'accuracy': 0.9506880733944955}


"""
