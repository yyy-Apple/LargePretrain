import json
import pathlib
import random
import math
import deepspeed
import loguru
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import os
import numpy as np
import fire
from transformers import (
    AdamW,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
    set_seed
)
from argparse import Namespace

from bert import ShardedRoBERTa
import mpu

logger = loguru.logger


### Helper Functions ###
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


def log_rank_0(message):
    if torch.distributed.get_rank() == 0:
        logger.info(f"[Global rank 0] {message}")


def get_model(args):
    log_rank_0("Buliding Sharded RoBERTa.")
    model_wrapper = ShardedRoBERTa(args)
    model_wrapper.shard()
    log_rank_0("Sharded RoBERTa Built.")
    return model_wrapper


def main(
        checkpoint_dir: str = None,
        # Dataset Params
        train_file: str = None,
        validation_file: str = None,
        dataset_cache_dir: str = None,
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
        local_rank: int = -1,
):
    deepspeed.init_distributed()
    mpu.initialize_model_parallel(4)
    global_rank = torch.distributed.get_rank()
    model_parallel_rank = mpu.get_model_parallel_rank()
    model_parallel_size = mpu.get_model_parallel_world_size()
    data_parallel_size = mpu.get_data_parallel_world_size()
    log_rank_0(f"model_parallel_size: {model_parallel_size}")
    log_rank_0(f"data_parallel_size: {data_parallel_size}")

    device = (
        torch.device("cuda", local_rank)
        if (local_rank > -1) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    assert checkpoint_dir is not None
    if os.path.exists(checkpoint_dir):
        os.system(f"rm -rf {checkpoint_dir}")
    checkpoint_dir = pathlib.Path(checkpoint_dir)

    ########## Creating Experiment Directory ###########
    torch.distributed.barrier()

    # Only allow rank 0 to create directory
    if global_rank == 0:
        logger.info("Creating Experiment Directory")
        checkpoint_dir.mkdir(exist_ok=False)
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

    set_seed(seed)
    torch.distributed.barrier()

    ######### Create Dataset #########
    data_files = {}
    if train_file is None or validation_file is None:
        logger.error("Need to specify both train_file and validation_file")
        return
    if (not train_file.endswith(".json")) or (not validation_file.endswith(".json")):
        logger.error("train_file and val_file should all be json files")

    data_files["train"] = train_file
    data_files["validation"] = validation_file
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=dataset_cache_dir)

    # Get the label list
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    max_seq_length = min(tokenizer.model_max_length, 512)
    log_rank_0(f"Tokenizer set, max_seq_length: {max_seq_length}")

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

    torch.distributed.barrier()

    if model_parallel_rank == 0:
        # Get model arguments
        model_args = Namespace(
            num_labels=num_labels,
            checkpoint=model_name_or_path,
        )
        model_wrapper = get_model(model_args)

        model_wrapper.model.config.label2id = label_to_id
        model_wrapper.model.config.id2label = {id: label for label, id in model_wrapper.model.config.label2id.items()}
    else:
        # Seudo Model
        model_wrapper = torch.nn.Linear(2, 3)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_wrapper.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model_wrapper.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=data_parallel_size,
        rank=(global_rank // model_parallel_size),  # 0, 1, 2, 3
        seed=seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # sampler option is mutually exclusive with shuffle
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
    log_rank_0("Creating DeepSpeed Engine")
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
        model=model_wrapper,
        optimizer=optimizer,
        model_parameters=model_wrapper.parameters(),
        lr_scheduler=lr_scheduler,
        mpu=mpu,
        config=ds_config
    )
    torch.distributed.barrier()
    log_rank_0("DeepSpeed Engine Created")

    # Get the initial zero-shot accuracy
    model.eval()
    if model_parallel_rank == 0:
        total_num = 0
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                _, logits = model.forward(**batch)
                predictions = logits.argmax(dim=-1)
                labels = batch["labels"]
                total_num += len(labels)
                metric.add_batch(
                    predictions=predictions,
                    references=labels,
                )
        eval_metric = metric.compute()
        logger.info(f"total_num: {total_num}, zero-shot accuracy: {eval_metric}")
        if global_rank == 0:
            summary_writer.add_text("Accuracy", f"Zero-shot accuracy: {eval_metric}")

    # Train !
    total_batch_size = batch_size * gradient_accumulation_steps * data_parallel_size

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {epoch}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=(global_rank != 0))

    losses = []
    # For later Seudo Loss
    a = torch.tensor(2., requires_grad=True, device=device)
    b = torch.tensor(6., requires_grad=True, device=device)
    for e in range(epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Forward pass
            if model_parallel_rank == 0:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                loss, _ = model(**batch)
                logger.info(f"mpu.get_model_parallel_group(): {mpu.get_model_parallel_group()}")
            else:
                loss = a**2 - b**2

            logger.info(f"[Rank {global_rank}] loss: {loss}")

            # Backward pass, don't need to aggregate from other GPUs
            model.backward(loss)
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # Optimizer Step
                model.step()
                progress_bar.update(1)

            losses.append(loss.item())
            if step % log_every == 0:
                logger.info("Loss: {0:.4f}".format(np.mean(losses)))
                if global_rank == 0:
                    summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)
            if step % checkpoint_every == 0:
                model.save_checkpoint(save_dir=checkpoint_dir, client_state={"checkpoint_step": step})
                logger.info(f"Saved model to {checkpoint_dir}")

        model.eval()
        if model_parallel_rank == 0:
            total_num = 0
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    for k, v in batch.items():
                        batch[k] = v.to(device)
                    _, logits = model(**batch)
                    predictions = logits.argmax(dim=-1)
                    labels = batch["labels"]
                    total_num += len(labels)
                    metric.add_batch(
                        predictions=predictions,
                        references=labels,
                    )
            eval_metric = metric.compute()
            logger.info(f"total_num: {total_num}, on epoch {e} accuracy: {eval_metric}")
            if global_rank == 0:
                summary_writer.add_text("Accuracy", f"total_num: {total_num}, on epoch {e} accuracy: {eval_metric}")


if __name__ == "__main__":
    fire.Fire(main)

"""
CC:  
deepspeed train.py --checkpoint_dir /scratch/aidream/trained --model_name_or_path /scratch/aidream/roberta-large --train_file /scratch/aidream/SST-2/train.json --validation_file /scratch/aidream/SST-2/dev.json --batch_size 16
CLIO:  
deepspeed train.py --checkpoint_dir trained --model_name_or_path roberta-large --train_file SST-2/train.json --validation_file SST-2/dev.json --batch_size 16
batch size 108
'accuracy': 0.948394495412844
"""
