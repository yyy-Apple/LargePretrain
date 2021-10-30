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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)

logger = loguru.logger
world_size = int(os.getenv('WORLD_SIZE', '1'))
logger.info(f"world_size: {world_size}")


def create_experiment_dir(
    checkpoint_dir_path: pathlib.Path, all_arguments
):
    """ Create an experiment directory and save all arguments in it."""
    hparams_file = checkpoint_dir_path / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)

    # Create the Tensorboard Dir
    tb_dir = checkpoint_dir_path / "tb_dir"
    tb_dir.mkdir()


def main(
    checkpoint_dir: str = None,
    # Dataset Params
    train_file: str = None,
    validation_file: str = None,
    dataset_chache_dir: str = None,
    # Model Params
    model_name_or_path: str = None,
    # Training Params
    epoch: int = 1,
    batch_size: int = 32,
    checkpoint_every: int = 5000,
    weight_decay: float = 0.0,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 4,
    lr_scheduler_type: SchedulerType = "linear",
    num_warmup_steps: int = 0,
    log_every: int = 50,
    seed=666,
    local_rank: int = -1,
):
    """ Train a T5 style model for seq2seq generation """
    # First init process group
    deepspeed.init_distributed()

    global_rank = torch.distributed.get_rank()
    logger.info(f"[Global Rank {global_rank}] starts")

    # local_rank will be automatically decided, don't need us to manually specify

    device = (
        torch.device("cuda", local_rank)
        if (local_rank > -1) and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Global Rank: {global_rank} Local Rank: {local_rank}")
    assert checkpoint_dir is not None
    checkpoint_dir_path = pathlib.Path(checkpoint_dir)

    ########## Creating Experiment Directory ###########
    if global_rank != 0:
        torch.distributed.barrier()

    # Only allow rank 0 to create directory
    if global_rank == 0:
        logger.info("Creating Experiment Directory")
        checkpoint_dir_path.mkdir(exist_ok=True)
        all_arguments = {
            "task": "seq2seq",
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
        create_experiment_dir(checkpoint_dir_path, all_arguments)
        logger.info(f"Experiment Directory created at {checkpoint_dir_path}")
        tb_dir = checkpoint_dir_path / "tb_dir"
        assert tb_dir.exists()
        summary_writer = SummaryWriter(log_dir=tb_dir)

    if global_rank == 0:
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
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=dataset_chache_dir)
    logger.info(f"[Rank {global_rank}] Loaded datasets.")

    assert model_name_or_path is not None, "You need to specify model_name_or_path"
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    max_seq_length = min(tokenizer.model_max_length, 1024)
    # Add special tokens to token embeddings, restructure texts
    special_tokens_dict = {'additional_special_tokens': ["<text>", "</text>", "<prompt>", "</prompt>",
                                                         "<answer>", "</answer>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    logger.info(f"Tokenizer set, max_seq_length: {max_seq_length}")

    # Save the tokenizer
    torch.distributed.barrier()

    if global_rank == 0:
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Tokenizer saved at {checkpoint_dir}")

    torch.distributed.barrier()

    # Preprocessing the datasets
    def preprocess_function(examples):
        inputs = examples["text"]
        targets = examples["answer"]
        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=False, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_seq_length, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "answer"],
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

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
        rank=global_rank,
        seed=seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Sampler option is mutually exclusive with shuffle
        collate_fn=data_collator
    )

    # Actually do not use eval dataset at all.
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

    ####### DeepSpeed Engine ########
    logger.info("Creating DeepSpeed Engine")
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "fp16": {
            "enabled": True
        },
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 3,
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

    # Train !
    total_batch_size = batch_size * gradient_accumulation_steps * world_size

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {epoch}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=(global_rank != 0))

    losses = []
    model.train()
    for e in range(epoch):
        for step, batch in enumerate(train_dataloader):
            # Forward pass
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass, don't need to aggregate from other GPUs, since DeepSpeed will do this for us
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
                model.save_checkpoint(save_dir=checkpoint_dir_path, client_state={"checkpoint_step": step})
                logger.info(f"Saved model to {checkpoint_dir_path}")

    model.save_checkpoint(save_dir=checkpoint_dir_path, client_state={"checkpoint_step": step})
    logger.info(f"Saved model to {checkpoint_dir_path}")


if __name__ == "__main__":
    fire.Fire(main)
"""
deepspeed train.py --checkpoint_dir trained --dataset_chache_dir /scratch/aidream/datasets --model_name_or_path /scratch/aidream/t5-11b --train_file /scratch/aidream/train_enc_dec.json --validation_file /scratch/aidream/val_enc_dec.json --batch_size 1
"""
