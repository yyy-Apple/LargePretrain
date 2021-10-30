import fire
import loguru
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (
AutoTokenizer,
AutoModelForSequenceClassification,
DataCollatorWithPadding,
AutoConfig
)
from torch.optim import AdamW
import deepspeed
import pathlib

logger = loguru.logger


def main(
        model_name_or_path: str = None,
        checkpoint_dir: str = None,
        validation_file: str = None,
        local_rank: int = -1
):
    device = (
        torch.device("cuda", local_rank)
        if (local_rank > -1) and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"local_rank: {local_rank}")
    load_checkpoint_dir = pathlib.Path(checkpoint_dir)
    assert load_checkpoint_dir.exists()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=AutoConfig.from_pretrained(model_name_or_path)
    )

    logger.info("Model created")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    logger.info("Tokenizer created")
    max_seq_length = tokenizer.model_max_length

    ######### Create Dataset #########
    data_files = {}
    if validation_file is None:
        logger.error("Need to specify validation_file")
        return

    data_files["validation"] = validation_file
    raw_datasets = load_dataset("json", data_files=data_files)

    # Get the label list
    label_list = raw_datasets["validation"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    # Preprocessing the datasets
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (examples["sentence"],)
        result = tokenizer(*texts, padding=False, max_length=max_seq_length, truncation=True)
        result["labels"] = [label_to_id[l] for l in examples["label"]]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = processed_datasets["validation"]
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=8)
    logger.info("Dataset prepared")

    ######### Load Checkpoint #########
    ds_config = {
        "train_micro_batch_size_per_gpu": 16,
        "zero_allow_untested_optimizer": True,
    }
    optimizer = AdamW(model.parameters())
    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        optimizer=optimizer
    )
    model.load_checkpoint(
        load_dir=load_checkpoint_dir,
        load_optimizer_states=False,
        load_lr_scheduler_states=False
    )
    logger.info("Model weights loaded")

    metric = load_metric("accuracy")

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
    logger.info(f"total_num: {total_num}, accuracy: {eval_metric}")


if __name__ == "__main__":
    fire.Fire(main)

"""
deepspeed --include localhost:1 infer_method2.py --checkpoint_dir trained --model_name_or_path roberta-large --validation_file SST-2/dev.json
total_num: 872, accuracy: {'accuracy': 0.944954128440367}
"""