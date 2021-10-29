import fire
import loguru
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

logger = loguru.logger


def main(
        model_name_or_path: str = None,
        validation_file: str = None,
        batch_size: int = 8,
        device: str = "cuda:0"
):
    ######### Create Dataset #########
    data_files = {}
    if validation_file is None:
        logger.error("Need to specify validation_file")
        return

    data_files["validation"] = validation_file
    raw_datasets = load_dataset("json", data_files=data_files)

    checkpoint = torch.load(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large")
    model.load_state_dict(checkpoint)
    model.to(device)
    logger.info("Model Loaded")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    logger.info("Tokenizer created")
    max_seq_length = tokenizer.model_max_length

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
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    logger.info("DeepSpeed Engine Created")
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
python infer.py --model_name_or_path trained/pytorch_model.bin --validation_file SST-2/dev.json
"""
