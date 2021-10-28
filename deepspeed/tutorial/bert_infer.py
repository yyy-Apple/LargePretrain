"""
Say we use 2 GPUs, then after training, we will have two directories saving each state
e.g.
bert_pretrain/
    bert_pretrain.2021.10.28.6.48.16.judd6wc3/
    bert_pretrain.2021.10.28.6.48.16.uwzuv97b/
We only need bert_pretrain to load weights
"""
import torch
import loguru
import fire
import deepspeed
import pathlib
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM

logger = loguru.logger


def main(
        model_name_or_path: str = None,
        checkpoint_dir: str = None,
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

    model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=AutoConfig.from_pretrained(model_name_or_path)
    )
    logger.info("Model created")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    logger.info("Tokenizer created")
    max_seq_length = tokenizer.model_max_length
    ds_config = {
        "train_micro_batch_size_per_gpu": 256,
        "fp16": {
            "enabled": True
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu"
            }
        }
    }

    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    logger.info("DeepSpeed Engine Created")
    _, client_state = model.load_checkpoint(load_dir=load_checkpoint_dir)
    logger.info("Model weights loaded")
    batch = tokenizer(
        ["This is a sentence for testing"],
        truncation=True,
        max_length=max_seq_length,
        padding=True,
        return_tensors="pt"
    )
    for k, v in batch.items():
        batch[k] = v.to(device)

    logger.info(f"batch: {batch}")
    outputs = model(**batch)
    logger.info(f"outputs: {outputs}")


if __name__ == "__main__":
    fire.Fire(main)

"""
deepspeed --include localhost:1 bert_infer.py --checkpoint_dir bert_pretrain --model_name_or_path bert-base-uncased
"""
