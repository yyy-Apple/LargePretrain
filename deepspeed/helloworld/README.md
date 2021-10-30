# DeepSpeed
This is the "hello world" programs for DeepSpeed, with a MLM pre-training script.
## Some notes
1. We cannot use `export CUDA_VISIBLE_DEVICES=1,2` or anything like that. Instead, we should use
> References: <br>
> [1] https://github.com/microsoft/DeepSpeed/issues/662<br>
> [2] https://zhuanlan.zhihu.com/p/256236705
```bash
deepspeed --include localhost:1,2 bert.py
```

The example of training a model is in `bert_train.py`. Although it's better to have a unified checkpoint dir for all processes for later convinient loading. See examples in `classification` folder.

To train, run
```bash
deepspeed --include localhost:1,2 bert_train.py --checkpoint_dir bert_pretrain --model_name_or_path bert-base-uncased --train_file train.json --validation_file val.json --batch_size 128
```
