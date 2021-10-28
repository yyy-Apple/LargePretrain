# DeepSpeed

## Some notes
1. We cannot use `export CUDA_VISIBLE_DEVICES=1,2` or anything like that. Instead, we should use
> References: <br>
> [1] https://github.com/microsoft/DeepSpeed/issues/662<br>
> [2] https://zhuanlan.zhihu.com/p/256236705
```bash
deepspeed --include localhost:1,2 bert.py
```

The example of training a model is in `bert_train.py`, and the example of loading a trained model is in `bert_infer.py`
