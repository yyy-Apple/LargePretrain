# Classification

This is mostly for debug use. We use a `roberta-large` and see its performance to make sure our deepspeed training has not problems.

To run training:
```bash
deepspeed --include localhost:1,2 train.py --checkpoint_dir trained --model_name_or_path roberta-large --train_file SST-2/train.json --validation_file SST-2/dev.json --batch_size 16
```

To run evaluation, there are two ways. We can either
* **Method 1** Transform all the weight checkpoint files (`.pt` files) into a `pytorch_model.bin` using the script provided by DeepSpeed `zero_to_fp32.py`.
* **Method 2** Use DeepSpeed as well.

## Method 1
To transform the weights, we need to run
```bash
# trained is a checkpoint folder got using DeepSpeed, contains latest, etc.
python zero_to_fp32.py trained trained/pytorch_model.bin
```
Then we can use our script to evaluate. Seems that there are some accuracy decrease (maybe due to precision problem?)
```bash
python infer_method1.py --model_name_or_path trained/pytorch_model.bin --validation_file SST-2/dev.json
```

## Method 2

