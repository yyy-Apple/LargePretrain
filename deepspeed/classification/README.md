# Classification

This is mostly for debug use. We use a `roberta-large` and see its performance to make sure our deepspeed training has not problems.

To run training:
```bash
deepspeed --include localhost:1,2 train.py --checkpoint_dir trained --model_name_or_path roberta-large --train_file SST-2/train.json --validation_file SST-2/dev.json --batch_size 16
```

To run evaluation:
```bash

```

# TODO:
Understand how to load a model checkpoint

