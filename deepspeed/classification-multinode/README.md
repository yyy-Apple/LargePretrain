# Multi-node Training
This is the folder that documents how to perform multi-node training on a [Slurm cluster](https://slurm.schedmd.com/documentation.html). All the scripts are stilled based on what we've learned in the [`classification`](../classification) example. Required packages are in [`setup.sh`](setup.sh).
> References: <br>
> [1] https://nlp.stanford.edu/mistral/tutorials/deepspeed.html

We will first create a hostfile. It should look like this:
```
node_1_ip slots=4
node_2_ip slots=4
```
After asking for available nodes, we can write a hostfile based on what we have, see [run.sh](run.sh) for details. One thing to notice is that you may need to use `chmod 777 path/to/bin/deepseed` in case you got permission denied error. Use the following command to run training
```bash
sbatch --account=rrg-bengioy-ad run.sh

UserWarning: Failed to initialize NumPy: numpy.core.multiarray failed to import (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:68.)

```

Please refer to [`run.sh`](run.sh) and [`train.py`](train.py) for further details.