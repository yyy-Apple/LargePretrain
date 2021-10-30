# Multi-node Training for seq2seq
This is the folder that documents how to perform multi-node training on a [Slurm cluster](https://slurm.schedmd.com/documentation.html). Required packages are in [`setup.sh`](setup.sh). We will run a seq2seq example using `t5-11b` as an example. 

More details about multi-mode training can refer to [`classification-multinode`](../classification-multinode) folder.

After asking for available nodes, we can write a hostfile based on what we have, see [run.sh](run.sh) for details. One thing to notice is that you may need to use `chmod 777 path/to/bin/deepseed` in case you got permission denied error. Use the following command to run training
```bash
sbatch --account=hahahaha run.sh
```

## Some Pitfalls
* Since `t5-11b` is a super large model. Even if you have 4GPUs, you may only run on 2 of them due to limited CPU memories you have.
* Or even worse, you need to do [Model Parallelism]() to fit the model into your GPUs. See [`LM-Hub`](../LM-Hub) for detailed examples.
