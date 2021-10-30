# Multi-node Training
This is the folder that documents how to perform multi-node training on a [Slurm cluster](https://slurm.schedmd.com/documentation.html). All the scripts are still based on what we've learned in the [`classification`](../classification) example. Required packages are in [`setup.sh`](setup.sh).
> References: <br>
> [1] https://nlp.stanford.edu/mistral/tutorials/deepspeed.html

We will first create a hostfile. It should look like this:
```
node_1_ip slots=4
node_2_ip slots=4
```
After asking for available nodes, we can write a hostfile based on what we have, see [run.sh](run.sh) for details. One thing to notice is that you may need to use `chmod 777 path/to/bin/deepseed` in case you got permission denied error. Use the following command to run training
```bash
sbatch --account=hahahaha run.sh
```

Please refer to [`run.sh`](run.sh) and [`train.py`](train.py) for further details.

## Some pitfalls
- Due to some strange issues on the cluster, I have to modify `deepspeed/launcher/multinode_runner.py` and do some additional initializations such as load modules etc. More specifically,
```python
exports = ""
for key, val in self.exports.items():
    exports += "export {}={}; ".format(key, val)
#### BEGIN CHANGES ####
loads = "module load python/3.8; module load scipy-stack; module load StdEnv/2020 gcc/9.3.0 cuda/11.4; module " \
        "load arrow/5.0.0; "
#### END CHANGES ####
deepspeed_launch = [
    exports,
    #### BEGIN CHANGES ####
    loads,
    #### END CHANGES ####
    "cd {};".format(os.path.abspath('.')),
    sys.executable,
    "-u",
    "-m",
    "deepspeed.launcher.launch",
    '--world_info={}'.format(self.world_info_base64),
    "--node_rank=%n",
    "--master_addr={}".format(self.args.master_addr),
    "--master_port={}".format(self.args.master_port)
]
```
