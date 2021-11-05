pip3 install torch
pip3 install transformers
pip3 install datasets
pip3 install deepspeed
pip3 install fire
pip3 install tensorboard
pip3 install loguru
pip3 install sklearn


git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

export TORCH_CUDA_ARCH_LIST=Ampere