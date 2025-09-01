# In conda env 
eval "$(/home/ens/AT74470/miniconda3/bin/conda shell.bash hook)"

conda create -n dmd2 python=3.8 -y 
conda activate dmd2 

which python

pip install --upgrade anyio


# For CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
python setup.py  develop
pip install lpips
# 
# add the submodule (no installs happen here)
git submodule add https://github.com/openai/guided-diffusion.git third_party/dhariwal

# make it importable, but don't touch your env's deps
python -m pip install -e third_party/dhariwal --no-deps --no-build-isolation