# In conda env 

conda create -n dmd2 python=3.8 -y 
conda activate dmd2 
pip install --upgrade anyio

# For CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
python setup.py  develop