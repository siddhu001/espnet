git clone https://github.com/huggingface/transformers

conda create -n myhuggingface python=3.8.15
conda activate myhuggingface

# install editable version (to implement copy-generate mechanism etc.)
# but non-editable and non-source version is OK
cd transformers
pip install -e .

# install torch
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# install datasets
pip install datasets==2.8.0
pip install nltk

# required to watch logs
pip install tensorboard
# to avoid AttributeError: module 'distutils' has no attribute 'version'
pip install setuptools==58.2.0

pip install evaluate
