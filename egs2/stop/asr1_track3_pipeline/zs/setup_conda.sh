git clone https://github.com/openai/whisper.git

conda create -n mywhisper python=3.8.15
conda activate mywhisper

cd whisper
pip install -e .

# install torch in advance
# and specify version 1.10.1 at requirements.txt
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
