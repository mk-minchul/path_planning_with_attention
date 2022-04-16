pip install -e .
pip install scipy==1.8.0
pip install pytorch-lightning==1.5.10
export NUPLAN_DATA_ROOT="/data/data/autod"
export PATH=$HOME/.local/bin:$PATH


tensorboard --logdir ./ --port 8084 --bind_all