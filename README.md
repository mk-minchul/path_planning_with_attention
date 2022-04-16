
# Installation 


## Downloading Data

I download the following and stored at the YOUR_DATA_ROOT

```
Metadata for the mini split (v0.2)  [US, Asia] 5.40 GB(5792841728 Bytes)
Metadata for the mini split (v0.1)  [US, Asia] 5.39 GB(5792751616 Bytes) 
nuPlan maps, required for mini and full dataset  [US, Asia] 1.21 GB(1297742811 Bytes)
```

## Env Setup

```
pip install -e .
pip install scipy==1.8.0
pip install pytorch-lightning==1.5.10

export NUPLAN_DATA_ROOT="<YOUR_DATA_ROOT>"
export NUPLAN_ROOT="<YOUR REPO/nuplan ROOT>"
export PATH=$HOME/.local/bin:$PATH
```

- Example for Environment Variable for reference.
```
NUPLAN_ROOT=~/MSU/course/autonomous_vehicle/nuplan-devkit_v2/nuplan
NUPLAN_DATA_ROOT=/data/data/autod
```

# Running Experiments
The repo is structured as follows. 

## Run scripts
```
nuplan_devkit
├── ci              
├── docs            
├── nuplan          
│   ├── common      
│   ├── database    
│   └── planning    
└── tutorials       - Where all the training scripts are.
```
The training scripts are in `tutorials/`.
Check out below for running attention experiments.
```
cd tutorials/
bash run_attention.sh
```

## Network Architecture Change
The modification in the network architecture is in

```
nuplan_devkit
├── ci              
├── docs            
├── nuplan          
│   ├── common      
│   ├── database    
│   └── planning    - Where training codes are    
└── tutorials      
```

Some important files are:
```
nuplan/planning/script/config/common/model/  : contains yaml file for model arch. (attention configuartion is set here)
nuplan/planning/script/config/training/lightning/default_lightning.yaml  : default config related to training. 

nuplan/planning/training/modeling/models/rater_model.py  : where rater_model forward is determined.
nuplan/planning/training/modeling/models/resnet.py       : resnet redefined for attention.
nuplan/planning/training/modeling/models/attentions.py   : attention modules.
```