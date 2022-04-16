from pathlib import Path
from nuplan.planning.script.run_training import main as main_train
import hydra
from omegaconf import OmegaConf
import os, sys

# os.environ['NUPLAN_DATA_ROOT'] = "/data/data/autod"
# os.environ['NUPLAN_ROOT'] = "/mckim/MSU/course/autonomous_vehicle/nuplan-devkit_v2/nuplan"

sys.path.append(os.path.join(os.path.expanduser('~'), '.local/bin'))

if __name__ == '__main__':

    # Location of path with all training configs
    CONFIG_PATH = '../nuplan/planning/script/config/training'
    CONFIG_NAME = 'default_training'

    # save directory
    SAVE_DIR = Path('./') / 'experiments'  # optionally replace with persistent dir
    EXPERIMENT_NAME = 'training_raster_experiment_short'
    LOG_DIR = str(SAVE_DIR / EXPERIMENT_NAME)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'group={str(SAVE_DIR)}',
        f'cache_dir={str(SAVE_DIR)}/cache',
        f'experiment_name={EXPERIMENT_NAME}',
        'py_func=train',
        '+training=training_raster_model',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
        'model=raster_model_cbam',
        'scenario_builder=nuplan_mini',  # use nuplan mini database
        'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=500',  # Choose 500 scenarios to train with
        'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.01',  # subsample scenarios from 20Hz to 0.2Hz
        'lightning.trainer.params.accelerator=ddp',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
        'lightning.trainer.params.max_epochs=10',
        'lightning.optimization.optimizer.learning_rate=1e-3',
        'data_loader.params.batch_size=8',
        'data_loader.params.num_workers=0',
    ])

    # Run the training loop, optionally inspect training artifacts through tensorboard (above cell)
    print(OmegaConf.to_yaml(cfg))
    main_train(cfg)
    print('Training finished and artifacts stored in \n{}'.format(LOG_DIR))