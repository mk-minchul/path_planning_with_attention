from pathlib import Path
import hydra
from nuplan.planning.script.run_simulation import main as main_simulation
import argparse
import os, sys

# os.environ['NUPLAN_DATA_ROOT'] = "/data/data/autod"
sys.path.append(os.path.join(os.path.expanduser('~'), '.local/bin'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--experiment_path', default='experiments/training_raster_experiment_long/2022.03.16.18.05.27/checkpoints/epoch=9.ckpt')
    parser.add_argument('--attention', type=str, default='')

    args = parser.parse_args()
    args.attention = "_" + args.attention.lower() if args.attention else ""

    # Location of path with all simulation configs
    CONFIG_PATH = '../nuplan/planning/script/config/simulation'
    CONFIG_NAME = 'default_simulation'

    CHALLENGE = 'challenge_1_open_loop_boxes'  # [challenge_1_open_loop_boxes, challenge_3_closed_loop_nonreactive_agents, challenge_4_closed_loop_reactive_agents]
    DATASET_PARAMS = [
        'scenario_builder=nuplan_mini',  # use nuplan mini database
        'scenario_builder/nuplan/scenario_filter=all_scenarios',  # initially select all scenarios in the database
        'scenario_builder.nuplan.scenario_filter.scenario_types=[nearby_dense_vehicle_traffic, ego_at_pudo, ego_starts_unprotected_cross_turn, ego_high_curvature]',  # select scenario types
        'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=10',  # use 10 scenarios per scenario type
        'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.05',  # subsample 20s scenario from 20Hz to 1Hz
    ]

    # Name of the experiment
    EXPERIMENT = args.experiment_path.split('/')[1].replace('training_', 'simulation_')
    # Get the checkpoint of the trained model
    MODEL_PATH = args.experiment_path
    MODEL_PATH = str(MODEL_PATH).replace("=", "\=")
    SAVE_DIR = str(Path('./') / 'experiments')  # optionally replace with persistent dir


    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'experiment_name={EXPERIMENT}',
        f'group={SAVE_DIR}',
        'planner=ml_planner',
        'model=raster_model{}'.format(args.attention),
        'planner.model_config=${model}',  # hydra notation to select model config
        f'planner.checkpoint_path={MODEL_PATH}',  # this path can be replaced by the checkpoint of the model trained in the previous section
        f'+simulation={CHALLENGE}',
        *DATASET_PARAMS,
    ])

    # Run the simulation loop
    main_simulation(cfg)

    # Fetch the filesystem location of the simulation results file for visualization in nuBoard (next section)
    parent_dir = Path(SAVE_DIR) / EXPERIMENT
    results_dir = list(parent_dir.iterdir())[0]  # get the child dir
    nuboard_file_2 = [str(file) for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]
    print(nuboard_file_2)

