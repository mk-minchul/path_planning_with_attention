import hydra
from nuplan.planning.script.run_nuboard import main as main_nuboard

if __name__ == '__main__':

    p1 = './experiments/simulation_raster_experiment_long/2022.03.17.13.50.44/nuboard_1647525047.nuboard'
    # Location of path with all nuBoard configs
    CONFIG_PATH = '../nuplan/planning/script/config/nuboard'
    CONFIG_NAME = 'default_nuboard'

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    # Compose the configuration
    # cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    #     'scenario_builder=nuplan_mini',  # set the database (same as simulation) used to fetch data for visualization
    #     f'simulation_path={[p1, p2]}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
    # ])

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        'scenario_builder=nuplan_mini',  # set the database (same as simulation) used to fetch data for visualization
        f'simulation_path={[]}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
    ])

    # Run nuBoard
    main_nuboard(cfg)