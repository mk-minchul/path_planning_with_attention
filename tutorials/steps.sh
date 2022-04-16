tensorboard --logdir ./ --port 8083 --bind_all
python train.py
python simulate_trained.py --experiment_path experiments/training_raster_experiment_long/2022.03.16.18.05.27/checkpoints/epoch=9.ckpt
python run_vis.py
