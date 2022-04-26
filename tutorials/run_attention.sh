
#python train_long.py
python train_long.py --attention CBAM
python train_long.py --attention SE
python train_long.py --attention NL

#python simulate_trained.py --experiment_path experiments/training_raster_experiment_long/2022.03.29.04.48.25/best_model/epoch=6-step=6250.ckpt
#python simulate_trained.py --attention SE --experiment_path experiments/training_raster_experiment_long_se/2022.03.29.05.21.42/best_model/epoch=0-step=888.ckpt
#python simulate_trained.py --attention CBAM --experiment_path experiments/training_raster_experiment_long_cbam/2022.03.29.05.07.48/best_model/epoch=5-step=5333.ckpt

#python train_long.py --attention vit

#python train_long_vit.py --attention vit2