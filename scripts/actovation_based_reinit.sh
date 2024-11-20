
# for training adam continually with reinitiliazation with percentage (only reinitialize top 10% in each layer)
python train_with_reinitialization.py -optimizer adam -training_mode continual -re_percentage 0.1
python train_with_reinitialization.py -optimizer adam -training_mode continual -re_percentage 0.25
python train_with_reinitialization.py -optimizer adam -training_mode continual -re_percentage 0.5

# for training adam continually with reinitiliazation
python train_with_reinitialization.py -optimizer adam -training_mode continual -re_th 10.0

# for training adam continually with reinitiliazation with percentage and threshold (only reinitialize top 10% in each layer that have activation greater than threshold)
python train_with_reinitialization.py -optimizer adam -training_mode continual -re_percentage 0.1 -re_th 10.0

# for training adam continually with reinitiliazation only on input weights
python train_with_reinitialization.py -optimizer adam -training_mode continual -re_th 10.0 -re_outputs False

# for training adam continually with reinitiliazation only on output weights
python train_with_reinitialization.py -optimizer adam -training_mode continual -re_th 10.0 -re_inputs False