#!/bin/bash

echo "Starting the start.sh script..."

# Start TensorBoard
tensorboard --logdir=/usr/src/app/lightning_logs --port=6006 --bind_all &
echo "TensorBoard started."

# Run python script with parameters and model checkpoint directory 
echo "Starting Python script..."
python DistilBERT_MRPC_Script.py --checkpoint_dir models --learning_rate 5.89e-05 --weight_decay 0.0519 --batch_size 64
echo "Python script started"

# Keep the script running
sleep infinity