#!/bin/bash

python3 ./train.py --batch_sz=20 --last_layer_num_neurons=512 --n_epochs=100
python3 ./train.py --batch_sz=20 --last_layer_num_neurons=512 --n_epochs=100

python3 ./train.py --batch_sz=20 --last_layer_num_neurons=512 --n_epochs=40
python3 ./train.py --batch_sz=20 --last_layer_num_neurons=512 --n_epochs=40

python3 ./train.py --batch_sz=10 --model=CNN1 --n_epoch=40
python3 ./train.py --batch_sz=30 --model=CNN1 --n_epoch=40

python3 ./train.py --batch_sz=10 --model=CNN2 --n_epoch=40
python3 ./train.py --batch_sz=30 --model=CNN2 --n_epoch=40

python3 ./train.py --batch_sz=10 --model=MobileNetV2 --n_epoch=40
python3 ./train.py --batch_sz=30 --model=MobileNetV2 --n_epoch=40

python3 ./train.py --batch_sz=10 --model=efficientnet_b0 --n_epoch=40 --last_layer_num_neurons=1280
python3 ./train.py --batch_sz=30 --model=efficientnet_b0 --n_epoch=40 --last_layer_num_neurons=1280


# python3 ./train.py --batch_sz=5 --learning_rate=0.001 --last_layer_num_neurons=512,64
# python3 ./train.py --batch_sz=5 --learning_rate=0.0001 --last_layer_num_neurons=512,64
# python3 ./train.py --batch_sz=5 --learning_rate=0.001 --last_layer_num_neurons=512

# python3 ./train.py --batch_sz=5 --learning_rate=0.01 --model=CNN1


# python3 ./train.py --batch_sz=5 --learning_rate=0.01 --model=MobileNetV2
# python3 ./train.py --batch_sz=5 --learning_rate=0.01 --model=MobileNetV2
# python3 ./train.py --batch_sz=5 --learning_rate=0.01 --model=MobileNetV2
