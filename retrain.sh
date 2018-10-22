export SERVER_NAME=virginia-ml
export SERVER_NAME=ohio-ml
export SERVER_NAME=oregon-ml
export SERVER_NAME=canada-ml
export SERVER_NAME=london-ml
export SERVER_NAME=frankfurt-ml


# 1. Check is it finished
ssh ${SERVER_NAME}

tmux attach -t train


# 2. Check results
ssh -L 6001:localhost:6006 ${SERVER_NAME}

source activate tensorflow_p36
tensorboard --logdir=logs

cd /Users/aaron/Desktop/Code/dish-detection/models
scp -r ${SERVER_NAME}:/home/ubuntu/logs/dish20181021T0457/events* .
# scp -r ${SERVER_NAME}:/home/ubuntu/logs/dish20181021T0457/mask_rcnn_dish_0033.h5 .


# 3. Prepare for restarting
ssh ${SERVER_NAME}

rm -rf logs
vim dish.py

## Change something if u want
## vim dish.py
tmux new -s train

source activate tensorflow_p36
pip install imgaug opencv-python
python3 dish.py train --dataset=${PWD}/data --weights=coco --pairs BACKBONE=resnet101