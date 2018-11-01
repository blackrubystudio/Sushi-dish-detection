export SERVER_NAME=virginia-dl
export SERVER_NAME=ohio-dl
export SERVER_NAME=oregon-dl
export SERVER_NAME=canada-dl
export SERVER_NAME=london-dl
export SERVER_NAME=frankfurt-dl


# 1. Check is it finished
ssh ${SERVER_NAME}

tmux attach -t train


# 2. Check results
ssh -L 6001:localhost:6006 ${SERVER_NAME}

source activate tensorflow_p36
tensorboard --logdir=logs

cd /Users/aaron/Desktop/Code/dish-detection/models
scp -r ${SERVER_NAME}:/home/ubuntu/logs/**/events* .


# 3. Prepare for restarting
ssh ${SERVER_NAME}

rm -rf logs
vim dish.py

## Change something if you want
## $ vim dish.py
## when in parell mode you must use 'keras==2.0.8' 
tmux new -s train

source activate tensorflow_p36
pip install imgaug opencv-python
python3 dish.py train --dataset=${PWD}/data --weights=coco --pairs BACKBONE=resnet101
