# 0. Initialize
## Prepare two tabs, one for train, one for transfering data and tensorboard
## AMI: Ubuntu deeplearning ami 
## Storage: 100gib


# 1. Initial Setting for both tabs
export SERVER_NAME=virginia-dl
export SERVER_NAME=ohio-dl
export SERVER_NAME=oregon-dl
export SERVER_NAME=canada-dl
export SERVER_NAME=london-dl
export SERVER_NAME=frankfurt-dl

## Access to server in first tab
ssh ${SERVER_NAME}

## Deeplearning EC2 Setup
sudo locale-gen ko_KR.UTF-8
sudo apt-get install tmux unzip
source activate tensorflow_p36
pip install --upgrade pip


# 2. Send Data to server in second tab
## local to remote
cd ~/Desktop/gatten/
scp -r dish_server/* ${SERVER_NAME}:/home/ubuntu


# 3. Unzip data in first tab
unzip *.zip
rm -rf data.zip
exit


# 4. Reconnect EC2 for trainning in first tab
ssh ${SERVER_NAME}

## change data if needed
## vim dish.py
tmux new -s train

source activate tensorflow_p36
pip install imgaug opencv-python
python3 dish.py train --dataset=${PWD}/data --weights=coco --pairs BACKBONE=resnet101


# 5. Run TensorBoard in second tab
ssh -L 6006:localhost:6006 ${SERVER_NAME}

source activate tensorflow_p36
tensorboard --logdir=${PWD}/logs


# Etc. Run in tensorflow docker
cd ~/Desktop/gatten/Gatten_sushi_dishi_detection
nvidia-docker run -d -p 8888:8888 -p 6006:6006 -e PASSWORD=1111 --name board -v ${PWD}:/notebooks/works tensorflow/tensorflow:latest-gpu-py3
nvidia-docker exec -it board bash

cd works
pip install scikit-image==0.13.1 imgaug opencv-python
apt-get install -y libsm6 libxext6 libxrender-dev
tensorboard --logdir=${PWD}/models
