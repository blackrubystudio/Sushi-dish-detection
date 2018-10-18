# 1. Initial Setting & Jupyter notebook
## Access to server
ssh -L 8888:localhost:8888 p3-tokyo-ml

## Deeplearning EC2 Setup
sudo locale-gen ko_KR.UTF-8
sudo apt-get install tmux unzip

## Fetch Dishi detection and mask rcnn file
git clone https://github.com/zaiyou12/Gatten_sushi_dishi_detection.git

## Run jupyter notebook
cd Gatten_sushi_dishi_detection
source activate tensorflow_p36
pip3 install --upgrade pip
jupyter notebook

# 2. Setting for Trainning model
## (after copy data to server)
ssh p3-tokyo-ml
cd Gatten_sushi_dishi_detection && unzip *.zip
rm -rf __MACOSX/ data.zip
tmux new -s train
source activate tensorflow_p36
pip install imgaug
pip install opencv-python

## Start Trainning
python3 dish.py train --dataset=${PWD}/data --weights=coco
python3 dish.py train --dataset=${PWD}/data --weights=last;mail -s 'Finished' zaiyou12@gmail.com; sudo shutdown now;

# 3. Run TensorBoard
ssh -L 6006:localhost:6006 p3-tokyo-ml
cd Gatten_sushi_dishi_detection
source activate tensorflow_p36
tensorboard --logdir=${PWD}/logs

# 9. Run in tensorflow docker
nvidia-docker run -it -p 8889:8888 -p 6007:6006 -v ${PWD}:~/notebooks/works tensorflow/tensorflow:latest-gpu-py3
pip install scikit-image imgaug opencv-python
apt-get install -y libsm6 libxext6 libxrender-dev
