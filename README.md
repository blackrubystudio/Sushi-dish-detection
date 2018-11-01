# Mask-RCNN Sushi Dish Detection

<p align="center">
    <img src="./image/dish_with_color_splash.png" alt="gatten dish detection with color splash" width="300"/>
</p>

## Introduction

The following is a summary of the "Sushi Dish Detection Project" in October 2018. If there is a mistake or a better way, please let us know by e-mail(jae.woo@blackrubystudio.com) or Github. We also hope that this summary will give some hints to others who are learning deep learning just like we do.

At the Korean branch of the Gatten Sushi(belt sushi restaurant), there was a suggestion to make a program that can easily measure the price of the sushi dishes that customers have eaten. At that time, we did not know that this was possible in reality, because dishes were piled up so that there would not be many exposed parts. Also, it has a direct impact on price, there can be a mistake that customers are overcharged or under-charged.

<p align="center">
    <img src="./image/gatten_stack_dishes.jpg" alt="gatten dishes" width="300"/>
</p>

> Is it possible? :grey_question:

There was other negative feedback, but for the pleasure of applying technology in real-life, we only received dishes from the Gatten Sushi and started the project. Tasks that are not needed are marked with star.

## Steps

### 1. Gather and Label Pictures

Before the dish arrives, we went to Gatten restaurant and took 50 pictures of dishes, and collected about 100 pictures from Google (Most of the pictures from the internet are heavily filtered and later exclude them from all training). As a simple toy project, 150 images showed sufficient accuracy.

#### 1a. Gather images from google

[Google Images Download](https://github.com/hardikvasa/google-images-download)

With above open source, we can easily search and download Google images to the local hard disk.

#### 1b. Gather images by ourself

After the dish arrived, we took pictures in various ways. We continuously added pictures to improve accuracy, and took about 1700 pictures at the end of the project.

#### 1c. Shuffle images *

We put all the images in one folder, use simple python code to sequence all the images in the folder, then reset them to a specific name.

```bash
python3 shuffle_images.py --dataset ${PWD}/data/train
```

#### 1d. Resize images *

When we first started, we compressed all the images for quick training. After that, we used only the original image to vary the image size in various ways. [Image resize code we refer to](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/resizer.py)

#### 1e. Image Labeling

The most difficult step. We made labels on the image to train. Inside the team, we used [Labelme](https://github.com/wkentaro/labelme), which has many shortcut keys. When we asked our friends for image labeling, we asked them to use [Via](https://www.robots.ox.ac.uk/~vgg/software/via/), which does not require installation.

```bash
## Recommended labelme config, autosave & nodata
labelme --autosave --nodata
```

<p align="center">
    <img src="./image/labelme-1.png" alt="labelme screenshot with blanket" width="300"/>
    <img src="./image/labelme-2.png" alt="labelme screenshot with other dishes" width="300"/>
</p>

> Labelme screenshot. (To take a variety of pictures in the same place, we used blankets and books with a colored cover. We also included other common dishes to test our program)

#### 1f. Synthetic Data *

We were tired of manually labeling, we found this by accident. Synthetic Data is a method to randomly generate fake data and to automatically extract the labeling. We were planning to use Unity to place 3d modeled dishes at random, but could not find any way to extract the labeling data. We decided to look it out when the project was over, but we did not figure it yet. ([Wired article about synthetic data](https://www.wired.com/story/some-startups-use-fake-data-to-train-ai/))

### 2. Set up Trainning Environment

In most cases we used "Deep learning AMI" in AWS. We did not ask for releasing the number limit of GPU, so we were forced to make one instance in various regions.

#### 2a. AWS p2 vs p3

We compared the two instances and found that p3 was the best fit for our situation.

Instance | GPU | GIB | Price | CPU times | Sys times | Wall time | Total time(s) | Total price(s)
-----|-----|-----|-----|-----|-----|-----|-----|-----|
p2.xlarge | 1 | 4 | 0.9 | 13min 13s | 3min 45s | 16min 58s | 1018 | 0.436043
p3.2xlarge | 1 | 16 | 3.06 | 4min 8s | 1min 4s | 5min 13s | 313 | 0.455849

#### 2a. Default Setting

```bash
# sudo locale-gen ko_KR.UTF-8
sudo apt-get install tmux unzip
source activate tensorflow_p36
pip install --upgrade pip
```

#### 2b. Set up for Tensorflow *

```bash
conda install pillow
## if you don't want to see scipy warning, better use 0.13.0
pip3 install -U scikit-image==0.13.0 cython

## if you need pycocotool
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

#### 2c. Set up for Mask RCNN *

```bash
pip install imgaug opencv-python
```

#### 2d. Local Setting for checking

We used the docker not installing Tensorflow-GPU locally, because it's enough to simply check the training results.

```bash
nvidia-docker run -d -p 8888:8888 -p 6006:6006 -e PASSWORD=1111 --name board -v ${PWD}:/notebooks/works tensorflow/tensorflow:latest-gpu-py3
nvidia-docker exec -it board bash

(docker) $ apt-get install -y libsm6 libxext6 libxrender-dev
(docker) $ pip install scikit-image==0.13.1 imgaug opencv-python
(docker) $ tensorboard --logdir=${PWD}/path/to/logs
```

#### 2e. Inspect Data *

We used Inspect.ipynb in [Mask R-CNN](https://github.com/matterport/Mask_RCNN) to check annotations.

```bash
nvidia-docker run -d -p 8889:8888 -e PASSWORD=1111 --name mask_rcnn -v ${PWD}:/notebooks/works tensorflow/tensorflow:latest-gpu-py3
```

### 3. 훈련 모델 조사

처음에는 모바일에서 딥러닝 모델을 구동시킬수 있는 Tensorflow lite를 사용해보고 싶은 마음에, Mobile과 real time 이라는 단어로 모델 조사.

#### 3a. Mobile에서 구동 가능한 모델 조사

- SSD: 손 인식 웹캠 기준, FPS 11~21의 성능, 모바일로는 부적격 https://towardsdatascience.com/how-to-build-a-real-time-hand-detector-using-neural-networks-ssd-on-tensorflow-d6bac0e4b2ce
- Tiny Yolo: 성능표 기준으로 뛰어난 성능이 나와 알아보고 싶었지만, 실제 예제가 없어 보류
- Squeeze Net: Tensorflow Lite 기준 Mobile로  테스트 결과 모델 사이즈는 5mb, 계산시간은 224ms, Top-1 정확도 49.0%, Top-5 정확도 72.9%, https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md
- Pelee: 모바일 딥러닝에 최적화된 모델. Tiny-YOLOv2와 SSD+MobileNet보다 성능이 약간 좋음, 용량도 5.4M로 작으나, Caffe 프레임워크 기반. Tensorflow 버전도 있으나 현재는 classification 만 있음.
- MobileNetv2. 구글이 18년 4월에 새롭게 발표한 모델. TensorFlow 기반이며 TensorFlow Lite와도 호완이 좋으며 용량은 3.4Mb, Top-1 정확도 70.8%, Top-5 정확도 89.9%, 637ms으로 나옴. (Squeeze Net 과 동일한 실험)

현재 프로젝트와 가장 적합한 방식은 Squeeze Net으로 판단됨. 다만 문서화의 정도로 MobileNetV2를 선택하여 우선적으로 개발하고, 속도상에 문제가 있다면 Squeeze Net으로 변경할 예정. 최후의 보루는 Pelee.

MobileNetV2로 약 300여장을 훈련한 결과, mIOU 50% 기준 정확도가 20% 정도 수준이여서 학습을 중단. 데이터 라벨링 코드를 tfrecord로 바꾸는 과정에서 실수를 했는지도 의심하였지만, 학습한 데이터를 보면서 박스의 문제가 있다고 생각하여, pixel-wise 모델로 변경.

#### 3b. Pixel-wise

- Mask-RCNN
- Tensorflow
- Caffe

Pixel-wise한 방식을 구현할수 있는 모듈이 많이 있었지만, Tensorflow 뿐만 아니라 AWS의 MX-net도 keras로 가능하다는 이야기를 듣고 keras로 되있는 Mask RCNN 선택(문서화가 잘되있었던 것도 한몫).

### 4. 훈련 준비 과정

#### 4a. Label map

#### 4b. Image 불러오기

- Image to Tfrecord
- Read Image from csv
- via json format to csv

#### 4c. Configure training

### 5. Run the Training

### 6. Result

## 참고 자료

- 123
- 123
- 123