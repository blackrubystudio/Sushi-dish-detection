# Dish Detection

## Shuffle Images

먼저 모든 이미지를 한 폴더에 넣고, 간단한 파이썬 코드를 활용하여 폴더 안에 있는 모든 이미지들의 순서를 섞은후 일정한 이름으로 재설정.

```bash
python3 shuffle_images.py --dataset ${PWD}/data/train
```

## Inspect Data

Inspect.ipynb 파일을 통해 Annotation 및 Config 파일 점검.

```bash
## run jupyter container
docker run -it --rm -p 8888:8888 -p 6666:6666 -v ${PWD}:/home/jovyan/work jupyter/tensorflow-notebook

## or run jupyter in local
jupyter notebook
```