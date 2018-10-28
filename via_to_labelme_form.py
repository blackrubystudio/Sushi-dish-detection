## Via saves all images in the form
# filename, filesize, file_attributes, region_count, region_id, region_shape_attributes
# image.jpg, 123456, {}, num, num, {"name":"string","all_points_x":[],"all_points_y":[]}


## LabelImg saves each image in the form
# { 'flags': {},
#   'shapes': [
#     {
#       'label': 'string',
#       'points': 
#       [
#         [
#           y0, x0
#         ],
#         [
#           y1, x1
#         ],
#         ...
#       ]
#     },
#     ... more regions ...
#   ],
#   'imagePath': '/path/to/img'
# }
# (left top is (0, 0))

import os
import json
import shutil
from collections import namedtuple

import pandas as pd

CSV_FILE_PATH = os.path.join(os.getcwd(), 'path/to/csv_data.csv')
IMAGE_DIR_PATH = os.path.join(os.getcwd(), 'path/to/image/dir')
SAVE_DIR_PATH = os.path.join(os.getcwd(), 'to/save/dir')
INDEX_NUM = 0
LABEL_MAP = ['green', 'red', 'purple', 'navy', 'silver', 'gold', 'black']


def group_by_raw(data_frame, row_name='filename'):
    DataGroup = namedtuple('data', [row_name, 'object'])
    grouped_data = data_frame.groupby(row_name)
    return [DataGroup(filename, grouped_data.get_group(x)) for filename, x
                     in zip(grouped_data.groups.keys(), grouped_data.groups)]


def get_true_attribute(data, filename):
    for key, value in data.items():
        if value == True:
            return int(key)
    print('No true index in attribute ' + filename)


def item_to_json(label, points):
    json_item = {
        'label': label,
        'line_color': None,
        'fill_color': None,
        'points': points
    }
    return json_item


def items_to_json_file(shape_data, image_name, filename, save_dir_path=SAVE_DIR_PATH):
    data = {
        'flags': {},
        'shapes': shape_data,
        'imagePath': image_name,
        'lineColor' : [0, 255, 0, 128],
        'fillColor' : [255, 0, 0, 128],
        'imageData': None
    }
    file_path = os.path.join(save_dir_path, filename)
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)


examples = pd.read_csv(CSV_FILE_PATH)
grouped = group_by_raw(examples, 'filename')


for index, group in enumerate(grouped):
    # data to json format
    shape_list = []
    for index, row in group.object.iterrows():
        try:
            region_json = json.loads(row['region_attributes'])['dish']
            label = LABEL_MAP[int(get_true_attribute(region_json, group.filename)) - 1]
            points_json = json.loads(row['region_shape_attributes'])
            points_list = []
            for y, x in zip(points_json['all_points_y'], points_json['all_points_x']):
                points_list.append([x, y])
            shape_list.append(item_to_json(label, points_list))
        except as e:
            print(e, group.filename)
    # save image and data in to new folder
    items_to_json_file(shape_list, 'dish_{}.jpg'.format(INDEX_NUM), 'dish_{}.json'.format(INDEX_NUM))
    src_img_path = os.path.join(IMAGE_DIR_PATH, group.filename)
    dir_img_path = os.path.join(SAVE_DIR_PATH, 'dish_{}.jpg'.format(INDEX_NUM))
    shutil.copy(src_img_path, dir_img_path)
    INDEX_NUM += 1