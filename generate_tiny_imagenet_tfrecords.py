"""Converts Tiny ImageNet data to TFRecords file format.

* Reference: TensowFlow example code for converting ImageNet data to TFRecords file format
    - Link: https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py

"""

# Imports
import numpy as np
from scipy.ndimage import imread
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow as tf
import pickle
import os

VALIDATION_RATIO = 0.3

# Directories
TINY_IMAGENET_DIRECTORY = "E:/squeezenet_training/tiny-imagenet-200/tiny-imagenet-200"
TRAIN_DIRECTORY = TINY_IMAGENET_DIRECTORY + "/train"
VALIDATION_DIRECTORTY = TINY_IMAGENET_DIRECTORY + "/val"
WNIDS_DIRECTORY = TINY_IMAGENET_DIRECTORY + "/wnids.txt"
WORDS_DIRECTORY = TINY_IMAGENET_DIRECTORY + "/words.txt"

TEST_IMAGE_DIRECTORY = VALIDATION_DIRECTORTY + "/images"
TEST_ANNOTATION = VALIDATION_DIRECTORTY + "/val_annotations.txt"

TFRECORD_DIRECTORY = "E:/squeezenet_training/tiny-imagenet-200/tfrecords/"
PICKLE_DIR = "E:/squeezenet_training/tiny-imagenet-200/tfrecords/tiny_imagenet.pickle"

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _onehot_encoder(label_list):
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_list.reshape(len(label_list), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    return onehot_encoded

def _process_image_files(tfrecord_dir, X, Y, Y_one_hot, Loc, num_label, shard_size=1000, prefix=''):
    """Processes and saves list of images as TFRecord.

    Args:
        - 
    Returns:
        no return
        
    """
    
    try: os.makedirs(tfrecord_dir)
    except: pass 
    
    num_shards = len(X) // shard_size
    num_shards += 0 if len(X) % shard_size == 0 else 1

    for idx in range(num_shards):
        X_shard = X[idx * shard_size: (idx + 1) * shard_size]
        Y_shard = Y[idx * shard_size: (idx + 1) * shard_size]
        Y_shard_one_hot = Y_one_hot[idx * shard_size: (idx + 1) * shard_size]
        Loc_shard = Loc[idx * shard_size: (idx + 1) * shard_size]
        
        output_filename = '%s-%.5d-of-%.5d.tfrecords' % (prefix, idx, num_shards)
        output_file = os.path.join(tfrecord_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        for i in range(len(X_shard)):
            label = int(Y_shard[i])            
            img_raw = X_shard[i].tostring()
            location_raw = Loc_shard[i].tostring()
            label_one_hot_raw = Y_shard_one_hot[i].tostring()
            (height, width, channels) = X_shard[i].shape

            example = _convert_to_example(height=height,
                                          width=width,
                                          channels=channels,
                                          num_label=num_label,
                                          label=label,
                                          label_one_hot_raw=label_one_hot_raw,
                                          img_raw=img_raw,
                                          location_raw=location_raw)

            writer.write(example.SerializeToString())

def _convert_to_example(height, width, channels, num_label, label, label_one_hot_raw, img_raw, location_raw):
    """Build an Example proto for an example.

    Args:
        -
    Returns:
        Example proto

    """
    channels = 3

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'channel': _int64_feature(channels),
        'label': _int64_feature(label),
        'label_depth': _int64_feature(num_label),

        'label_one_hot_raw': _bytes_feature(label_one_hot_raw),
        'image_raw': _bytes_feature(img_raw),
        'location_raw': _bytes_feature(location_raw)}))
    return example

def _read_train_validation_data(train_directory, train_id_list, num_class):
    element_list = ['X', 'Y', 'Loc']
    train_origin = {'X':[], 'Y': [], 'Loc': []}
    train = {'X':[], 'Y': [], 'O': [], 'Loc': []}
    valid = {'X':[], 'Y': [], 'O': [], 'Loc': []}
    # Read train_origin data and separate it into two pieces: train and validation
    train_dir_list = [os.path.join(train_directory, dir_) for dir_ in os.listdir(train_directory)]

    # Read train, valid
    for class_ in train_dir_list:    
        bbox_dir = class_ + "/" + os.path.basename(class_) + "_boxes.txt"
        image_dir = class_ + "/images"
        image_dir_list = [os.path.join(image_dir, dir_) for dir_ in os.listdir(image_dir)]

        with open(bbox_dir, "r") as bbox_f:
            bbox_list = bbox_f.readlines()

        image_to_bbox = {}
        for line in bbox_list:
            (image_file, x_pos, y_pos, w_pos, h_pos) = [w if i == 0 else int(w) for i, w in enumerate(line.split('\t'))]
            image_to_bbox[image_file] = [x_pos, y_pos, w_pos, h_pos]

        for image_dir in image_dir_list:
            train_origin['X'].append(imread(image_dir, mode='RGB'))
            train_origin['Y'].append(train_id_list.index(os.path.basename(class_)))
            train_origin['Loc'].append(image_to_bbox[os.path.basename(image_dir)])

    train_origin_list = [np.stack(train_origin[e]) for e in element_list]
    num_train_data = len(train_origin_list[0])
    rand = np.random.permutation(range(num_train_data))
    for idx in range(len(train_origin_list)):
        train_origin_list[idx] = train_origin_list[idx][rand]
        
    validation_ratio = VALIDATION_RATIO
    train_idx = round(num_train_data*(1-validation_ratio))
    train_list = []
    valid_list = []
    for part in train_origin_list:
        train_list.append(part[:train_idx])
    for part in train_origin_list:
        valid_list.append(part[train_idx:])        
    for i, e in enumerate(element_list):
        train[e] = train_list[i]
        valid[e] = valid_list[i] 
    
    train['O'] = _onehot_encoder(train['Y'])
    valid['O'] = _onehot_encoder(valid['Y'])
        
    return train, valid

def _read_test_data(test_image_directory, train_id_list, num_class):
    
    element_list = ['X', 'Y', 'Loc']
    test = {'X':[], 'Y': [], 'O': [], 'Loc': []}
    
    image_dir_list = [os.path.join(test_image_directory, dir_) for dir_ in os.listdir(test_image_directory)]
    with open(TEST_ANNOTATION, "r") as val_f:
        val_list = val_f.readlines()

    image_to_bbox = {}
    image_to_wnid = {}
    for line in val_list:
        (image_file, wnid, x_pos, y_pos, w_pos, h_pos) = [w if i == 0 or i == 1 else int(w) for i, w in enumerate(line.split('\t'))]
        image_to_bbox[image_file] = [x_pos, y_pos, w_pos, h_pos]
        image_to_wnid[image_file] = wnid

    for image_dir in image_dir_list:
        test['X'].append(imread(image_dir, mode='RGB'))
        test['Y'].append(train_id_list.index(image_to_wnid[os.path.basename(image_dir)]))
        test['Loc'].append(image_to_bbox[os.path.basename(image_dir)])
        
    for e in element_list:
        test[e] = np.stack(test[e])

    test['O'] = _onehot_encoder(test['Y'])
        
    return test

# ## Convert the raw Tiny ImageNet dataset into TFRecords

# Dictionaries
label_to_word = {}
train = {}
valid = {}
test = {}
data_dict = {}

# Dictionary for tiny imagenet
train_id_list = os.listdir(TRAIN_DIRECTORY) # len is 200
num_class = len(train_id_list)
with open(WORDS_DIRECTORY, "r") as words_f:
    line_lists = words_f.readlines()

for l in line_lists:
    wnid, word = l.split('\t')
    if wnid in train_id_list:
        label = train_id_list.index(wnid)
        word = str(label) + ": " + word
        label_to_word[label] = word 
data_dict['label_to_word']=label_to_word

train, valid = _read_train_validation_data(TRAIN_DIRECTORY,
                                           train_id_list,
                                           num_class)

test = _read_test_data(TEST_IMAGE_DIRECTORY, train_id_list, num_class)

for data_type in ['train', 'valid', 'test']:   
    for e in ['X', 'Y', 'O', 'Loc']:
        data_dict[data_type + '_' + e] = eval(data_type + '[\'' + e + '\']')

    _process_image_files(TFRECORD_DIRECTORY + data_type,
                   X=data_dict[data_type + '_' + 'X'],
                   Y=data_dict[data_type + '_' + 'Y'],
                   Y_one_hot=data_dict[data_type + '_' + 'O'],
                   Loc=data_dict[data_type + '_' + 'Loc'],
                   num_label=200,
                   shard_size=2000,
                   prefix=data_type)
    
data_dict['mean_RGB'] = train['X'].mean()

try: os.makedirs(os.path.dirname(pickle_save_path))
except: pass
with open(PICKLE_DIR, "wb") as pickle_f:
    pickle.dump(data_dict, pickle_f)
