import os
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

def _parse_args():
    parser = argparse.ArgumentParser("test_squeezenet.py")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory of the checkpoint to run inference on.")

    parser.add_argument(
        "--image_dir",
        required=True,
        help="Path to the folder of the images to run inference on.")

    parser.add_argument(
        '--data_format',
        default='NCHW',
        choices=['NCHW', 'NHWC'])

    parser.add_argument(
        '--sample_size',
        type=int,
        default=5000)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--network',
        required=True,
        choices=['squeezenet_cifar', 'squeezenet_tiny'],
        help="Network that the model was trained with (e.g. squeezenet_cifar)")

    return parser.parse_args()

def run_inference(args, class_labels, sample_label_pairs):
    images = None
    image_batches = []
    label_batches = []
    current_labels = []

    for index, (sample, label) in enumerate(sample_label_pairs):
        image = plt.imread(sample)
        image = np.expand_dims(image, axis=0)

        # Transform grayscale images into RGB color space
        if image.ndim == 3:
            image = np.expand_dims(image, axis=3)
            image = np.concatenate((image, image, image), axis=3)

        if args.data_format == "NCHW":
            image = np.transpose(image, [0, 3, 1, 2])

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        current_labels.append(label)

        if (index + 1) % args.batch_size == 0:
            image_batches.append(images)
            label_batches.append(current_labels)
            images = None
            current_labels = []

    if images is not None:
        image_batches.append(images)
        label_batches.append(current_labels)

    good_prediction_count = 0

    with tf.compat.v1.Session(
            graph=tf.Graph(),
            config=tf.compat.v1.ConfigProto()) as sess:
        model = tf.compat.v2.saved_model.load(os.path.join(args.model_dir, "models", "0"))
        predict = model.signatures["predict"]

        for index, (images_batch, labels_batch) in enumerate(zip(image_batches, label_batches)):
            print(f"Testing batch {index}...")

            images_batch = tf.image.convert_image_dtype(images_batch, tf.float32)
            squeezenet = predict(images=images_batch,
                                 is_training=tf.constant(False))

            sess.run(tf.compat.v1.global_variables_initializer())
            predictions = sess.run(squeezenet)["predictions"]

            for prediction, expected_label in zip(predictions, labels_batch):
                if class_labels[prediction] == expected_label:
                    good_prediction_count += 1

    print(f"Test Accuracy: {good_prediction_count / args.sample_size}")

def test_squeezenet_cifar(args):
    args = _parse_args()
    labels_path = os.path.join(args.model_dir, "labels.txt")
    class_labels = np.array(open(labels_path).read().splitlines())
    class_labels = [label.split(":")[1] for label in class_labels]

    sample_label_pairs = []

    for (dirpath, _, filenames) in os.walk(args.image_dir):
        sample_label_pairs.extend([(os.path.join(dirpath, filename), os.path.basename(os.path.normpath(dirpath))) for filename in filenames])

    sample_label_pairs = random.sample(sample_label_pairs, args.sample_size)
    run_inference(args, class_labels, sample_label_pairs)

def test_squeezenet_tiny(args):
    val_annotations_dir = os.path.split(args.image_dir)[0]
    val_annotations_path = os.path.join(val_annotations_dir, "val_annotations.txt")
    val_annotations_lines = np.array(open(val_annotations_path).read().splitlines())

    class_labels_dir = os.path.split(val_annotations_dir)[0]
    class_labels_path = os.path.join(class_labels_dir, "wnids.txt")
    class_labels = np.array(open(class_labels_path).read().splitlines())

    image_label_map = {}

    for annotation_line in val_annotations_lines:
        line_parts = annotation_line.split()
        image_name = line_parts[0]
        label = line_parts[1]
        image_label_map[image_name.lower()] = label

    sample_label_pairs = []

    for (dirpath, _, filenames) in os.walk(args.image_dir):
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            sample_label_pairs.append((image_path, image_label_map[filename.lower()]))

    sample_label_pairs = random.sample(sample_label_pairs, args.sample_size)
    run_inference(args, class_labels, sample_label_pairs)

def main():
    args = _parse_args()

    if args.network == "squeezenet_tiny":
        test_squeezenet_tiny(args)
    else:
        test_squeezenet_cifar(args)

if __name__ == '__main__':
    main()
