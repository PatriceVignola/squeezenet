import os
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

def _parse_args():
    parser = argparse.ArgumentParser("run_squeezenet.py")
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
        default=5000)

    parser.add_argument(
        '--batch_size',
        default=256)

    return parser.parse_args()

def main():
    with tf.compat.v1.Session(
            graph=tf.Graph(),
            config=tf.compat.v1.ConfigProto()) as sess:
        args = _parse_args()

        model = tf.compat.v2.saved_model.load(os.path.join(args.model_dir, "models", "0"))
        labels_path = os.path.join(args.model_dir, "labels.txt")
        model_labels = np.array(open(labels_path).read().splitlines())
        model_labels = [label.split(":")[1] for label in model_labels]

        sample_label_pairs = []

        for (dirpath, _, filenames) in os.walk(args.image_dir):
            sample_label_pairs.extend([(os.path.join(dirpath, filename), os.path.basename(os.path.normpath(dirpath))) for filename in filenames])

        sample_label_pairs = random.sample(sample_label_pairs, args.sample_size)

        images = None
        image_batches = []
        label_batches = []
        current_labels = []

        for index, (sample, label) in enumerate(sample_label_pairs):
            image = plt.imread(sample)
            image = np.expand_dims(image, axis=0) * 255

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

        predict = model.signatures["predict"]

        good_prediction_count = 0

        for index, (images_batch, labels_batch) in enumerate(zip(image_batches, label_batches)):
            print(f"Testing batch {index}...")

            squeezenet = predict(images=tf.constant(images_batch, dtype=tf.float32),
                                is_training=tf.constant(False))

            sess.run(tf.compat.v1.global_variables_initializer())
            predictions = sess.run(squeezenet)["predictions"]

            for prediction, expected_label in zip(predictions, labels_batch):
                if model_labels[prediction] == expected_label:
                    good_prediction_count += 1

        print(f"Test Accuracy: {good_prediction_count / args.sample_size}")


if __name__ == '__main__':
    main()
