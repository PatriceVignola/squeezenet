import os
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt

def _parse_args():
    parser = argparse.ArgumentParser("run_squeezenet.py")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory of the checkpoint to run inference on.")

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the 32x32 image to classify.")

    return parser.parse_args()

def main():
    with tf.compat.v1.Session(
            graph=tf.Graph(),
            config=tf.compat.v1.ConfigProto()) as sess:
        args = _parse_args()

        model = tf.compat.v2.saved_model.load(
                os.path.join(args.model_dir, "models", "0"))
        labels_path = os.path.join(args.model_dir, "labels.txt")
        labels = np.array(open(labels_path).read().splitlines())
        image = plt.imread(args.image)
        image = np.expand_dims(image, axis=0) * 255

        predict = model.signatures["predict"]

        squeezenet = predict(images=tf.constant(image, dtype=tf.float32),
                             is_training=tf.constant(False))

        sess.run(tf.compat.v1.global_variables_initializer())
        results = sess.run(squeezenet)["predictions"]

        label = labels[results[0]].split(":")[1]
        print(f"result: {label}")

if __name__ == '__main__':
    main()
