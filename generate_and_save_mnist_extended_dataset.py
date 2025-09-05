# Modified code from LukeTonin?s repo https://github.com/LukeTonin/simple-deep-learning

import numpy as np
import matplotlib.pyplot as plt

from simple_deep_learning.mnist_extended.semantic_segmentation import (
    create_semantic_segmentation_dataset,
    display_segmented_image,
    display_grayscale_array,
    plot_class_masks,
)


def main(
    flag_save_data=False,
    n_training=900,
    n_test=100,
    image_shape=(64, 64),
    num_classes=5,
):
  """Generate a training and test dataset for segmenting overlapping handwritten digits

  Parameters
  ----------
  flag_save_data : bool, optional
      Whether to export the data in a binary file, by default False
  n_training : int, optional
      number of training images, by default 900
  n_test : int, optional
      number of testing/validation images, by default 100
  image_shape : tuple, optional
      shape of the overlapping handwritten digits, by default (64, 64)
  num_classes : int, optional
      number of classes from 0 to num_classes-1, by default 5
  """
    train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(
        num_train_samples=n_training,
        num_test_samples=n_test,
        image_shape=image_shape,
        num_classes=num_classes,
    )

    test_y = test_y > 0
    train_y = train_y > 0

    train_x = train_x.astype(dtype="float16")
    train_y = train_y.astype(dtype="float16")
    test_x = test_x.astype(dtype="float16")
    test_y = test_y.astype(dtype="float16")

    plt.close("all")

    plt.imshow(test_x[10, :, :, 0], cmap="gray")
    plt.axis("off")

    fig, axs = plt.subplots(nrows=1, ncols=5)
    # fig.suptitle("Segmentierungsmasken")
    for label in range(5):
        axs[label].imshow(test_y[10, :, :, label], cmap="gray")
        axs[label].set_title("Label " + str(label))
        axs[label].axis("off")

    all_data = (train_x, train_y, test_x, test_y)

    if flag_save_data == True:
        import pickle as pkl

        pkl.dump(all_data, open("extended_mnist_train_test_data", "wb"))


if __name__ == "__main__":
    flag_save_data = False
    n_training = 900
    n_test = 100
    image_shape = (64, 64)
    num_classes = 5

    np.random.seed(seed=42)

    main(
        flag_save_data=flag_save_data,
        n_training=n_training,
        n_test=n_test,
        image_shape=image_shape,
        num_classes=num_classes,
    )
