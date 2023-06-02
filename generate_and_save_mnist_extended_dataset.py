import tensorflow as tf

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# print(train_images.shape, train_labels.shape)
# print(test_images.shape, test_labels.shape)

import numpy as np
np.random.seed(seed=42)

from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset, display_segmented_image,
                                                                       display_grayscale_array, plot_class_masks)

train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=900,
                                                                        num_test_samples=100,
                                                                        image_shape=(64, 64),
                                                                        num_classes=5)


test_y = test_y > 0
train_y = train_y > 0

train_x = train_x.astype(dtype = "float16")
train_y = train_y.astype(dtype = "float16")
test_x = test_x.astype(dtype = "float16")
test_y = test_y.astype(dtype = "float16")

import matplotlib.pyplot as plt 
plt.close("all")

plt.imshow(test_x[10,:,:,0], cmap = "gray")
plt.axis("off")


fig, axs = plt.subplots(nrows = 1, ncols = 5)
#fig.suptitle("Segmentierungsmasken")
for label in range(5):
  axs[label].imshow(test_y[10, :,:, label], cmap = "gray")
  axs[label].set_title("Label " + str(label))
  axs[label].axis("off")







flag_save_data = False

all_data = (train_x, train_y, test_x, test_y)

if flag_save_data == True:
    import pickle as pkl 
    pkl.dump(all_data, open("extended_mnist_train_test_data", "wb"))
    
