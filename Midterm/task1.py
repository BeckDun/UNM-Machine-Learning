import matplotlib.pyplot as plt
import idx2numpy
train_image_file = 'train-images-idx3-ubyte'
train_labels_file = 'train-labels-idx1-ubyte'
train_images = idx2numpy.convert_from_file(train_image_file)
train_labels = idx2numpy.convert_from_file(train_labels_file)

# print(len(train_images))
# print(set(train_labels))
# plt.imshow(train_images[1])
# plt.show()


