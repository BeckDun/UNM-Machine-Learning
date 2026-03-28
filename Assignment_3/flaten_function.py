
def flaten_images(train_images, test_images):
    train_images = train_images.reshape(60000, 784)
    test_images = test_images.reshape(10000, 784)
    return train_images, test_images