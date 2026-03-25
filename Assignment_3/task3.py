from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mnist_loader import load_mnist
from sklearn.svm import SVC


def flaten_images(train_images, test_images):
    train_images = train_images.reshape(60000, 784)
    test_images = test_images.reshape(10000, 784)
    return train_images, test_images

train_images, train_labels, test_images, test_labels = load_mnist()
train_images, test_images = flaten_images(train_images, test_images)

#Standarize Data
scaler = StandardScaler()
scaler.fit(train_images)
standarized_train_images = scaler.transform(train_images)
standarized_test_images = scaler.transform(test_images)

#Dimensionality reduction to 50 features
pca = PCA(n_components=50)
pca.fit(standarized_train_images)
train_images_pca_50 = pca.transform(standarized_train_images)
test_images_pca_50 = pca.transform(standarized_test_images)

#Dimensionality reduction to 100 features
pca = PCA(n_components=100)
pca.fit(standarized_train_images)
train_images_pca_100 = pca.transform(standarized_train_images)
test_images_pca_100 = pca.transform(standarized_test_images)

#Dimensionality reduction to 200 features
pca = PCA(n_components=200)
pca.fit(standarized_train_images)
train_images_pca_200 = pca.transform(standarized_train_images)
test_images_pca_200 = pca.transform(standarized_test_images)

svc = SVC(kernel="linear",C=1)












