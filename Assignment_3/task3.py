from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mnist_loader import load_mnist
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def flaten_images(train_images, test_images):
    train_images = train_images.reshape(60000, 784)
    test_images = test_images.reshape(10000, 784)
    return train_images, test_images

train_images, train_labels, test_images, test_labels = load_mnist()
train_images, test_images = flaten_images(train_images, test_images)

pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50)), ("svc", SVC(kernel= "linear", C = 10))])
pipeline.fit(train_images,train_labels)

train_predictions = pipeline.predict(train_images)
test_predictions = pipeline.predict(test_images)

training_prediction_error = 1- accuracy_score(train_labels, train_predictions)
test_prediction_error = 1 - accuracy_score(test_labels, test_predictions)

print(training_prediction_error)
print(test_prediction_error)











