from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from mnist_loader import load_mnist
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from flaten_function import flaten_images
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

def create_subset(X, y, n_models):
    size = len(X) // n_models
    subsets = []

    for i in range(n_models):
        start = i * size
        end = start + size
        subsets.append((X[start:end], y[start:end]))

    return subsets

def train_models(subsets):

    models = []

    for i, (X,y) in enumerate(subsets):

        if i % 3 == 0:
            model = SVC(kernel="linear", C = 1)
        elif(i % 3 == 1):
            model = SVC(kernel='rbf', C = 50, gamma= 0.001)
        else:
            model = SVC(kernel="poly", C = 0.1, gamma=0.01, degree=3)

        model.fit(X,y)
        models.append(model)

    return models

def majority_vote(predictions):
    final = []

    predictions = np.array(predictions)

    for col in predictions.T:
        counts = np.bincount(col)
        final.append(np.argmax(counts))

    return np.array(final)

def models_predict(models, X):
    all_predictions = []
    
    for model in models:
        predictions = model.predict(X)
        all_predictions.append(predictions)

    return majority_vote(all_predictions)

def main():
    train_images, train_labels, test_images, test_labels = load_mnist()
    train_images, test_images = flaten_images(train_images, test_images)
    scaler = StandardScaler()

    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    n_models = 8

    subsets = create_subset(train_images, train_labels, n_models)
    models = train_models(subsets)
    predictions = models_predict(models,test_images)
    accuracy = accuracy_score(test_labels, predictions)
    print("Bootstrap Aggregating Accuracy:", accuracy)

if(__name__ == "__main__"):
    main()



