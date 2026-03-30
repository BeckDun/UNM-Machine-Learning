from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from mnist_loader import load_mnist
from fashion_mnist_loader import load_fahion_mnist
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from flaten_function import flaten_images
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from task3 import create_pca_pipeline

def create_subset(X, y, n_models):
    size = len(X) // n_models
    subsets = []

    np.random.seed(1)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    for i in range(n_models):
        start = i * size
        end = start + size
        subsets.append((X[start:end], y[start:end]))

    return subsets

def train_models(subsets, kernel):
    models = []

    for (X, y) in subsets:
        if kernel == "linear":
            model = create_pca_pipeline(100, "linear", C=1)
        elif(kernel == "rbf"):
            model = create_pca_pipeline(100, "rbf", C=50, gamma=0.001)
        elif(kernel == "poly"):
            model = create_pca_pipeline(100, "poly", C=0.1, gamma=0.01, degree=3)

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

def test(data_name, kernel):
    if(data_name == "mnist"):
        train_images, train_labels, test_images, test_labels = load_mnist()
    elif(data_name == "fashion"):
        train_images, train_labels, test_images, test_labels = load_fahion_mnist()
    
    train_images, test_images = flaten_images(train_images, test_images)
    
    n_models = 8
    subsets = create_subset(train_images, train_labels, n_models)

    import time
    start_time = time.time()
    models = train_models(subsets, kernel)
    end_time = time.time()
    training_time = end_time - start_time
    
    predictions = models_predict(models,test_images)
    accuracy = accuracy_score(test_labels, predictions)
    test_error = 1 - accuracy
    print(f"\nDataset: {data_name}")
    print(f"Kernel: {kernel}")
    print(f"Bagging Training Time: {training_time:.2f} seconds")
    print(f"Bagging Test Error: {test_error}")

def main():
    test("mnist", "linear")
    test("mnist", "rbf")
    test("mnist", "poly")
    test("fashion", "linear")
    test("fashion", "rbf")
    test("fashion", "poly")
    
    
if(__name__ == "__main__"):
    main()



