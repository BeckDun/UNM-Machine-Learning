from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mnist_loader import load_mnist
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from flaten_function import flaten_images
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def create_pca_pipeline(dimension, kernel_type, C, gamma = None, degree = None):
    
    if(kernel_type == "linear"):
        svc = SVC(kernel="linear", C=C)
    elif(kernel_type == "rbf"):
        svc = SVC(kernel="rbf", C=C, gamma=gamma)
    elif(kernel_type == "poly"):
        svc = SVC(kernel="poly", C=C, gamma=gamma, degree=degree)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=dimension)),
        ("svc", svc)
        ])
    
    return pipeline
    

def create_lda_pipeline(kernel_type, C, gamma = None, degree = None):
    if(kernel_type == "linear"):
        svc = SVC(kernel="linear", C=C)
    elif(kernel_type == "rbf"):
        svc = SVC(kernel="rbf", C=C, gamma=gamma)
    elif(kernel_type == "poly"):
        svc = SVC(kernel="poly", C=C, gamma=gamma, degree=degree)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis()),
        ("svc", svc)
    ])
    
    return pipeline

def test():
    train_images, train_labels, test_images, test_labels = load_mnist()
    train_images, test_images = flaten_images(train_images, test_images)

    pca_pipeline = create_pca_pipeline(dimension = 100, kernel_type = "linear", C = .1)
    pca_pipeline.fit(train_images,train_labels)

    pca_train_predictions = pca_pipeline.predict(train_images)
    pca_test_predictions = pca_pipeline.predict(test_images)

    pca_training_prediction_error = 1- accuracy_score(train_labels, pca_train_predictions)
    pca_test_prediction_error = 1 - accuracy_score(test_labels, pca_test_predictions)
    
    print(pca_training_prediction_error)
    print(pca_test_prediction_error)

if(__name__ == "__main__"):
    test()
    

    













