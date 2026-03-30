import time
import itertools
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

    param_grids = {
        "linear": {"C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
        "rbf":    {"C": [0.1, 1, 5, 10, 50], "gamma": ["scale", 0.01, 0.005, 0.001]},
        "poly":   {"C": [0.1, 1, 5, 10], "gamma": ["scale", 0.01, 0.005, 0.001], "degree": [2, 3, 4]},
    }

    pca_dims = [50, 100, 200]

    for kernel_type, grid in param_grids.items():
        keys = list(grid.keys())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]

        for dim in pca_dims:
            for i, params in enumerate(combos):
                pipe = create_pca_pipeline(dim, kernel_type, params["C"],
                                           params.get("gamma"), params.get("degree"))
                t0 = time.time()
                pipe.fit(train_images, train_labels)
                t = time.time() - t0

                train_err = 1 - accuracy_score(train_labels, pipe.predict(train_images))
                test_err = 1 - accuracy_score(test_labels, pipe.predict(test_images))

                ps = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"PCA-{dim} {kernel_type} [{i+1}/{len(combos)}] {ps:40s} "
                      f"train_err={train_err:.4f}  test_err={test_err:.4f}  time={t:.1f}s")

        for i, params in enumerate(combos):
            pipe = create_lda_pipeline(kernel_type, params["C"],
                                       params.get("gamma"), params.get("degree"))
            t0 = time.time()
            pipe.fit(train_images, train_labels)
            t = time.time() - t0

            train_err = 1 - accuracy_score(train_labels, pipe.predict(train_images))
            test_err = 1 - accuracy_score(test_labels, pipe.predict(test_images))

            ps = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"LDA-9 {kernel_type} [{i+1}/{len(combos)}] {ps:40s} "
                  f"train_err={train_err:.4f}  test_err={test_err:.4f}  time={t:.1f}s")

if(__name__ == "__main__"):
    test()
