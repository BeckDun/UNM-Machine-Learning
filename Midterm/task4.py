"""
=============================================================================
CS 429/529 Midterm Project — Task 4
Bootstrap Aggregating (Bagging) with SVCs
=============================================================================

Assignment requirement:
    "Write a program that trains a finite set of SVCs using bootstrap
     aggregating. Please use at least 8 models. The training dataset should
     be divided into multiple disjoint sets and each individual model is
     trained based on a subset. The final prediction for an image is
     obtained via voting. Please use the three kernels along with their
     best hyperparameters you found in the previous task.
     Please do not use sklearn.ensemble."

What this script does:
    1. Loads the best hyperparameters from Task 3 (results/best_params.json).
    2. For each dataset (MNIST, Fashion-MNIST):
       a. Standardizes + reduces dimensionality (same as Task 3).
       b. For each kernel (linear, rbf, poly):
          - Splits training data into N_MODELS disjoint subsets.
          - Trains one SVC on each subset using the best hyperparameters.
          - Predicts on the test set with ALL models.
          - Combines predictions via majority voting.
       c. Also trains a single SVC on the full training set for comparison.
    3. Saves comparison results to CSV.

NOTE: We do NOT use sklearn.ensemble. Voting is implemented manually.

Usage:
    python task3.py   # must run first to generate results/best_params.json
    python task4.py

Outputs:
    results/task4_results.csv     — bagging vs single SVC comparison
    results/task4_detail.csv      — individual model accuracies
"""

import os
import time
import warnings
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import stats
import idx2numpy

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of models in the ensemble (assignment requires at least 8)
N_MODELS = 10

# Which PCA dimensionality to use for bagging experiments
# Using PCA-100 as a good balance of speed and accuracy
PCA_N_COMPONENTS = 100

# Paths to the extracted IDX files (update for your machine)
DATA_DIRS = {
    "MNIST":         "data/mnist",
    "Fashion-MNIST": "data/fashion_mnist",
}

# Path to best_params.json from Task 3
BEST_PARAMS_PATH = "results/best_params.json"

# Output directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_dataset(ds_dir):
    """
    Load one dataset from IDX files and flatten images to 784-length vectors.
    Same as Task 3.
    """
    train_images = idx2numpy.convert_from_file(os.path.join(ds_dir, "train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(os.path.join(ds_dir, "train-labels-idx1-ubyte"))
    test_images  = idx2numpy.convert_from_file(os.path.join(ds_dir, "t10k-images-idx3-ubyte"))
    test_labels  = idx2numpy.convert_from_file(os.path.join(ds_dir, "t10k-labels-idx1-ubyte"))

    X_train = train_images.reshape(train_images.shape[0], -1).astype(np.float64)
    X_test  = test_images.reshape(test_images.shape[0], -1).astype(np.float64)

    return X_train, train_labels, X_test, test_labels


# =============================================================================
# STEP 2: SPLIT DATA INTO DISJOINT SUBSETS
# =============================================================================

def split_disjoint(X, y, n_splits):
    """
    Split the training data into n_splits disjoint subsets.

    The assignment says: "The training dataset should be divided into
    multiple disjoint sets." This means no overlap — each sample appears
    in exactly one subset.

    We shuffle first so each subset gets a mix of all classes.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training features.
    y : np.ndarray, shape (n_samples,)
        Training labels.
    n_splits : int
        Number of disjoint subsets to create.

    Returns
    -------
    subsets : list of (X_subset, y_subset) tuples
        Each subset contains approximately n_samples / n_splits samples.
    """
    n_samples = len(X)

    # Shuffle the indices so subsets are randomized
    np.random.seed(42)
    indices = np.random.permutation(n_samples)

    # Split indices into n_splits roughly equal chunks
    # np.array_split handles uneven division gracefully
    index_chunks = np.array_split(indices, n_splits)

    subsets = []
    for chunk in index_chunks:
        subsets.append((X[chunk], y[chunk]))

    return subsets


# =============================================================================
# STEP 3: MAJORITY VOTING
# =============================================================================

def majority_vote(predictions_list):
    """
    Combine predictions from multiple models via majority voting.

    The assignment says: "The final prediction for an image is obtained
    via voting."

    For each test sample, each model casts a vote (its predicted class).
    The final prediction is the class that received the most votes.
    Ties are broken by choosing the smallest class label.

    Parameters
    ----------
    predictions_list : list of np.ndarray
        Each element is a (n_test,) array of predictions from one model.
        Length of the list = number of models.

    Returns
    -------
    final_predictions : np.ndarray, shape (n_test,)
        The majority-voted prediction for each test sample.
    """
    # Stack all predictions into a 2D array: (n_models, n_test)
    all_preds = np.array(predictions_list)

    # For each test sample (column), find the most common prediction
    # scipy.stats.mode returns the most frequent value along an axis
    mode_result = stats.mode(all_preds, axis=0, keepdims=False)
    final_predictions = mode_result.mode

    return final_predictions


# =============================================================================
# STEP 4: TRAIN ENSEMBLE (BAGGING)
# =============================================================================

def train_bagging(X_train, y_train, X_test, y_test, kernel, params, n_models):
    """
    Train an ensemble of SVCs using bootstrap aggregating (bagging).

    Steps:
      1. Split training data into n_models disjoint subsets.
      2. Train one SVC on each subset.
      3. Each model predicts on the full test set.
      4. Combine predictions via majority voting.

    NOTE: We do NOT use sklearn.ensemble as the assignment prohibits it.

    Parameters
    ----------
    X_train, y_train : full training data
    X_test, y_test   : test data
    kernel : str
        SVC kernel: 'linear', 'rbf', or 'poly'
    params : dict
        Best hyperparameters for this kernel from Task 3
    n_models : int
        Number of models in the ensemble

    Returns
    -------
    result : dict with keys:
        ensemble_test_acc, ensemble_test_err, ensemble_train_time,
        individual_accs (list of each model's test accuracy)
    """
    # Step 1: Split training data into disjoint subsets
    subsets = split_disjoint(X_train, y_train, n_models)

    print(f"        Split {len(X_train)} samples into {n_models} "
          f"disjoint subsets of ~{len(X_train)//n_models} each")

    # Step 2 & 3: Train each model and collect predictions
    all_predictions = []     # list of prediction arrays from each model
    individual_accs = []     # test accuracy of each individual model
    total_train_time = 0.0   # sum of training times for all models

    for i, (X_sub, y_sub) in enumerate(subsets):
        # Create an SVC with the best hyperparameters from Task 3
        svc = SVC(kernel=kernel, **params)

        # Train on this subset only
        t0 = time.time()
        svc.fit(X_sub, y_sub)
        train_time = time.time() - t0
        total_train_time += train_time

        # Predict on the full test set
        test_pred = svc.predict(X_test)
        all_predictions.append(test_pred)

        # Record this model's individual accuracy
        ind_acc = accuracy_score(y_test, test_pred)
        individual_accs.append(ind_acc)

        print(f"          Model {i+1}/{n_models}: "
              f"{len(X_sub)} samples, "
              f"test_acc={ind_acc:.4f}, "
              f"train_time={train_time:.1f}s")

    # Step 4: Majority voting to combine all predictions
    ensemble_pred = majority_vote(all_predictions)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_err = 1.0 - ensemble_acc

    print(f"        Ensemble (majority vote): test_acc={ensemble_acc:.4f}, "
          f"test_err={ensemble_err:.4f}, total_time={total_train_time:.1f}s")

    return {
        "ensemble_test_acc":  round(ensemble_acc, 4),
        "ensemble_test_err":  round(ensemble_err, 4),
        "ensemble_train_time": round(total_train_time, 2),
        "individual_accs":    [round(a, 4) for a in individual_accs],
        "n_models":           n_models,
    }


# =============================================================================
# STEP 5: TRAIN SINGLE SVC (for comparison)
# =============================================================================

def train_single(X_train, y_train, X_test, y_test, kernel, params):
    """
    Train a single SVC on the FULL training set for comparison with bagging.

    Parameters
    ----------
    X_train, y_train : full training data
    X_test, y_test   : test data
    kernel : str
        SVC kernel type
    params : dict
        Best hyperparameters from Task 3

    Returns
    -------
    result : dict with single model's accuracy, error, and training time
    """
    svc = SVC(kernel=kernel, **params)

    t0 = time.time()
    svc.fit(X_train, y_train)
    train_time = time.time() - t0

    train_acc = accuracy_score(y_train, svc.predict(X_train))
    test_acc  = accuracy_score(y_test,  svc.predict(X_test))

    print(f"        Single SVC: train_acc={train_acc:.4f}, "
          f"test_acc={test_acc:.4f}, time={train_time:.1f}s")

    return {
        "single_train_acc":  round(train_acc, 4),
        "single_test_acc":   round(test_acc, 4),
        "single_test_err":   round(1.0 - test_acc, 4),
        "single_train_time": round(train_time, 2),
    }


# =============================================================================
# STEP 6: RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Load best hyperparameters from Task 3
    # ------------------------------------------------------------------
    if not os.path.exists(BEST_PARAMS_PATH):
        print(f"ERROR: {BEST_PARAMS_PATH} not found.")
        print("Please run task3.py first to generate best hyperparameters.")
        exit(1)

    with open(BEST_PARAMS_PATH, "r") as f:
        all_best_params = json.load(f)
    print(f"Loaded best params from {BEST_PARAMS_PATH}")

    # ------------------------------------------------------------------
    # Run bagging experiments on both datasets
    # ------------------------------------------------------------------
    comparison_rows = []  # for the report comparison table
    detail_rows = []      # individual model details

    for dataset_name, data_dir in DATA_DIRS.items():

        print(f"\n{'='*70}")
        print(f"  DATASET: {dataset_name}")
        print(f"{'='*70}")

        # Load and flatten data
        X_train, y_train, X_test, y_test = load_dataset(data_dir)
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

        # Standardize (same as Task 3 — fit on train only)
        print("\n  Standardizing...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Dimensionality reduction with PCA
        print(f"  Reducing to {PCA_N_COMPONENTS} dims with PCA...")
        t0 = time.time()
        pca = PCA(n_components=PCA_N_COMPONENTS)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca  = pca.transform(X_test_scaled)
        pca_time = time.time() - t0
        print(f"  PCA done in {pca_time:.2f}s")

        # Get best params for this dataset
        best_params = all_best_params[dataset_name]

        # --- Loop over each kernel ---
        for kernel_name, params in best_params.items():

            # Convert string values back to proper types if needed
            # (JSON may have stored gamma as string "scale")
            clean_params = {}
            for k, v in params.items():
                if k == "C" or k == "gamma":
                    # Try to convert to float; keep as string if it fails (e.g. "scale")
                    try:
                        clean_params[k] = float(v)
                    except (ValueError, TypeError):
                        clean_params[k] = v
                elif k == "degree":
                    clean_params[k] = int(v)
                else:
                    clean_params[k] = v
            params = clean_params

            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"\n    Kernel: {kernel_name} ({param_str})")

            # ── Train single SVC on full data (baseline) ──
            print(f"\n      --- Single SVC (full {len(X_train_pca)} samples) ---")
            single_result = train_single(
                X_train_pca, y_train, X_test_pca, y_test,
                kernel_name, params
            )

            # ── Train bagging ensemble ──
            print(f"\n      --- Bagging Ensemble ({N_MODELS} models) ---")
            bag_result = train_bagging(
                X_train_pca, y_train, X_test_pca, y_test,
                kernel_name, params, N_MODELS
            )

            # ── Store comparison row for report ──
            comparison_rows.append({
                "dataset":             dataset_name,
                "kernel":              kernel_name,
                "params":              param_str,
                "single_test_acc":     single_result["single_test_acc"],
                "single_test_err":     single_result["single_test_err"],
                "single_train_time":   single_result["single_train_time"],
                "ensemble_test_acc":   bag_result["ensemble_test_acc"],
                "ensemble_test_err":   bag_result["ensemble_test_err"],
                "ensemble_train_time": bag_result["ensemble_train_time"],
                "n_models":            N_MODELS,
                "acc_diff":            round(bag_result["ensemble_test_acc"]
                                             - single_result["single_test_acc"], 4),
            })

            # ── Store individual model details ──
            for i, ind_acc in enumerate(bag_result["individual_accs"]):
                detail_rows.append({
                    "dataset":    dataset_name,
                    "kernel":     kernel_name,
                    "model_id":   i + 1,
                    "n_samples":  len(X_train_pca) // N_MODELS,
                    "test_acc":   ind_acc,
                    "test_err":   round(1.0 - ind_acc, 4),
                })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(RESULTS_DIR, "task4_results.csv"), index=False)
    print(f"\n  Saved comparison results to results/task4_results.csv")

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(os.path.join(RESULTS_DIR, "task4_detail.csv"), index=False)
    print(f"  Saved individual model details to results/task4_detail.csv")

    # ------------------------------------------------------------------
    # Print comparison table for the report
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TASK 4: SINGLE SVC vs BAGGING COMPARISON")
    print(f"{'='*70}")
    print(f"  Using PCA-{PCA_N_COMPONENTS}, {N_MODELS} models in ensemble")

    for dataset_name in DATA_DIRS.keys():
        print(f"\n  ── {dataset_name} ──")
        print(f"  {'Kernel':<10} {'Single Err':>11} {'Bag Err':>11} "
              f"{'Diff':>8} {'Single Time':>12} {'Bag Time':>12}")
        print(f"  {'─'*68}")

        ds_rows = [r for r in comparison_rows if r["dataset"] == dataset_name]
        for r in ds_rows:
            diff = r["acc_diff"]
            # Positive diff means bagging is better
            diff_str = f"{diff:+.4f}"
            print(f"  {r['kernel']:<10} "
                  f"{r['single_test_err']:>11.4f} "
                  f"{r['ensemble_test_err']:>11.4f} "
                  f"{diff_str:>8} "
                  f"{r['single_train_time']:>11.2f}s "
                  f"{r['ensemble_train_time']:>11.2f}s")

    print(f"\n  Note: 'Diff' = ensemble_acc - single_acc")
    print(f"        Positive means bagging improved accuracy.")
    print(f"\n{'='*70}")
    print("  Task 4 complete!")
    print(f"{'='*70}")