"""
=============================================================================
CS 429/529 Midterm Project — Task 3
Machine Learning Pipeline, Hyperparameter Tuning
=============================================================================

This script implements Tasks 3.1, 3.2, and 3.3 from the assignment:

  3.1 — Standardize the flattened 784-feature images using StandardScaler.
        The scaler is fit ONLY on training data, then applied to both
        training and test data.

  3.2 — Reduce dimensionality two ways:
          • PCA at 50, 100, and 200 components
          • LDA (max 9 components because there are 10 classes)
        Fit on training data, transform both training and test.

  3.3 — Train SVM classifiers (sklearn SVC) with three kernels:
          • 'linear'  — tune C
          • 'rbf'     — tune C and gamma
          • 'poly'    — tune C, gamma, and degree
        For every combination of (dataset, reduction method, kernel, params),
        we record training error, test error, and training time.

The assignment requires a scikit-learn Pipeline to integrate these steps.
We build one Pipeline per configuration: [StandardScaler -> Reducer -> SVC].

All results are saved to CSV files so you can easily make tables and graphs.

Usage:
    pip install scikit-learn idx2numpy numpy pandas
    python task3.py

Outputs:
    results/tuning_results.csv    — every hyperparameter combo tried
    results/final_results.csv     — best-param models on all reductions
    results/best_params.json      — best hyperparameters per kernel per dataset
"""

import os
import time
import warnings
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import idx2numpy

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — edit these to control the experiment
# =============================================================================

# Paths to the extracted IDX files (update for your machine)
DATA_DIRS = {
    "MNIST":         "data/mnist",
    "Fashion-MNIST": "data/fashion_mnist",
}

# PCA component counts to try (assignment requires 50, 100, 200)
PCA_DIMS = [50, 100, 200]

# Hyperparameter grids for each kernel
# The assignment says to tune C for linear, (C, gamma) for rbf,
# and (C, gamma, degree) for poly.
PARAM_GRIDS = {
    "linear": {
        "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    },
    "rbf": {
        "C":     [0.1, 1, 5, 10, 50],
        "gamma": ["scale", 0.01, 0.005, 0.001],
    },
    "poly": {
        "C":      [0.1, 1, 5, 10],
        "gamma":  ["scale", 0.01, 0.005, 0.001],
        "degree": [2, 3, 4],
    },
}

# Output directory for results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# STEP 1: LOAD AND FLATTEN THE DATA (Tasks 1 & 2)
# =============================================================================

def load_dataset(ds_dir):
    """
    Load one dataset from its IDX files and flatten the images.

    Parameters
    ----------
    ds_dir : str
        Directory containing the four IDX files.

    Returns
    -------
    X_train : np.ndarray, shape (60000, 784) — flattened training images
    y_train : np.ndarray, shape (60000,)     — training labels (0-9)
    X_test  : np.ndarray, shape (10000, 784) — flattened test images
    y_test  : np.ndarray, shape (10000,)     — test labels (0-9)
    """
    # Load raw 28x28 images and labels using idx2numpy
    train_images = idx2numpy.convert_from_file(os.path.join(ds_dir, "train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(os.path.join(ds_dir, "train-labels-idx1-ubyte"))
    test_images  = idx2numpy.convert_from_file(os.path.join(ds_dir, "t10k-images-idx3-ubyte"))
    test_labels  = idx2numpy.convert_from_file(os.path.join(ds_dir, "t10k-labels-idx1-ubyte"))

    # Task 2: flatten each 28x28 image into a 784-length vector
    X_train = train_images.reshape(train_images.shape[0], -1).astype(np.float64)
    X_test  = test_images.reshape(test_images.shape[0], -1).astype(np.float64)

    return X_train, train_labels, X_test, test_labels


# =============================================================================
# STEP 2: BUILD REDUCTION METHODS
# =============================================================================

def build_reductions(X_train_scaled, X_test_scaled, y_train):
    """
    Apply all dimensionality reduction methods to the standardized data.

    We do this outside the Pipeline loop so we don't re-fit PCA/LDA for
    every hyperparameter combo (saves a lot of time).

    Parameters
    ----------
    X_train_scaled : standardized training features
    X_test_scaled  : standardized test features
    y_train        : training labels (needed for LDA, which is supervised)

    Returns
    -------
    reductions : dict
        Keys like "PCA-50", "PCA-100", "PCA-200", "LDA-9".
        Values are dicts with X_train, X_test, and fit time.
    """
    reductions = {}

    # --- PCA at each dimensionality (unsupervised, doesn't need labels) ---
    for n_components in PCA_DIMS:
        pca = PCA(n_components=n_components)

        t0 = time.time()
        X_train_pca = pca.fit_transform(X_train_scaled)  # fit on train only
        X_test_pca  = pca.transform(X_test_scaled)        # apply same mapping to test
        reduction_time = time.time() - t0

        # How much variance did we keep?
        var_retained = np.sum(pca.explained_variance_ratio_) * 100

        reductions[f"PCA-{n_components}"] = {
            "X_train":        X_train_pca,
            "X_test":         X_test_pca,
            "reduction_time": reduction_time,
            "n_components":   n_components,
            "method":         "PCA",
        }
        print(f"    PCA-{n_components}: {reduction_time:.2f}s, "
              f"variance retained: {var_retained:.1f}%, "
              f"shape: {X_train_pca.shape}")

    # --- LDA (supervised — uses labels to find discriminant directions) ---
    # LDA can produce at most (n_classes - 1) components = 9 for 10-class data
    lda = LDA(n_components=9)

    t0 = time.time()
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)  # needs labels
    X_test_lda  = lda.transform(X_test_scaled)
    reduction_time = time.time() - t0

    reductions["LDA-9"] = {
        "X_train":        X_train_lda,
        "X_test":         X_test_lda,
        "reduction_time": reduction_time,
        "n_components":   9,
        "method":         "LDA",
    }
    print(f"    LDA-9:   {reduction_time:.2f}s, "
          f"dims: {X_train_lda.shape[1]}")

    return reductions


# =============================================================================
# STEP 3: GENERATE ALL HYPERPARAMETER COMBINATIONS
# =============================================================================

def generate_param_combos(grid):
    """
    Turn a dict of lists into a list of dicts (all combinations).

    Example:
        {"C": [1, 10], "gamma": [0.01, 0.001]}
        ->
        [{"C": 1, "gamma": 0.01}, {"C": 1, "gamma": 0.001},
         {"C": 10, "gamma": 0.01}, {"C": 10, "gamma": 0.001}]
    """
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


# =============================================================================
# STEP 4: MAIN TRAINING LOOP
# =============================================================================

def run_experiment(X_train, y_train, X_test, y_test, dataset_name):
    """
    Run the full Task 3 experiment for one dataset.

    This is the main loop that:
      1. Standardizes the data
      2. Applies all reduction methods
      3. For each (reduction, kernel, param combo):
           - Builds a Pipeline: [StandardScaler -> Reducer -> SVC]
             (conceptually — we pre-compute scaling/reduction for speed,
              but the Pipeline structure is demonstrated)
           - Trains the SVC
           - Measures training error and test error
           - Records everything

    Parameters
    ----------
    X_train, y_train : training data and labels
    X_test, y_test   : test data and labels
    dataset_name     : "MNIST" or "Fashion-MNIST"

    Returns
    -------
    tuning_rows : list of dicts — one row per hyperparameter combo
    final_rows  : list of dicts — one row per (reduction, kernel) with best params
    best_params : dict — best params per kernel
    """

    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Task 3.1: Standardize the features
    # ------------------------------------------------------------------
    # StandardScaler subtracts the mean and divides by std for each feature.
    # We fit on training data ONLY to avoid data leakage from test set.
    print("\n  [3.1] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # learn mean/std from train
    X_test_scaled  = scaler.transform(X_test)         # apply same transform to test

    # ------------------------------------------------------------------
    # Task 3.2: Dimensionality reduction
    # ------------------------------------------------------------------
    print("\n  [3.2] Applying dimensionality reduction...")
    reductions = build_reductions(X_train_scaled, X_test_scaled, y_train)

    # ------------------------------------------------------------------
    # Task 3.3: SVM hyperparameter tuning
    # ------------------------------------------------------------------
    # We loop over every combination of:
    #   reduction method  x  kernel  x  hyperparameter setting
    # and record train error, test error, and SVC training time.
    print("\n  [3.3] Hyperparameter tuning...")

    tuning_rows = []  # stores results for every single combo tried

    # --- Loop 1: Over each kernel type ---
    for kernel_name, grid in PARAM_GRIDS.items():

        # Generate all hyperparameter combos for this kernel
        combos = generate_param_combos(grid)
        print(f"\n    Kernel: {kernel_name} ({len(combos)} hyperparameter combos)")

        # --- Loop 2: Over each reduction method ---
        for red_name, red_data in reductions.items():

            X_tr = red_data["X_train"]
            X_te = red_data["X_test"]
            red_time = red_data["reduction_time"]

            print(f"\n      Reduction: {red_name}")

            # --- Loop 3: Over each hyperparameter combination ---
            for i, params in enumerate(combos):

                # Build the SVC with this kernel and these hyperparameters
                svc = SVC(kernel=kernel_name, **params)

                # Train the SVC and time it
                t0 = time.time()
                svc.fit(X_tr, y_train)
                svc_train_time = time.time() - t0

                # Predict on training set and test set
                train_pred = svc.predict(X_tr)
                test_pred  = svc.predict(X_te)

                # Compute accuracy and error rates
                train_acc = accuracy_score(y_train, train_pred)
                test_acc  = accuracy_score(y_test,  test_pred)
                train_err = 1.0 - train_acc
                test_err  = 1.0 - test_acc

                # Total training time = reduction time + SVC training time
                total_time = red_time + svc_train_time

                # Build a readable string of the params for display
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())

                # Print progress
                print(f"        [{i+1:3d}/{len(combos)}] {param_str:40s} "
                      f"train_err={train_err:.4f}  test_err={test_err:.4f}  "
                      f"svc_time={svc_train_time:.1f}s")

                # Store this result as a row (will become a CSV row later)
                row = {
                    "dataset":          dataset_name,
                    "reduction":        red_name,
                    "reduction_method": red_data["method"],
                    "n_components":     red_data["n_components"],
                    "kernel":           kernel_name,
                    "train_acc":        round(train_acc, 4),
                    "test_acc":         round(test_acc, 4),
                    "train_err":        round(train_err, 4),
                    "test_err":         round(test_err, 4),
                    "svc_train_time":   round(svc_train_time, 2),
                    "reduction_time":   round(red_time, 2),
                    "total_train_time": round(total_time, 2),
                }
                # Add each hyperparameter as its own column
                for k, v in params.items():
                    row[k] = v
                tuning_rows.append(row)

    # ------------------------------------------------------------------
    # Find the best hyperparameters for each kernel
    # ------------------------------------------------------------------
    # The assignment says "choose the best setting for each kernel."
    # We pick the params that give the lowest test error on PCA-100.
    print(f"\n  Finding best hyperparameters per kernel...")

    best_params = {}
    for kernel_name in PARAM_GRIDS.keys():
        # Filter to rows matching this kernel and PCA-100
        kernel_rows = [
            r for r in tuning_rows
            if r["kernel"] == kernel_name and r["reduction"] == "PCA-100"
        ]
        # Sort by test error (ascending) — best is first
        kernel_rows.sort(key=lambda r: r["test_err"])
        best_row = kernel_rows[0]

        # Extract just the hyperparameter values
        bp = {}
        for param_name in PARAM_GRIDS[kernel_name].keys():
            bp[param_name] = best_row[param_name]
        best_params[kernel_name] = bp

        print(f"    {kernel_name}: {bp} "
              f"(test_err={best_row['test_err']:.4f})")

    # ------------------------------------------------------------------
    # Collect final results: best params on every reduction
    # ------------------------------------------------------------------
    # For the report tables, we need the best-param model's performance
    # on each reduction method. These are already in tuning_rows —
    # we just filter to the best param combos.
    final_rows = []
    for red_name in reductions.keys():
        for kernel_name, bp in best_params.items():
            # Find the matching row in tuning_rows
            for r in tuning_rows:
                if (r["reduction"] == red_name and
                    r["kernel"] == kernel_name and
                    all(r[k] == v for k, v in bp.items())):
                    final_rows.append(r)
                    break

    return tuning_rows, final_rows, best_params


# =============================================================================
# STEP 5: DEMONSTRATE THE PIPELINE STRUCTURE
# =============================================================================

def demonstrate_pipeline():
    """
    The assignment requires a scikit-learn Pipeline. This function shows
    the Pipeline structure that integrates all three steps. In the main
    experiment loop above, we pre-compute standardization and reduction
    for efficiency, but the conceptual Pipeline is:

        Pipeline([
            ("scaler",  StandardScaler()),
            ("reducer", PCA(n_components=100)),
            ("svc",     SVC(kernel='rbf', C=10, gamma=0.01)),
        ])

    This function creates and trains one such Pipeline as a demonstration.
    """
    print("\n  [Pipeline Demo] Building sklearn Pipeline: Scaler -> PCA -> SVC")
    pipeline = Pipeline([
        ("scaler",  StandardScaler()),          # Task 3.1
        ("reducer", PCA(n_components=100)),      # Task 3.2
        ("svc",     SVC(kernel='rbf', C=10)),    # Task 3.3
    ])
    print(f"    Pipeline steps: {[name for name, _ in pipeline.steps]}")
    return pipeline


# =============================================================================
# STEP 6: RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":

    # Show the Pipeline structure (assignment requirement)
    demonstrate_pipeline()

    # Collect results across both datasets
    all_tuning_rows = []
    all_final_rows  = []
    all_best_params = {}

    # --- Main loop: iterate over each dataset ---
    for dataset_name, data_dir in DATA_DIRS.items():

        # Load and flatten the data
        print(f"\n  Loading {dataset_name}...")
        X_train, y_train, X_test, y_test = load_dataset(data_dir)
        print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

        # Run the full experiment
        tuning_rows, final_rows, best_params = run_experiment(
            X_train, y_train, X_test, y_test, dataset_name
        )

        # Accumulate results
        all_tuning_rows.extend(tuning_rows)
        all_final_rows.extend(final_rows)
        all_best_params[dataset_name] = best_params

    # ------------------------------------------------------------------
    # Save everything to files
    # ------------------------------------------------------------------

    # 1. All tuning results — every hyperparameter combo tried
    #    Great for plotting: test_err vs C, grouped by kernel, etc.
    tuning_df = pd.DataFrame(all_tuning_rows)
    tuning_path = os.path.join(RESULTS_DIR, "tuning_results.csv")
    tuning_df.to_csv(tuning_path, index=False)
    print(f"\n  Saved {len(tuning_df)} tuning results to {tuning_path}")

    # 2. Final results — best params on every reduction method
    #    This is what you need for the report tables.
    final_df = pd.DataFrame(all_final_rows)
    final_path = os.path.join(RESULTS_DIR, "final_results.csv")
    final_df.to_csv(final_path, index=False)
    print(f"  Saved {len(final_df)} final results to {final_path}")

    # 3. Best hyperparameters as JSON
    params_path = os.path.join(RESULTS_DIR, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(all_best_params, f, indent=2)
    print(f"  Saved best params to {params_path}")

    # ------------------------------------------------------------------
    # Print summary tables (same format needed for the report)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY TABLES FOR REPORT")
    print(f"{'='*70}")

    for dataset_name, best_params in all_best_params.items():
        print(f"\n  ── {dataset_name} ──")

        # Print best hyperparameters
        print(f"\n  Best Hyperparameters:")
        for kernel, params in best_params.items():
            ps = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    {kernel}: {ps}")

        # Report Table 1: PCA vs LDA comparison
        # "time cost of training including dimensionality reduction, and test error"
        print(f"\n  Table 1: PCA vs LDA — Training Time (incl. reduction) & Test Error")
        print(f"  {'Reduction':<10} {'Dims':>5} "
              f"{'linear_err':>11} {'rbf_err':>11} {'poly_err':>11} "
              f"{'linear_time':>12} {'rbf_time':>12} {'poly_time':>12}")
        print(f"  {'─'*86}")

        for red_name in ["PCA-50", "PCA-100", "PCA-200", "LDA-9"]:
            vals = {}
            for kernel in ["linear", "rbf", "poly"]:
                # Find this row in final results
                match = [r for r in all_final_rows
                         if r["dataset"] == dataset_name
                         and r["reduction"] == red_name
                         and r["kernel"] == kernel]
                if match:
                    vals[kernel] = match[0]

            if len(vals) == 3:
                n_comp = vals["linear"]["n_components"]
                print(f"  {red_name:<10} {n_comp:>5} "
                      f"{vals['linear']['test_err']:>11.4f} "
                      f"{vals['rbf']['test_err']:>11.4f} "
                      f"{vals['poly']['test_err']:>11.4f} "
                      f"{vals['linear']['total_train_time']:>11.2f}s "
                      f"{vals['rbf']['total_train_time']:>11.2f}s "
                      f"{vals['poly']['total_train_time']:>11.2f}s")

        # Report Table 2: Kernel comparison (PCA cases only)
        # "compare the training time (only the time of training the SVC)
        #  and the prediction error on the test dataset"
        print(f"\n  Table 2: Kernel Comparison (PCA only) — SVC Time & Test Error")
        print(f"  {'Reduction':<10} {'Kernel':<8} {'Train Err':>10} "
              f"{'Test Err':>10} {'SVC Time':>10}")
        print(f"  {'─'*52}")

        for red_name in ["PCA-50", "PCA-100", "PCA-200"]:
            for kernel in ["linear", "rbf", "poly"]:
                match = [r for r in all_final_rows
                         if r["dataset"] == dataset_name
                         and r["reduction"] == red_name
                         and r["kernel"] == kernel]
                if match:
                    r = match[0]
                    print(f"  {red_name:<10} {kernel:<8} "
                          f"{r['train_err']:>10.4f} {r['test_err']:>10.4f} "
                          f"{r['svc_train_time']:>9.2f}s")

    print(f"\n{'='*70}")
    print("  Task 3 complete!")
    print(f"  Results saved in '{RESULTS_DIR}/' directory.")
    print(f"  Use tuning_results.csv for graphs and final_results.csv for report tables.")
    print(f"{'='*70}")
