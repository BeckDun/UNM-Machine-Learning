from problem2 import make_classification
from linear_svc import LinearSVC
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC as SklearnLinearSVC

param_combos = [(x,y) for x in [10,50,100] for y in [500,5000,50000]]

u = 100
seed = 40

time_results = {}
loss_results = {}
acc_results = {}

for d, n in param_combos:
    X_train, X_test, y_train, y_test, _ = make_classification(d,n,u,seed)

    svc = LinearSVC(eta=0.001, n_iter=50, random_state=1, lambda_param=0.01)

    start = time.time()
    svc.fit(X_train,y_train)
    total_time = time.time() - start

    train_acc = np.mean(svc.predict(X_train) == y_train) * 100
    test_acc = np.mean(svc.predict(X_test) == y_test) * 100
    final_loss = svc.losses_[1]

    time_results[(d,n)] = total_time
    loss_results[(d,n)] = svc.losses_
    acc_results[(d,u)] = (train_acc, test_acc)

    print(f"  Time: {total_time:.4f}s | Final Loss: {final_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    

dims = [10, 50, 100]
samples = [500, 5000, 50000]

# Print time cost table
print("\n\nTime Cost Table (seconds):")
print(f"{'d \\\\ n':<10}", end="")
for n in samples:
    print(f"{'n='+str(n):<15}", end="")
print()
for d in dims:
    print(f"d={d:<7}", end="")
    for n in samples:
        print(f"{time_results[(d,n)]:<15.4f}", end="")
    print()

# Print final loss table
print("\n\nFinal Loss Table:")
print(f"{'d \\\\ n':<10}", end="")
for n in samples:
    print(f"{'n='+str(n):<15}", end="")
print()
for d in dims:
    print(f"d={d:<7}", end="")
    for n in samples:
        print(f"{loss_results[(d,n)][-1]:<15.4f}", end="")
    print()




dims = [10, 50, 100]
samples = [500, 5000, 50000]
u = 100
seed = 42

print("\n" + "="*70)
print("Task 4: sklearn LinearSVC - Primal vs Dual")
print("="*70)

primal_times = {}
dual_times = {}
primal_losses = {}
dual_losses = {}

for d in dims:
    for n in samples:
        print(f"\n--- d={d}, n={n} ---")
        X_train, X_test, y_train, y_test, _ = make_classification(d, n, u, seed)

        # Primal (dual=False) - sklearn requires squared_hinge for primal
        svc_primal = SklearnLinearSVC(loss='squared_hinge', dual=False, max_iter=10000, random_state=1)
        start = time.time()
        svc_primal.fit(X_train, y_train)
        primal_time = time.time() - start
        primal_train_acc = svc_primal.score(X_train, y_train) * 100
        primal_test_acc = svc_primal.score(X_test, y_test) * 100

        # Dual (dual=True)
        svc_dual = SklearnLinearSVC(loss='hinge', dual=True, max_iter=10000, random_state=1)
        start = time.time()
        svc_dual.fit(X_train, y_train)
        dual_time = time.time() - start
        dual_train_acc = svc_dual.score(X_train, y_train) * 100
        dual_test_acc = svc_dual.score(X_test, y_test) * 100

        primal_times[(d, n)] = primal_time
        dual_times[(d, n)] = dual_time

        print(f"  Primal: {primal_time:.4f}s | Train: {primal_train_acc:.2f}% | Test: {primal_test_acc:.2f}%")
        print(f"  Dual:   {dual_time:.4f}s | Train: {dual_train_acc:.2f}% | Test: {dual_test_acc:.2f}%")

# Print comparison table
print("\n\nTime Cost Comparison Table (seconds):")
print(f"{'d \\\\ n':<10}", end="")
for n in samples:
    print(f"{'n='+str(n)+' (P/D)':<25}", end="")
print()
for d in dims:
    print(f"d={d:<7}", end="")
    for n in samples:
        p = primal_times[(d, n)]
        du = dual_times[(d, n)]
        print(f"{p:.4f} / {du:.4f}{'':>7}", end="")
    print()