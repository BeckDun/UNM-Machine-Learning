# Beckett dunlavy CS 429
from problem2 import make_classification
from linear_svc import LinearSVC
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC as SklearnLinearSVC
import matplotlib.pyplot as plt

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
    final_loss = svc.losses_[-1]

    time_results[(d,n)] = total_time
    loss_results[(d,n)] = svc.losses_
    acc_results[(d,n)] = (train_acc, test_acc)

    print(f"  Time: {total_time:.4f}s | Final Loss: {final_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    

dims = [10, 50, 100]
samples = [500, 5000, 50000]

print("\n\nTime Cost Table (seconds):")
print(f"{'d : n':<10}", end="")
for n in samples:
    print(f"{'n='+str(n):<15}", end="")
print()
for d in dims:
    print(f"d={d:<7}", end="")
    for n in samples:
        print(f"{time_results[(d,n)]:<15.4f}", end="")
    print()

print("\n\nFinal Loss Table:")
print(f"{'d : n':<10}", end="")
for n in samples:
    print(f"{'n='+str(n):<15}", end="")
print()
for d in dims:
    print(f"d={d:<7}", end="")
    for n in samples:
        print(f"{loss_results[(d,n)][-1]:<15.4f}", end="")
    print()

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, d in enumerate(dims):
    for j, n in enumerate(samples):
        axes[i][j].plot(loss_results[(d, n)])
        axes[i][j].set_title(f'd={d}, n={n}')
        axes[i][j].set_xlabel('Epoch')
        axes[i][j].set_ylabel('Loss')
plt.suptitle('Task 3: Loss Convergence of Custom LinearSVC', fontsize=14)
plt.tight_layout()
plt.savefig('task3_loss_convergence.png', dpi=150)
plt.close()


def compute_hinge_loss(model, X, y, C=1.0):
    decision = model.decision_function(X)
    hinge = np.mean(np.maximum(0, 1 - y * decision))
    reg = 0.5 * np.dot(model.coef_[0], model.coef_[0])
    return reg + C * hinge


dims = [10, 50, 100]
samples = [500, 5000, 50000]
u = 100
seed = 42

print("\n" + "="*70)
print("Task 4: sklearn LinearSVC - Primal vs Dual")
print("="*70)

primal_times = {}
dual_times = {}
primal_hinge_losses = {}
dual_hinge_losses = {}

for d in dims:
    for n in samples:
        print(f"\n--- d={d}, n={n} ---")
        X_train, X_test, y_train, y_test, _ = make_classification(d, n, u, seed)

        svc_primal = SklearnLinearSVC(loss='squared_hinge', dual=False, max_iter=10000, random_state=1)
        start = time.time()
        svc_primal.fit(X_train, y_train)
        primal_time = time.time() - start
        primal_train_acc = svc_primal.score(X_train, y_train) * 100
        primal_test_acc = svc_primal.score(X_test, y_test) * 100
        primal_loss = compute_hinge_loss(svc_primal, X_train, y_train)

        svc_dual = SklearnLinearSVC(loss='hinge', dual=True, max_iter=10000, random_state=1)
        start = time.time()
        svc_dual.fit(X_train, y_train)
        dual_time = time.time() - start
        dual_train_acc = svc_dual.score(X_train, y_train) * 100
        dual_test_acc = svc_dual.score(X_test, y_test) * 100
        dual_loss = compute_hinge_loss(svc_dual, X_train, y_train)

        primal_times[(d, n)] = primal_time
        dual_times[(d, n)] = dual_time
        primal_hinge_losses[(d, n)] = primal_loss
        dual_hinge_losses[(d, n)] = dual_loss

        print(f"  Primal: {primal_time:.4f}s | Loss: {primal_loss:.4f} | Train: {primal_train_acc:.2f}% | Test: {primal_test_acc:.2f}%")
        print(f"  Dual:   {dual_time:.4f}s | Loss: {dual_loss:.4f} | Train: {dual_train_acc:.2f}% | Test: {dual_test_acc:.2f}%")

print("\n\nTime Cost Comparison Table (seconds):")
print(f"{'d : n':<10}", end="")
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

print("\n\nHinge Loss Comparison Table - Primal / Dual:")
print(f"{'d : n':<10}", end="")
for n in samples:
    print(f"{'n='+str(n):<25}", end="")
print()
for d in dims:
    print(f"d={d:<7}", end="")
    for n in samples:
        p = primal_hinge_losses[(d, n)]
        du = dual_hinge_losses[(d, n)]
        print(f"{p:.4f} / {du:.4f}{'':>7}", end="")
    print()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for j, n in enumerate(samples):
    primal_t = [primal_times[(d, n)] for d in dims]
    dual_t = [dual_times[(d, n)] for d in dims]
    x = np.arange(len(dims))
    width = 0.35
    axes[j].bar(x - width/2, primal_t, width, label='Primal', color='steelblue')
    axes[j].bar(x + width/2, dual_t, width, label='Dual', color='coral')
    axes[j].set_xlabel('Dimensions')
    axes[j].set_ylabel('Time (s)')
    axes[j].set_title(f'n={n}')
    axes[j].set_xticks(x)
    axes[j].set_xticklabels([f'd={d}' for d in dims])
    axes[j].legend()
plt.suptitle('Task 4: Primal vs Dual Time Comparison (sklearn LinearSVC)', fontsize=14)
plt.tight_layout()
plt.savefig('task4_time_comparison.png', dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for j, n in enumerate(samples):
    primal_l = [primal_hinge_losses[(d, n)] for d in dims]
    dual_l = [dual_hinge_losses[(d, n)] for d in dims]
    x = np.arange(len(dims))
    width = 0.35
    axes[j].bar(x - width/2, primal_l, width, label='Primal', color='steelblue')
    axes[j].bar(x + width/2, dual_l, width, label='Dual', color='coral')
    axes[j].set_xlabel('Dimensions')
    axes[j].set_ylabel('Hinge Loss')
    axes[j].set_title(f'n={n}')
    axes[j].set_xticks(x)
    axes[j].set_xticklabels([f'd={d}' for d in dims])
    axes[j].legend()
plt.suptitle('Task 4: Primal vs Dual Loss Comparison (sklearn LinearSVC)', fontsize=14)
plt.tight_layout()
plt.savefig('task4_loss_comparison.png', dpi=150)
plt.close()
