import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

nltk.download('stopwords', quiet=True)
stop = stopwords.words('english')


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub(r'[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def stem_and_stop_remove(text):
    tokens = [w for w in tokenizer_porter(text) if w not in stop]
    return " ".join(tokens)

data = pd.read_csv("movie_data.csv")

X_train = data.loc[:34999, 'review']
y_train = data.loc[:34999, 'sentiment']
X_test = data.loc[35000:, 'review']
y_test = data.loc[35000:, 'sentiment']

X_train = X_train.apply(preprocessor).apply(stem_and_stop_remove)
X_test = X_test.apply(preprocessor).apply(stem_and_stop_remove)

cv = CountVectorizer(max_features=10000)
train_bag = cv.fit_transform(X_train.tolist())
test_bag = cv.transform(X_test.tolist())

tf_idf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
train_bag = tf_idf.fit_transform(train_bag)
test_bag = tf_idf.transform(test_bag)

X_train_tensor = torch.tensor(train_bag.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)
X_test_tensor = torch.tensor(test_bag.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long)

train_data = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=128, shuffle=True
)


class FNN(nn.Module):
    def __init__(self, input_dim=10000):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 3500)
        self.fc2 = nn.Linear(3500, 2000)
        self.fc3 = nn.Linear(2000, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DropoutFNN(nn.Module):
    def __init__(self, input_dim=10000, p1=0.5, p2=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 3500)
        self.fc2 = nn.Linear(3500, 2000)
        self.fc3 = nn.Linear(2000, 2)
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)

    def forward(self, x):
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        return self.fc3(x)


def train_model(model, loader, n_epochs=2, lr=0.003, weight_decay=1e-4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, time.time() - t0


def _batched_predict(model, X, batch_size=2000):
    model.eval()
    preds_all = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i+batch_size].to(device)
            logits = model(xb)
            preds_all.append(torch.argmax(logits, dim=1).cpu())
    return torch.cat(preds_all)


def evaluate(model):
    preds = _batched_predict(model, X_test_tensor)
    acc = (preds == y_test_tensor).float().mean().item()
    return acc, preds


def train_accuracy(model):
    preds = _batched_predict(model, X_train_tensor)
    return (preds == y_train_tensor).float().mean().item()



print("\n" + "=" * 60)
print("PART 1: Single dropout model vs baseline")
print("=" * 60)

input_dim = X_train_tensor.shape[1]

print("\nTraining baseline (3500-2000, lr=0.003, wd=1e-4, 2 epochs)...")
baseline = FNN(input_dim=input_dim)
baseline, base_time = train_model(baseline, train_data, n_epochs=2, lr=0.003, weight_decay=1e-4)
base_train_acc = train_accuracy(baseline)
base_acc, _ = evaluate(baseline)
print(f"Baseline  | time: {base_time:.2f}s | train acc: {base_train_acc:.4f} | test acc: {base_acc:.4f}")

print("\nTraining single dropout model (p=0.5, 0.3)...")
dropout_model = DropoutFNN(input_dim=input_dim, p1=0.5, p2=0.3)
dropout_model, drop_time = train_model(dropout_model, train_data, n_epochs=10, lr=0.003, weight_decay=1e-4)
drop_train_acc = train_accuracy(dropout_model)
drop_acc, _ = evaluate(dropout_model)
print(f"Dropout   | time: {drop_time:.2f}s | train acc: {drop_train_acc:.4f} | test acc: {drop_acc:.4f}")


print("PART 2: Bagging ensemble of dropout models")

def make_bootstrap_loader(X, y, batch_size=128, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.integers(0, n, size=n)
    return DataLoader(TensorDataset(X[idx], y[idx]),
                      batch_size=batch_size, shuffle=True)

dropout_configs = [
    (0.3, 0.2),
    (0.4, 0.2),
    (0.5, 0.3),
    (0.6, 0.3),
    (0.5, 0.4),
]

models = []
bag_total_time = 0.0
for i, (p1, p2) in enumerate(dropout_configs):
    print(f"\nModel {i+1}/5: dropout=({p1}, {p2})")
    loader = make_bootstrap_loader(X_train_tensor, y_train_tensor, seed=i)
    m = DropoutFNN(input_dim=input_dim, p1=p1, p2=p2)
    m, t = train_model(m, loader, n_epochs=10, lr=0.003, weight_decay=1e-4)
    bag_total_time += t
    tr_acc = train_accuracy(m)
    te_acc, _ = evaluate(m)
    print(f"  time: {t:.2f}s | train acc: {tr_acc:.4f} | test acc: {te_acc:.4f}")
    models.append(m)

all_preds = np.array([_batched_predict(m, X_test_tensor).numpy() for m in models])
final_preds = np.array([np.bincount(col).argmax() for col in all_preds.T])
ensemble_test_acc = (final_preds == y_test_tensor.numpy()).mean()

all_train_preds = np.array([_batched_predict(m, X_train_tensor).numpy() for m in models])
final_train_preds = np.array([np.bincount(col).argmax() for col in all_train_preds.T])
ensemble_train_acc = (final_train_preds == y_train_tensor.numpy()).mean()


print(f"{'Model':<32}{'Time (s)':>12}{'Train Acc':>14}{'Test Acc':>12}")
print(f"{'Baseline (no dropout)':<32}{base_time:>12.2f}{base_train_acc:>14.4f}{base_acc:>12.4f}")
print(f"{'Single dropout model':<32}{drop_time:>12.2f}{drop_train_acc:>14.4f}{drop_acc:>12.4f}")
print(f"{'Bagging ensemble (5 models)':<32}{bag_total_time:>12.2f}{ensemble_train_acc:>14.4f}{ensemble_test_acc:>12.4f}")