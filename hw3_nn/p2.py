# %% [markdown]
# # CS 429 Assignment 3
# Nathan Nail

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

nltk.download('stopwords')
stop = stopwords.words('english')



# %%
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub(r'[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def stem_and_stop_remove(text):
    tokens =  [w for w in tokenizer_porter(text) if w not in stop]
    return " ".join(tokens)


# %%
# 1. transform imdb data into tf-idf form.
# read in data

path = "movie_data.csv"
print("Reading CSV...")
data = pd.read_csv(path)
print("Done!")
print()

X_train = data.loc[:34999, 'review']
y_train = data.loc[:34999, 'sentiment']
X_test = data.loc[35000:, 'review']
y_test = data.loc[35000:, 'sentiment']  

# clean data of HTML formatting (taken from pg. 255) confirmed working. 
print("Beginning preprocessing:")
X_train = X_train.apply(preprocessor)
X_test = X_test.apply(preprocessor)
print("Done!")
print()

print("Beginning stemmer and stop word remover...")
X_train = X_train.apply(stem_and_stop_remove)
X_test = X_test.apply(stem_and_stop_remove)

print("running CountVectorizer...")
cv = CountVectorizer(max_features=20000)
# this needs the data as a list
train_bag = cv.fit_transform(np.array(X_train.tolist()))
test_bag = cv.transform(np.array(X_test.tolist()))
print("Done!")
print()

# do we use the vectorizer or the transformer here? book uses the vectorizer for the log reg code. 
print("running tf-idf...")
tf_idf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
train_bag = tf_idf.fit_transform(train_bag)
test_bag = tf_idf.transform(test_bag)
print("Done!")


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# device = torch.device("cpu")

# torch.manual_seed(1)

X_train_tensor = torch.tensor(train_bag.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)
X_test_tensor = torch.tensor(test_bag.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long)

X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

joint_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_data = DataLoader(joint_dataset, batch_size = 2500, shuffle=True)


# %%
class FNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(20000, 1000)
        self.fc2 = torch.nn.Linear(1000, 250)
        # self.fc3 = torch.nn.Linear(250, 250)
        self.fc3 = torch.nn.Linear(250, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x), dim=1)
        return x

    
net = FNN().to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nn_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)

n_iter = 5
for i in range(n_iter):
    print("epoch", i, ":")

    num_correct = 0
    total = 0

    for (x, y) in train_data:
        x = x.to(device)
        y = y.to(device)
        output = net.forward(x)

        loss = nn_loss(output, y)
        loss.backward()
        optimizer.step()
        net.zero_grad()

        preds = torch.argmax(output, dim=1)
        num_correct += (preds == y).sum().item()
        total += y.size(0)

    
    print("accuracy:", num_correct / total)
    print()


# %%
net.eval()

with torch.no_grad():
    y_pred = net(X_test_tensor)
    pred_labels = torch.argmax(y_pred, dim=1)
    accuracy = (pred_labels == y_test_tensor).float().mean()

    print("accuracy:", accuracy)


