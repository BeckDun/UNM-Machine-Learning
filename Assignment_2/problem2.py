import numpy as np
from sklearn.model_selection import train_test_split
def make_classification(d, n, u, random_seed):
    random_generator = np.random.RandomState(random_seed)
    a = random_generator.normal(size=d)
    X = random_generator.uniform(-u,u,size = (n,d))
    y = np.where(np.dot(X,a) < 0, -1,1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=random_seed)
    return X_train,X_test,y_train,y_test,a

    




