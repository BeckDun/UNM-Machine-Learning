import numpy as np
class LinearSVC:

    def __init__(self, eta = 0.1,n_iter = 50, random_state=1,lambda_param = .01):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.lambda_param = lambda_param

        

    def fit(self,X,y): 
        random_generator = np.random.RandomState(self.random_state)
        self.w_ = random_generator.normal(loc=0.0, scale=.1, size = X.shape[1])
        self.b_ = 0.0

        for _ in range(self.n_iter):
            for xi,target in zip(X,y):

                margin = target * self.net_input(xi)

                if margin >= 1:
                    self.w_ -= self.eta * (2.0 * self.lambda_param * self.w_)
                
                else:
                    self.w_ -= self.eta * (2.0 * self.lambda_param * self.w_ - target * xi)
                    self.b_ += self.eta * target
        
        return self


        
    
    def net_input(self,X):
        return np.dot(X,self.w_) + self.b_
    
    def predict(self,X):
        return np.where(self.net_input(X) >= 0,1,-1)
    