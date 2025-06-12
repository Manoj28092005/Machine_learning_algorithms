import numpy as np 
class Logistic_reg:
    def __init__(self, learning_rate=0.01, n_iters=1000):

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias=None
    def fit(self,x,y):
        n_samples,n_features=x.shape 
        self.weights=np.zeros(n_features)
        self.bias=0
        for _ in range(self.n_iters):
            z=np.dot(x,self.weights)+self.bias
            y_predicted=self.sigmoid(z)
            dw=1/n_samples*np.dot(x.T,(y_predicted-y))
            db=1/n_samples*np.sum(y_predicted-y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def sigmoid(self,z):

        return 1/(1+np.exp(-z))
    def predict(self,x):
        z=np.dot(x,self.weights)+self.bias            
        y_predicted=self.sigmoid(z)
        class_label=[1 if i>0.5 else 0 for i in y_predicted]
        return np.array(class_label)
        