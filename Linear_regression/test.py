from Linear_reg import Linear_reg
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
print("Linear Regression Model Test")
X=np.random.rand(100,1)*10
y=2.5*X.flatten()+np.random.randn(100)*2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model= Linear_reg()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("Mean Squared Error:", model.mean_squared_error(y_test, y_pred))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.show()
