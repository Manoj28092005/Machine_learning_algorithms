import numpy as np 
np.random.seed(0)
np.random.seed(0)
x = np.random.randn(100, 2)
y = (x[:, 0] + x[:, 1] > 0).astype(int)
from Logistic_reg import Logistic_reg
model = Logistic_reg(learning_rate=0.01, n_iters=1000)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_predicted)
print(f"Accuracy: {accuracy * 100:.2f}%")
