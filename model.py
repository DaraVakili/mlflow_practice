from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

X,y = load_boston(return_X_y=True)
X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=0.2)
model = LinearRegression()
model.fit(X_train,y_train)
output = model.predict(X_test)

score = accuracy_score(y_test, output)

with mlflow.start_run():
    mlflow.log_metric("score", score)
