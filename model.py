import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mlflow
import argparse 
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import datetime

parser = argparse.ArgumentParser(description="mlFlowArguments")
parser.add_argument('--model', type=str, nargs='+',
                    help='chooses the model', required=True)
                    
argument = parser.parse_args() 

argument = argument.model[0]

if argument == 'linearRegression':
    model = LinearRegression()

elif argument == 'decisionTree':
    model = tree.DecisionTreeRegressor()

elif argument == 'randomForestRegressor':
    model = RandomForestRegressor()

elif argument == 'sGDRegressor':
    model = SGDRegressor()

elif argument == 'kNeighborsRegressor':
    model = KNeighborsRegressor()




with mlflow.start_run():
    run_time_name = '~/documents/mlflowPractice/savedModels/mlflow_model_' + str(datetime.datetime.now())
    X,y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    model.fit(X_train,y_train)
    output = model.predict(X_test)
    score = r2_score(y_test, output)
    mlflow.log_metric("score", score)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model"
    )

    mlflow.sklearn.save_model(model, run_time_name)
