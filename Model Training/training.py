from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
Load data
split into test and training

'''
# set seed for reproducability
seed = 42
test_split = .2

'''
basic code
# split and train
x_train, x_test, y_train, y_test = train_test_split(features, regressor, test_size=test_size, random_state=seed)
basic = XGBClassifer()
basic.fit(x_train, y_train)
y_pred = basic.predict(x_test)

# get accuracy
accuracy = accuracy_score(y_test, predictions)
print("Here is accuracy of most basic XGBoost model: " + accuracy)
'''
