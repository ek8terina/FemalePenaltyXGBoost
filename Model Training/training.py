from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

'''
Load data
features: BERT embeddings, gender
regressand: Change in hedge words
split into test and training

'''
# set seed for reproducability
seed = 42
test_split = .2

'''
basic code
# split and train
x_train, x_test, y_train, y_test = train_test_split(features, regressand, test_size=test_size, random_state=seed)
basic = XGBClassifer()
basic.fit(x_train, y_train)
y_pred = basic.predict(x_test)

# get accuracy
accuracy = accuracy_score(y_test, predictions)
print("Here is accuracy of most basic XGBoost model: " + accuracy)
'''

'''
slightly more in-depth
test = [(x_train, y_train), (x_test, y_test)]
model_2 = XGBClassifer()
model_2.fit(x_train, y_train, eval_metric="rmse", eval_set=test, verbose=True)      #print how model is doing

training_results = model_2.evals_result()
fig, ax = pyplot.subplots()
ax.plot(x_axis, training_results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, training_results['validation_1']['rmse'], label='Test')
ax.legend()
pyplot.ylabel('RMSE')
pyplot.title('XGBoost RMSE')
pyplot.show()
'''

