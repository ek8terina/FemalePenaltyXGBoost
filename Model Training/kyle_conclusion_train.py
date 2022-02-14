import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from training import rmse_how_training_went, test_predictions
from statistics import mean

data = pd.read_csv("../Data/train_test_data/conclusions_kyle.csv")
data = data.drop(columns=['title'])
indices = data['ArticleID']  # ArticleID
count = 0
for key in data.keys():
    print(str(count) + ": " + key)
    count = count + 1
features = data.iloc[:, 1:390]  # see note above for list (note: 390 is not included as per split obj format)
regressand = data[["ArticleID", "hedge_conclusion_change"]]  # keep ArticleID just for ease

# set seed for reproducability
seed = 42  # change for experimentation
test_split = .2  # full data size = ..

# split
x_train, x_test, y_train, y_test = train_test_split(features, regressand, test_size=test_split, random_state=seed)
# create train and validation set
test = [(x_train.iloc[:, 1:], y_train.iloc[:, 1]), (x_test.iloc[:, 1:], y_test.iloc[:, 1])]
# train XGBRegessor
model_2 = XGBRegressor()
# optional: add early_stopping_rounds=z param
# verbose: print how model is doing
model_2.fit(x_train.iloc[:, 1:], y_train.iloc[:, 1], eval_metric="rmse", eval_set=test, verbose=True)

# show how training went
save_fig = "../Figures/kyle_conclusion/model_selection.png"
rmse_how_training_went(model_2, save_file=save_fig)

# look at prediction for test set
y_pred = model_2.predict(x_test.iloc[:, 1:])
mse = mean_squared_error(list(y_test.iloc[:, 1]), list(y_pred))
print("Here is MSE of this XGBoost model: " + str(mse))
test = {'ArticleID': y_test.iloc[:, 0], 'y_test': y_test.iloc[:, 1], 'prediction': y_pred}
test['avg_y_test'] = mean(y_test.iloc[:, 1])
difference = np.subtract(np.array(list(y_pred)), np.array(list(y_test.iloc[:, 1])))
sq_diff = [x**2 for x in list(difference)]
test['error'] = difference
test['squared_error'] = sq_diff
# quick classification
results_as_classifier = []
accuracy = []
for idx in range(len(test['y_test'])):
    if test['prediction'][idx] == 0:
        res = 0
        results_as_classifier.append(0)
    elif test['prediction'][idx] > 0:
        res = 1
        results_as_classifier.append(1)
    else:
        res = -1
        results_as_classifier.append(-1)
    if (res == 0) & (y_test.iloc[idx, 1] == 0):
        accuracy.append(1)
    elif (res == 1) & (y_test.iloc[idx, 1] > 0):
        accuracy.append(1)
    elif (res == -1) & (y_test.iloc[idx, 1] < 0):
        accuracy.append(1)
    else:
        accuracy.append(0)
test['results_as_classifier'] = results_as_classifier
test['rough_accuracy'] = accuracy
print("final accuracy if converting to classifer:")
print(str(sum(accuracy)/len(accuracy)))
test = pd.DataFrame.from_dict(test).sort_values(by=['squared_error'], ascending=False)
test.to_csv("../Results/kyle_conclusion_test.csv")
# save model
model_2.save_model("../Models/kyle_conclusion.txt")       # save as .txt, smaller than .json format

# create graph of test predictions
x_ax = range(len(y_test.iloc[:, 1]))
test_predictions(x_ax, y_test=y_test.iloc[:, 1], y_pred=y_pred, save_file="../Figures/kyle_conclusion/test_results.png")
