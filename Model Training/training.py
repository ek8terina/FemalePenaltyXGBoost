import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd


def rmse_how_training_went(model, save_file=None, ylab="RMSE", xlab="Epochs", title="XGBoost RMSE"):
    training_results = model.evals_result()
    epochs = len(training_results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, training_results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, training_results['validation_1']['rmse'], label='Test')
    ax.legend()
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title)
    if save_file is not None:
        plt.savefig(save_file)
    # plt.show()
    plt.clf()

def test_predictions(x_ax, y_test, y_pred, save_file, title = "Abstract hedge change test and predicted data"):
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig(save_file)


'''
Load data
features: SBERT embeddings, gender, publicationYear, published word number, original word number
regressand: Change in hedge words
for print of keys, add this: 
count = 0
for key in data.keys():
   print(str(count) + ": " + key)
   count = count + 1
stop

data = pd.read_csv("../Data/train_test_data/abstracts_kyle.csv")
indices = data['ArticleID']  # ArticleID
features = data.iloc[:, 1:390]  # see note above for list (note: 390 is not included as per split obj format)
regressand = data[["ArticleID", "hedge_abstract_change"]]  # keep ArticleID just for ease

# set seed for reproducability
seed = 42  # change for experimentation
test_split = .2  # data size = 3250

# split
x_train, x_test, y_train, y_test = train_test_split(features, regressand, test_size=test_split, random_state=seed)
# load model
loaded = XGBRegressor()
loaded.load_model("../Models/basic_model.txt")
# create train and validation set
test = [(x_train.iloc[:, 1:], y_train.iloc[:, 1]), (x_test.iloc[:, 1:], y_test.iloc[:, 1])]
# train XGBRegessor
model_2 = XGBRegressor()
# optional: add early_stopping_rounds=z param
# verbose: print how model is doing
model_2.fit(x_train.iloc[:, 1:], y_train.iloc[:, 1], eval_metric="rmse", eval_set=test, verbose=True)

# show how training went
training_results = model_2.evals_result()
epochs = len(training_results['validation_0']['rmse'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, training_results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, training_results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.ylabel('Epochs')
plt.title('XGBoost RMSE')
plt.savefig("../Figures/kyle_abstract/model_selection.png")
# plt.show()
plt.clf()

y_pred = model_2.predict(x_test.iloc[:, 1:])
mse = mean_squared_error(list(y_test.iloc[:, 1]), list(y_pred))
print("Here is MSE of this XGBoost model: " + str(mse))
test = {'ArticleID': y_test.iloc[:, 0], 'y_test': y_test.iloc[:, 1], 'prediction': y_pred}
difference = np.subtract(np.array(list(y_pred)), np.array(list(y_test.iloc[:, 1])))
sq_diff = [x**2 for x in list(difference)]
test['error'] = difference
test['squared_error'] = sq_diff
test = pd.DataFrame.from_dict(test).sort_values(by=['squared_error'], ascending=False)
test.to_csv("/Users/efedorov/Downloads/basic_model_mse.csv")
# save model
model_2.save_model("../Models/kyle_abstract.txt")       # save as .txt, smaller than .json format

x_ax = range(len(y_test.iloc[:, 1]))
plt.plot(x_ax, y_test.iloc[:, 1], label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Abstract hedge change test and predicted data")
plt.legend()
# plt.show()
plt.savefig("../Figures/kyle_abstract/test_results.png")
'''