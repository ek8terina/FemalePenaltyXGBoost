import matplotlib.pyplot as plt


# create plot of training validation
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
    plt.clf()


# create plot of test set vs predictions
def test_predictions(x_ax, y_test, y_pred, save_file=None, title="Abstract hedge change test and predicted data"):
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title(title)
    plt.legend()
    if save_file is not None:
        plt.savefig(save_file)
    plt.clf()
