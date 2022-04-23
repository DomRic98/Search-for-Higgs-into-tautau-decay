"""
This python script implements some useful functions for ML_Higgs.py code

@ Authors: Domenico Riccardi & Viola Floris

@ Creation Date: 09/04/2022

@ Last Update: 22/04/2022
"""
# Import packages/library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mplhep as hep
# Import the ML_infoplot.py file that contains some information for the plotting of variables.
import ML_infoplot

# The name of the directory in which save the plots.
NAME_DIR = 'ML_plots'


def correlations(channel_name, data):
    """
    The function plots the correlation between ML variables.

    :param channel_name: Name of signal variable.
    :param data: Dataset of variables that we characterise.

    :return: The correlation's plot.
    """
    plt.figure()
    corrmat = data.corr()
    heatmap1 = plt.pcolor(corrmat)  # get heatmap
    plt.colorbar(heatmap1)  # plot colorbar

    plt.title(f'Correlations, {channel_name}')  # set title
    x_variables = corrmat.columns.values  # get variables from data columns

    plt.xticks(np.arange(len(x_variables)) + 0.5, x_variables, rotation=60)  # x-tick for each label
    plt.yticks(np.arange(len(x_variables)) + 0.5, x_variables)  # y-tick for each label
    plt.savefig(f'{NAME_DIR}/correlation_{channel_name}.pdf', format='pdf', bbox_inches='tight')
    plt.clf()


def plotting_loss(channel_name, loss, val_loss):
    """
    The function plots the loss function on training and validation set vs epochs.

    :param channel_name: Name of signal variable.
    :param loss: The history of loss function during training phase.
    :param val_loss: The history of loss function on validation set.

    :return: The loss functions plot comparison.
    """
    plt.figure()
    plt.plot(loss, label='loss train', color='red')
    plt.plot(val_loss, label='loss validation', color='green')
    plt.title(f'Loss function ANN, {channel_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(f'{NAME_DIR}/Loss_ANN_{channel_name}.pdf', format='pdf')
    plt.clf()


def plotting_accuracy(channel_name, accuracy, val_accuracy):
    """
    This plot shows the ANN accuracy on training and validation set vs epochs,
    with accuracy defined as the number of correctly matches between the predictions and the true labels.

    :param channel_name: Name of signal variable.
    :param accuracy: Accuracy metric on the training set.
    :param val_accuracy: Accuracy metric on the validation set.

    :return: The ANN accuracies plot comparison.
    """
    plt.figure()
    plt.plot(accuracy, label='loss train', color='red')
    plt.plot(val_accuracy, label='loss validation', color='green')
    plt.title(f'Accuracy ANN, {channel_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.savefig(f'{NAME_DIR}/Accuracy_ANN_{channel_name}.pdf', format='pdf')
    plt.clf()


def plotting_ROC(channel_name, fpr_test, tpr_test, fpr_train, tpr_train, thresholds, ix, roc_auc_test, roc_auc_train):
    """
    This function plots the ROC curve (Receiver Operating Characteristic)
    that shows the True Positive Rate vs False Positive Rate compared to random chance.
    The Best Threshold found is also highlighted,
    so this summary visual is used to assess the performance of classification models.

    :param channel_name: ANN or RFC ML algorithm depending on the case.
    :param fpr_test: False Positive Rate on test dataset.
    :param tpr_test: True Positive Rate on test dataset.
    :param fpr_train: False Positive Rate on training dataset.
    :param tpr_train: True Positive Rate on training dataset.
    :param thresholds: Thresholds on the ANN's and RF's score.
    :param ix: Best Threshold's index
    :param roc_auc_test: Area under the ROC curve on test set.
    :param roc_auc_train: Area under the ROC curve on training set.

    :return: The plot of ROC on test and training sets, random chance and Best Threshold.
    """
    names = channel_name.split("_")
    plt.figure()
    plt.plot(fpr_test, tpr_test, color='blue', label=f'AUC test = {roc_auc_test:.3f}')
    plt.plot(fpr_train, tpr_train, color='red', label=f'AUC train = {roc_auc_train:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random Chance')
    plt.scatter(fpr_test[ix], tpr_test[ix], marker='o', color='blue', label=f'Best Threshold = {thresholds[ix]:.3f}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Receiver Operating Characteristic (ROC) {names[0]}, {names[1]}')
    plt.legend(loc='lower right')
    plt.savefig(f'{NAME_DIR}/ROC_{channel_name}.pdf', format='pdf')
    plt.clf()


def plotting_purity_vs_efficiency(channel_name, t_test, t_train, recall_test, recall_train, purity_test, purity_train,
                                  ix):
    """
    The function plots the product of purity (indicates how accurate the model is)
    and efficiency (indicates how properly selective the model is) metrics against the threshold already calculated.
    It compared both test set results and training set results, moreover Best Threshold found is also highlighted.

    :param channel_name: ANN or RFC algorithm depending on the case.
    :param t_test: Threshold already calculated on test set.
    :param t_train: Threshold already calculated on training set.
    :param recall_test: Efficiency values on test set.
    :param recall_train: Efficiency values on training set.
    :param purity_test: Purity values on test set.
    :param purity_train: Purity values on training set.
    :param ix: Best Threshold's index

    :return: The curves of purity times efficiency vs threshold and Best Threshold.
    """
    names = channel_name.split("_")
    plt.figure()
    plt.plot(t_test, recall_test[:-1] * purity_test[:-1], color='blue',
             label=r'Purity $\times$ Efficiency test')
    plt.plot(t_train, recall_train[:-1] * purity_train[:-1], color='red',
             label=r'Purity $\times$ Efficiency train')
    plt.scatter(t_test[ix], recall_test[ix] * purity_test[ix], marker='o', color='blue',
                label=f'Best Threshold = {t_test[ix]:.3f}')
    plt.xlabel(f'Threshold/cut on the {names[0]} score')
    plt.ylabel(r'Purity $\times$ Efficiency')
    plt.title(fr'Purity $\times$ Efficiency vs Threshold {names[0]}, {names[1]}')
    plt.legend(loc='lower left')
    plt.savefig(f'{NAME_DIR}/metrics_PR_{channel_name}.pdf', format='pdf')
    plt.clf()


def plotting_output_score(channel_name, y_train, y_prediction_train, y_test, y_prediction_test, cut_dnn):
    """
    This summary plot shows the output score both with histograms of predictions on
    the training set and with the trend (points) of predictions on the test set.
    The dashed line is the threshold that splits signals and background events.

    :param channel_name: ANN or RFC algorithm depending on the case.
    :param y_train: True label events on training set.
    :param y_prediction_train: Prediction label events on training set.
    :param y_test: True label events on test set.
    :param y_prediction_test: Prediction label events on test set.
    :param cut_dnn: Separating threshold.

    :return: The histograms and trends of output score on training and test set, with highlighted threshold.
    """
    names = channel_name.split("_")
    plt.figure()
    xx = np.linspace(0.0, 1.0, 10)
    plt.hist(y_prediction_train[y_train == 1], bins=xx, density=1, label='Signal (train)', alpha=0.5, color='red')
    plt.hist(y_prediction_train[y_train == 0], bins=xx, density=1, label='Background (train)', alpha=0.5, color='blue')

    hist_test_sig, bin_edges = np.histogram(y_prediction_test[y_test == 1], bins=xx, density=True)
    hist_test_bgk, _ = np.histogram(y_prediction_test[y_test == 0], bins=xx, density=True)

    center = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centres

    scale = len(y_prediction_test[y_test == 1]) / sum(hist_test_sig)
    err_sig = np.sqrt(hist_test_sig * scale) / scale
    plt.errorbar(center, hist_test_sig, yerr=err_sig, label='Signal (test)', c='red', fmt='o')

    scale = len(y_prediction_test[y_test == 0]) / sum(hist_test_bgk)
    err_bgk = np.sqrt(hist_test_bgk * scale) / scale
    plt.errorbar(center, hist_test_bgk, yerr=err_bgk, label='Background (test)', c='blue', fmt='o')
    plt.axvline(x=cut_dnn, ymin=0.005, ymax=0.65, color='orange', label=f'Threshold = {cut_dnn:.3f}', linestyle='--')
    plt.title(f'Output score {names[0]}, {names[1]}')
    plt.xlabel('Output score')
    plt.ylabel('Arbitrary units')
    plt.legend()
    plt.savefig(f'{NAME_DIR}/{channel_name}_output_score.pdf', format='pdf')
    plt.clf()


def plotting_confusion_matrix(channel_name, y_test, y_prediction_test, w_test):
    """
    This function plots the confusion matrix that summarizes the predictions classification
    of the model between various classes.
    The elements, from the top-left side clockwise, represent TN, FP, FN and TP rates on test dataset.

    :param channel_name: ANN or RFC algorithm depending on the case.
    :param y_test: True label events on test set.
    :param y_prediction_test: Predicted label events on test set.

    :return: The confusion matrix with information from the evaluation metrics.
    """
    names = channel_name.split("_")
    plt.figure()
    mat_test = confusion_matrix(y_test, y_prediction_test, sample_weight=w_test, normalize='all')
    sns.heatmap(mat_test.T, square=True, annot=True, cbar=True, linewidths=1, linecolor='black')
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.title(f'Normalized Confusion Matrix for the test dataset {names[0]}, {names[1]}')
    plt.savefig(f'{NAME_DIR}/confusion_matrix_{channel_name}.pdf', format='pdf')
    plt.clf()


def plotting_physical_variables(CHOICE, s, feature, y_test, y_prediction_test, y_prediction_test_rf):
    """
    These plots show histograms of some physical variables (split into signal and background)
    with the ANN's and RFC's predictions.

    :param s: Physical variables name
    :param feature: Physical variable
    :param y_test: Target vector for the test (true labels)
    :param y_prediction_test: ANN's target vector for predictions
    :param y_prediction_test_rf: RF's target vector for predictions

    :return: The physical variables plotted with the ANN's and RFC's predictions.
    """
    plt.style.use(hep.style.CMS)  # or ATLAS/LHCb2
    fig = plt.figure()
    plt.hist(feature[y_test == 0], label='Background', density=1, bins=ML_infoplot.plot[s]['bins'],
             alpha=0.5, linestyle='-')
    plt.hist(feature[y_test == 1], label=f'Signal {CHOICE}', density=1, bins=ML_infoplot.plot[s]['bins'], alpha=0.5,
             linestyle='-')
    plt.hist(feature[y_prediction_test[:, 0] == 1], histtype='step', label='ANN', density=1,
             bins=ML_infoplot.plot[s]['bins'], color='red', linestyle='--')
    plt.hist(feature[y_prediction_test_rf == 1], histtype='step', label='RF', density=1,
             bins=ML_infoplot.plot[s]['bins'], color='green', linestyle='--')
    plt.ylabel(ML_infoplot.plot[s]['ylabel'])
    plt.xlabel(ML_infoplot.plot[s]['xlabel'])
    plt.legend()
    plt.text(0.01, 1.02, r'$\bf{CMS}$ Open Data', transform=plt.gca().transAxes)
    plt.text(0.65, 1.02, r'11.5 $fb^{-1}$ (8 TeV)', transform=plt.gca().transAxes)
    plt.savefig(f'{NAME_DIR}/{s}_{CHOICE}.pdf', format='pdf')
    plt.close(fig)
