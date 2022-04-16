"""
This python script implements some useful functions for ML_Higgs.py code
@ Author: Domenico Riccardi & Viola Floris
@ Creation Date: 09/04/2022
@ Last Update: 16/04/2022
"""
# Import packages/library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mplhep as hep
# Import the ML_infoplot.py file that contains some information for the plotting of variables.
import ML_infoplot

# Path of the directory in which memorise the plots
PATH = 'Plot'


def plotting_loss(type, loss, val_loss):
    """
    The function plots the loss function on training and validation set vs epochs.
    :param type: The name of classifier/signal variables.
    :param loss: Loss function during training phase.
    :param val_loss: Loss function on validation set.
    :return: The loss function plot comparison.
    """
    plt.figure()
    plt.plot(loss, label='loss train', color='red')
    plt.plot(val_loss, label='loss validation', color='green')
    plt.title('Loss function ANN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(PATH + f'/Loss_ANN for {type}.pdf', format='pdf')
    plt.clf()
    return None


def plotting_accuracy(type, accuracy, val_accuracy):
    """
    This plot shows the ANN accuracy on training and validation set vs epochs,
    with accuracy defined as the number of correctly matches between the predictions and the true labels.
    :param type: The name of classifier/signal variables.
    :param accuracy: Accuracy metric on the training set.
    :param val_accuracy: Accuracy metric on the validation set.
    :return: The ANN accuracy plot comparison.
    """
    plt.figure()
    plt.plot(accuracy, label='loss train', color='red')
    plt.plot(val_accuracy, label='loss validation', color='green')
    plt.title('Accuracy ANN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.savefig(PATH + f'/Accuracy_ANN for {type}.pdf', format='pdf')
    plt.clf()
    return None


def plotting_ROC(type, fpr_test, tpr_test, fpr_train, tpr_train, thresholds, ix, roc_auc_test, roc_auc_train):
    """
    This function plots the ROC curve (Receiver Operating Characteristic)
    that shows the True Positive Rate vs False Positive Rate compared to random chance.
    The Best Threshold found is also highlighted,
    so this summary visual is used to assess the performance of classification models.
    :param type: ANN or RFC ML algorithm depending on the case.
    :param fpr_test: False Positive Rate on test dataset.
    :param tpr_test: True Positive Rate on test dataset.
    :param fpr_train: False Positive Rate on training dataset.
    :param tpr_train: True Positive Rate on training dataset.
    :param thresholds: Thresholds on the ANN's and RF's score.
    :param ix: Best Threshold's index
    :param roc_auc_test: Area under the ROC curve on test set.
    :param roc_auc_train: Area under the ROC curve on training set.
    :return: plot of ROC on test and training sets, random chance and Best Threshold.
    """
    plt.figure()
    plt.plot(fpr_test, tpr_test, color='blue', label=type + f' AUC_test = {roc_auc_test:.3f}')
    plt.plot(fpr_train, tpr_train, color='red', label=type + f' AUC_train = {roc_auc_train:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')
    plt.scatter(fpr_test[ix], tpr_test[ix], marker='o', color='blue', label=f'Best Threshold={thresholds[ix]:.3f}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(PATH + '/ROC_' + type + '.pdf', format='pdf')
    plt.clf()
    return None


def plotting_purity_vs_efficiency(type, t_test, t_train, recall_test, recall_train, purity_test, purity_train, ix):
    """
    The function plots the product of purity (indicates how accurate the model is)
    and efficiency (indicates how properly selective the model is) metrics against the threshold already calculated.
    It compared both test set results and training set results, moreover Best Threshold found is also highlighted.
    :param type:  ANN or RFC Machine Learning algorithm depending on the case.
    :param t_test: Threshold already calculated on test set.
    :param t_train: Threshold already calculated on training set.
    :param recall_test: Efficiency values on test set.
    :param recall_train: Efficiency values on training set.
    :param purity_test: Purity values on test set.
    :param purity_train: Purity values on training set.
    :param ix: Best Threshold's index
    :return: Curves of purity times efficiency vs threshold and Best Threshold.
    """
    plt.figure()
    plt.plot(t_test, recall_test[:-1] * purity_test[:-1], color='blue',
             label=type + r' purity $\times$ efficiency test')
    plt.plot(t_train, recall_train[:-1] * purity_train[:-1], color='red',
             label=type + r'purity $\times$ efficiency train')
    plt.scatter(t_test[ix], recall_test[ix] * purity_test[ix], marker='o', color='blue',
                label=f'Best Threshold={t_test[ix]:.3f}')
    plt.xlabel(f'Threshold/cut on the {type} score')
    plt.ylabel(r'Purity $\times$efficiency')
    plt.title(fr'Purity $\times$ efficiency vs Threshold on the {type} score')
    plt.legend(loc='lower left')
    plt.savefig(PATH + '/metrics_PR_' + type + '.pdf', format='pdf')
    plt.clf()
    return None


def plotting_output_score(type, y_train, y_prediction_train, y_test, y_prediction_test, cut_dnn):
    """
    This summary plot shows output score both with histograms of predictions on training set
    and with trend (points) of predictions on test set.
    The dashed line is the threshold that splits signals and background events.
    :param type: ANN or RFC Machine Learning algorithm depending on the case.
    :param y_train: True label events on training set.
    :param y_prediction_train: Prediction label events on training set.
    :param y_test: True label events on test set.
    :param y_prediction_test: Prediction label events on test set.
    :param cut_dnn: Separating threshold.
    :return: Histogram and trend of output score on training and test set, with highlighted threshold.
    """
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
    plt.axvline(x=cut_dnn, ymin=0.005, ymax=0.65, color='orange', label=f'Threshold={cut_dnn:.3f}', linestyle='--')
    plt.xlabel(type + ' output score')
    plt.ylabel('Arbitrary units')
    plt.legend(loc='upper center')
    plt.savefig(PATH + '/' + type + ' output score.pdf', format='pdf')
    plt.clf()
    return None


def plotting_confusion_matrix(type, y_test, y_prediction_test, w_test):
    """
    This is the confusion matrix plot that summarizes the predictions classification
    of the model between various classes.
    The table y-axis is the label that the model predicted and the x-axis is the true label.
    So the elements, from the top-left side, represent TN, FP, FN and TP rates on test dataset.
    :param type: ANN or RFC Machine Learning algorithm depending on the case.
    :param y_test: True label events on test set.
    :param y_prediction_test: Predicted label events on test set.
    :return: The confusion matrix with information from the evaluation metrics.
    """
    plt.figure()
    mat_test = confusion_matrix(y_test, y_prediction_test, sample_weight=w_test, normalize='all')
    sns.heatmap(mat_test.T, square=True, annot=True, cbar=True, linewidths=1, linecolor='black')
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.title(f'Normalized Confusion Matrix for the test dataset - {type}')
    plt.savefig(PATH + '/Confusion matrix_' + type + '.pdf', format='pdf')
    plt.close()
    return None


def plotting_physical_variables(s, index_vars, X_test, y_test, y_prediction_test, y_prediction_test_rf):
    """
    These plots show histograms of some physical variables (split into signal and background)
    with the ANN's and RFC's predictions. Some of the plotted physical variables are Mass of Dilepton System,
    Transverse Momentum of Dilepton System, dPhi of Dilepton System, Number of Jets,
    Output multivariate b-tagging algorithm and so on. In addition, we are using the ML_infoplot.py file where
    there are physical variables information.
    :param s: Physical variables name
    :param index_vars: Physical variables index
    :param X_test: Features' matrix
    :param y_test: Target vector for the test
    :param y_prediction_test: ANN's target vector for predictions
    :param y_prediction_test_rf: RF's target vector for predictions
    :return: Some physical variables plotted with the ANN's and RFC's predictions.
    """
    plt.style.use(hep.style.CMS)  # or ATLAS/LHCb2
    fig = plt.figure()
    plt.hist(X_test[:, index_vars][y_test == 0], label='Background', density=1, bins=ML_infoplot.plot[s]['bins'],
             alpha=0.5, linestyle='-')
    plt.hist(X_test[:, index_vars][y_test == 1], label='Signal', density=1, bins=ML_infoplot.plot[s]['bins'], alpha=0.5,
             linestyle='-')
    plt.hist(X_test[:, index_vars][y_prediction_test[:, 0] == 1], histtype='step', label='ANN', density=1,
             bins=ML_infoplot.plot[s]['bins'], color='red', linestyle='--')
    plt.hist(X_test[:, index_vars][y_prediction_test_rf == 1], histtype='step', label='RF', density=1,
             bins=ML_infoplot.plot[s]['bins'], color='green', linestyle='--')
    plt.ylabel(ML_infoplot.plot[s]['ylabel'])
    plt.xlabel(ML_infoplot.plot[s]['xlabel'])
    plt.legend()
    plt.text(0.01, 1.02, r'$\bf{CMS}$ Open Data', transform=plt.gca().transAxes)
    plt.text(0.65, 1.02, r'11.5 $fb^{-1}$ (8 TeV)', transform=plt.gca().transAxes)
    plt.savefig(PATH + '/' + ML_infoplot.plot[s]['title'] + '.pdf', format='pdf')
    plt.close(fig)
    return None


def correlations(data):
    """
    The function plots the correlation between variables.
    :param data: dataset of variables that we characterise.
    :return: the correlations plot.
    """
    corrmat = data.corr()
    heatmap1 = plt.pcolor(corrmat)  # get heatmap
    plt.colorbar(heatmap1)  # plot colorbar

    plt.title("Correlations")  # set title
    x_variables = corrmat.columns.values  # get variables from data columns

    plt.xticks(np.arange(len(x_variables)) + 0.5, x_variables, rotation=60)  # x-tick for each label
    plt.yticks(np.arange(len(x_variables)) + 0.5, x_variables)  # y-tick for each label
    plt.savefig('Plot/correlation.pdf', format='pdf')
