"""
This python script implements ML algorithms for the classification
signal/background of Higgs decay events.

@ Authors: Domenico Riccardi & Viola Floris

@ Creation Date: 09/04/2022

@ Last Update: 22/04/2022
"""

# Import packages/library
import os
import sys
import uproot
import numpy as np
import pandas as pd
from keras import Model, Input
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree

# Import some useful functions from another python script (plot_function.py)
from plot_functions import plotting_ROC, plotting_loss, plotting_accuracy, \
    plotting_output_score, plotting_purity_vs_efficiency, plotting_confusion_matrix, \
    plotting_physical_variables, correlations

pd.options.mode.chained_assignment = None  # default='warn'

MLfiles = ["GluGluToHToTauTau", "VBF_HToTauTau",  # signals
           "DYJetsToLL", "TTbar", "W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu" # backgrounds
           ]
VARS = ["nGoodJets", "PV_npvs", "weight",
        "muon_pt", "muon_eta", "muon_phi", "muon_m", "muon_iso", "mt_mu",
        "tau_pt", "tau_eta", "tau_phi", "tau_m", "tau_iso", "mt_tau",
        "jpt_1", "jeta_1", "jphi_1", "jm_1", "jbtag_1",
        "jpt_2", "jeta_2", "jphi_2", "jm_2", "jbtag_2",
        "MET_pt", "MET_phi", "m_vis", "pt_vis", "dRmu_tau", "jj_m", "jj_pt", "jj_delta",
        ]


def read_root(file_name):
    """
    The function implements the read of root files to prepare the datasets for ML.
    In particular, it uses the uproot library to read the root files and Pandas
    DataFrame to create the dataset.

    :param file_name: The name of the root file to process.

    :return: None
    """
    PATH = "ROOT_workspace/"
    df = {}
    for file in MLfiles:
        Events = uproot.open(PATH + file + "_selected.root" + ":Events")
        df[file] = pd.DataFrame(Events.arrays(VARS, library="np"), columns=VARS)
        print(f"\tProcessing: {file}, Number of events: {len(df[file])}")
        if file == file_name:
            df[file]["event"] = 1.
        else:
            df[file]["event"] = 0.
    return pd.concat(df.values(), ignore_index=True)


def g_mean_threshold(tpr, fpr, thresholds):
    """
    This function calculates the geometric mean for each pair of the first
    two arguments of the function, i.e. tpr, fpr.
    After that, the function finds the threshold with the optimal balance between false
    positive and true positive rates, by looking for the maximum values of the parameters
    that produce the greatest g-mean.

    :param tpr: True Positive Rate value (TPR is also called sensitivity).
    :param fpr: False Positive Rate value. Its inverse is called specificity.
    :param thresholds: Threshold values from ROC curve.

    :return: The index of the largest g-mean
    """
    g_means = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(g_means)  # locate the index of the largest g-mean
    print(f'Best Threshold={thresholds[index]:.3f}, G-Mean={g_means[index]:.3f}')
    return index


def fscore_threshold(purity, recall, thresholds):
    """
    This function calculates a simplify f-score for each pair
    of the first two arguments of the function (purity, recall).
    After that, the function finds optimal threshold that produces
    the best balance between precision and recall,
    by looking for the maximum values of the parameters that produce
    the the greatest simplify f-score.

    :param purity: Vector of values between 0 and 1. This variable is the ability of the classifier \
    not to label as signal an event that is background.
    :param recall: Vector of values between 0 and 1. This variable is the ability of the classifier \
    to find all the signal events.
    :param thresholds: Threshold values from precision-recall curve method.

    :return: The index of the largest f-score.
    """
    fscore = purity[:-1] * recall[:-1]
    index = np.argmax(fscore)
    print(f'Best Threshold={thresholds[index]:.3f}, F-Score simplify={fscore[index]:.3f}')
    return index


if __name__ == "__main__":
    ML_dict = {
        'ggH':
            {
                'complete_name': 'GluGluToHToTauTau',
                # 'ML_VARS' = ["nGoodJets", "PV_npvs", "muon_pt", "muon_eta", "muon_phi", "muon_iso",
                # "tau_pt", "tau_eta", "tau_phi", "tau_iso", "mt_tau", "MET_phi", "pt_vis"],
                'ML_VARS': ["nGoodJets", "PV_npvs", "MET_pt", "MET_phi", "pt_vis",
                            "muon_pt", "muon_eta", "muon_phi", "muon_iso",
                            "tau_pt", "tau_eta", "tau_phi"],
                'number_input': 10,
                'N_EPOCHS': 15,
            },
        'VBF':
            {
                'complete_name': 'VBF_HToTauTau',
                'ML_VARS': ["nGoodJets", "PV_npvs", "MET_phi", "m_vis", "pt_vis",
                            "muon_pt", "muon_eta", "muon_phi", "muon_iso",
                            "tau_pt", "tau_eta", "tau_phi", "tau_iso",
                            "jbtag_1", "jj_delta"],
                'number_input': 15,
                'N_EPOCHS': 10
            }
    }

    # Create a new directory contains ML plots
    NAME_DIR = '/ML_plots'
    PATH = os.getcwd()
    if not os.path.exists(PATH + NAME_DIR):
        os.mkdir(PATH + NAME_DIR)
        print(f">>> ..{NAME_DIR} directory has been created at {PATH}")
    else:
        print(f">>> ..{NAME_DIR} folder is already present at {PATH}")

    # Ask to user what signal channel would like to use
    print("Which file would you process? [ggH o VBF]")
    CHOICE = input()
    print(f"***************** {CHOICE} is our signal channel *****************")
    if (CHOICE != "ggH") and (CHOICE != "VBF"):
        print("This CHOICE isn't present")
        sys.exit()

    events = read_root(ML_dict[CHOICE]['complete_name'])
    NUM_VARS = len(ML_dict[CHOICE]['ML_VARS'])
    # Remove undefined variables
    for i in range(NUM_VARS):
        events = events[(events[VARS[i]] > -999)]

    print(f"Number of signal (1) and background events (0):\n{events.event.value_counts()}")

    # Renormalization of events
    sig = events["event"] == 1.
    bgk = events["event"] == 0.
    weight_sig_sum = events["weight"][sig].sum(axis=0)
    weight_bkg_sum = events["weight"][bgk].sum(axis=0)
    events["weight"][sig] = events["weight"][sig] / weight_sig_sum
    events["weight"][bgk] = events["weight"][bgk] / weight_bkg_sum

    # Plot correlation matrix of ML variables
    correlations(CHOICE, events[ML_dict[CHOICE]["ML_VARS"]])

    # Generate features' matrix
    features = events.filter(ML_dict[CHOICE]["ML_VARS"])
    X = np.asarray(features.values).astype(np.float32)
    # Generate target vector
    target = events.filter(['event'])
    y = np.asarray(target.values).astype(np.float32)
    # Generate event weights vector
    weight = events.filter(['weight'])
    W = np.asarray(weight.values).astype(np.float32)

    # Split into training and testing dataset
    X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(X, y, W, test_size=0.3, shuffle=True)
    print(f'Number of events for training phase: {len(X_train)}')
    print(f'Number of events for test phase: {len(X_test)}')

    # Perform features scaling with StandardScaler
    SC = StandardScaler()
    X_train = SC.fit_transform(X_train)
    X_test = SC.transform(X_test)

    print("\n******************** ARTIFICIAL NEURAL NETWORK *******************")
    print(f"\tML variables ({NUM_VARS} features) are:")
    for var in ML_dict[CHOICE]['ML_VARS']:
        print(f"\t{var}")

    # Create Artificial Neural Network (with 3 hidden layers)
    input_layer = Input(shape=(NUM_VARS,), name='input')
    hidden = Dense(NUM_VARS * ML_dict[CHOICE]['number_input'], name='hidden1', activation='selu')(input_layer)
    hidden = Dropout(rate=0.1)(hidden)
    hidden = Dense(NUM_VARS * 2, name='hidden2', activation='selu')(hidden)
    hidden = Dropout(rate=0.5)(hidden)
    hidden = Dense(NUM_VARS, name='hidden3', activation='selu')(hidden)
    hidden = Dropout(rate=0.1)(hidden)
    output_layer = Dense(1, name='output', activation='sigmoid')(hidden)
    # Initialising ANN model
    ANN = Model(inputs=input_layer, outputs=output_layer, name='ANN') 
    ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],
                weighted_metrics=['accuracy'])  # Compiling ANN
    ANN.summary()  # Print on screen the model summary

    # Fit the ANN model
    HISTORY = ANN.fit(X_train, y_train, sample_weight=W_train, batch_size=20,
                      epochs=ML_dict[CHOICE]["N_EPOCHS"], verbose=1, validation_split=0.3)

    plotting_loss(CHOICE, HISTORY.history['loss'], HISTORY.history['val_loss'])
    plotting_accuracy(CHOICE, HISTORY.history['accuracy'], HISTORY.history['val_accuracy'])

    # Get ANN model event predictions
    y_prediction_test = ANN.predict(X_test)
    y_prediction_train = ANN.predict(X_train)

    # Get False Positive Rate (FPR), True Positive Rate (TPR) and Thresholds/Cut on the ANN's score
    fpr_test, tpr_test, thresholds_test = roc_curve(y_true=y_test,
                                                    y_score=y_prediction_test,
                                                    sample_weight=W_test)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_true=y_train,
                                                       y_score=y_prediction_train,
                                                       sample_weight=W_train)

    # Optimal Threshold for ROC Curve with g-mean
    index_g = g_mean_threshold(tpr_test, fpr_test, thresholds_test)


    # Plot the ANN ROC curve on the test and training datasets
    # with plotting_ROC function implemented in plot_function.py script
    roc_auc_test = auc(fpr_test, tpr_test)
    roc_auc_train = auc(fpr_train, tpr_train)
    plotting_ROC(f'ANN_{CHOICE}', fpr_test, tpr_test, fpr_train, tpr_train,
                 thresholds_test, index_g, roc_auc_test, roc_auc_train)

    # Get precision (purity) and recall (signal efficiency) with precision_recall_curve method.
    purity_test, recall_test, t_test = precision_recall_curve(y_true=y_test,
                                                              probas_pred=y_prediction_test,
                                                              sample_weight=W_test)
    purity_train, recall_train, t_train = precision_recall_curve(y_true=y_train,
                                                                 probas_pred=y_prediction_train,
                                                                 sample_weight=W_train)

    # Optimal threshold for Precision-Recall Curve (PR) with simplify f-score 
    index_f = fscore_threshold(purity_test[:-1], recall_test[:-1], t_test)

    plotting_purity_vs_efficiency(f'ANN_{CHOICE}', t_test, t_train, recall_test,
                                  recall_train, purity_test, purity_train, index_f)

    # Threshold selection according to signal channel
    cut_dnn = 0.
    if CHOICE == "ggH":
        cut_dnn = thresholds_test[index_g]
    else:
        cut_dnn = t_test[index_f]

    # Plotting output score distribution
    plotting_output_score(f'ANN_{CHOICE}', y_train, y_prediction_train, y_test,
                          y_prediction_test, cut_dnn)

    # Transform predictions into an array of 0, 1 
    filter_test_sig = y_prediction_test >= cut_dnn  # classify as signal
    filter_test_bkg = y_prediction_test < cut_dnn  # classify as background
    y_prediction_test[filter_test_sig] = 1
    y_prediction_test[filter_test_bkg] = 0

    # Metrics values for the ANN algorithm having fixed an ANN score threshold
    accuracy = accuracy_score(y_test[:, 0], y_prediction_test[:, 0], sample_weight=W_test[:, 0])
    precision = precision_score(y_test[:, 0], y_prediction_test[:, 0], sample_weight=W_test[:, 0])
    recall = recall_score(y_test[:, 0], y_prediction_test[:, 0], sample_weight=W_test[:, 0])
    print(f'Threshold on the ANN output : {cut_dnn:.3f}')
    print(f'ANN Test Accuracy: {accuracy:.3f}')
    print(f'ANN Test Precision/Purity: {precision:.3f}')
    print(f'ANN Test Sensitivity/Recall/TPR/Signal Efficiency: {recall:.3f}')
    plotting_confusion_matrix(f'ANN_{CHOICE}', y_test[:, 0], y_prediction_test[:, 0], W_test[:, 0])


    print("************************************* RANDOM FOREST **************************************")

    # Uncomment the following lines to perform the tuning of hyper-parameters of the Random Forest Classifier
    """
    classifier = RandomForestClassifier(random_state=7, verbose=1)
    grid_param = {
        # 'criterion': ['gini', 'entropy'],
        'n_estimators': [300, 500],
        'bootstrap': [True, False],
        'max_depth': [5, 7],
        # 'max_features': [10, None],
    }
    tuner_RF = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='accuracy', cv=3, n_jobs=-1)
    tuner_RF.fit(X_train, np.ravel(y_train))
    print(f"Best parameters: {tuner_RF.best_params_}")
    print(f"Best metrics score {tuner_RF.best_score_}")
    """

    # Create a RF classifier
    RFC = RandomForestClassifier(n_estimators=500, criterion='gini', verbose=1, max_depth=5,
                                 max_features=None, bootstrap=True, n_jobs=-1)

    # Train the model on the training dataset
    randomforest = RFC.fit(X_train, np.ravel(y_train), np.ravel(W_train))

    # Get RF model event predictions on the test and training dataset.
    # Return a vector of 0s and 1s (the two labels of the classification problem)
    y_prediction_test_rf = RFC.predict(X_test)
    y_prediction_train_rf = RFC.predict(X_train)

    # Method predict_proba return a 2D vector.
    # The first column is the probability of the event being 'bkg'
    # and the second is the probability of it being 'sig'
    y_prediction_test_prob = randomforest.predict_proba(X_test)
    y_prediction_train_prob = randomforest.predict_proba(X_train)

    print(f"Accuracy of the RF model: {accuracy_score(y_test, y_prediction_test_rf):.3f} (test dataset)")
    print(f"Accuracy of the RF model: {accuracy_score(y_train, y_prediction_train_rf):.3f} (train dataset)")

    fpr_test_rf, tpr_test_rf, thresholds_test_rf = roc_curve(y_true=y_test,
                                                             y_score=y_prediction_test_prob[:, -1],
                                                             sample_weight=W_test)
    fpr_train_rf, tpr_train_rf, thresholds_train_rf = roc_curve(y_true=y_train,
                                                                y_score=y_prediction_train_prob[:, -1],
                                                                sample_weight=W_train)

    # Calculate area under the ROC curve
    roc_auc_test_rf = auc(fpr_test_rf, tpr_test_rf)
    roc_auc_train_rf = auc(fpr_train_rf, tpr_train_rf)

    index_g_RF = g_mean_threshold(tpr_test_rf, fpr_test_rf, thresholds_test_rf)
    plotting_ROC(f'RF_{CHOICE}', fpr_test_rf, tpr_test_rf, fpr_train_rf, tpr_train_rf,
                 thresholds_test_rf, index_g_RF, roc_auc_test_rf, roc_auc_train_rf)

    # Get precision and recall
    purity_test_rf, recall_test_rf, t_test_rf = precision_recall_curve(y_true=y_test,
                                                                       probas_pred=y_prediction_test_prob[:, -1],
                                                                       sample_weight=W_test)
    purity_train_rf, recall_train_rf, t_train_rf = precision_recall_curve(y_true=y_train,
                                                                          probas_pred=y_prediction_train_prob[:, -1],
                                                                          sample_weight=W_train)

    index_f_RF = fscore_threshold(purity_test_rf, recall_test_rf, t_test_rf)

    # Threshold selection according to signal channel
    cut_RF = 0.
    if CHOICE == "ggH":
        cut_RF = thresholds_test_rf[index_g_RF]
    else:
        cut_RF = t_test_rf[index_f_RF]

    plotting_purity_vs_efficiency(f'RF_{CHOICE}', t_test_rf, t_train_rf,
                                  recall_test_rf, recall_train_rf, purity_test_rf, purity_train_rf, index_f_RF)
    plotting_output_score(f'RF_{CHOICE}', y_train[:, 0], y_prediction_train_prob[:, 1], y_test[:, 0],
                          y_prediction_test_prob[:, 1], cut_RF)

    # Transform predictions into an array of 0, 1
    filter_test_sig_rf = y_prediction_test_prob[:, -1] >= cut_RF  # classify as signal
    filter_test_bkg_rf = y_prediction_test_prob[:, -1] < cut_RF  # classify as background
    y_prediction_test_rf[filter_test_sig_rf] = 1
    y_prediction_test_rf[filter_test_bkg_rf] = 0

    # Metrics values for the ANN algorithm having fixed an ANN score threshold
    accuracy_rf = accuracy_score(y_test[:, 0], y_prediction_test_rf, sample_weight=W_test[:, 0])
    precision_rf = precision_score(y_test[:, 0], y_prediction_test_rf, sample_weight=W_test[:, 0])
    recall_rf = recall_score(y_test[:, 0], y_prediction_test_rf, sample_weight=W_test[:, 0])
    print(f'Threshold on the RF output : {cut_RF:.3f}')
    print(f'RF Test Accuracy: {accuracy_rf:.3f}')
    print(f'RF Test Precision/Purity: {precision_rf:.3f}')
    print(f'RF Test Sensitivity/Recall/TPR/Signal Efficiency: {recall_rf:.3f}')
    plotting_confusion_matrix(f'RF_{CHOICE}', y_test[:, 0], y_prediction_test_rf, W_test[:, 0])

    # Draw a tree of the forest
    # estimator = RFC.estimators_[10]
    # fig = plt.figure()
    # _ = tree.plot_tree(estimator, filled=True)
    # fig.savefig("decision_tree.pdf",  format='pdf', bbox_inches='tight')

    X_test = SC.inverse_transform(X_test)
    # Plotting of ML variables with models predictions
    for s in ML_dict[CHOICE]['ML_VARS']:
        index_var = ML_dict[CHOICE]['ML_VARS'].index(s)
        plotting_physical_variables(CHOICE, s, X_test[:, index_var], y_test[:, 0], y_prediction_test, y_prediction_test_rf)

