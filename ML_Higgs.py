import uproot
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Dropout
import keras_tuner

MLfiles = ["GluGluToHToTauTau",
           "VBF_HToTauTau",
           "DYJetsToLL",
           "TTbar",
           "W1JetsToLNu",
           "W2JetsToLNu",
           "W3JetsToLNu"]

VARS = ["nGoodJets", "PV_npvs",
        "muon_pt", "muon_eta", "muon_phi", "muon_m", "muon_iso", "mt_mu",  # muon variables
        "tau_pt", "tau_eta", "tau_phi", "tau_m", "tau_iso", "mt_tau",  # tau variables
        "jpt_1", "jeta_1", "jphi_1", "jm_1", "jbtag_1",  # leading jet variables
        "jpt_2", "jeta_2", "jphi_2", "jm_2", "jbtag_2",  # trial jet variables
        "MET_pt", "MET_phi", "m_vis", "pt_vis", "dRmu_tau", "jj_m", "jj_pt", "jj_delta",  # high level variables
        "weight"]


def read_root(choice):
    df = {}
    for file in MLfiles:
        events = uproot.open(file + "_selected.root" + ":Events")
        df[file] = pd.DataFrame(events.arrays(VARS, library="np"), columns=VARS)
        print(f"\tProcessing: {file}, Number of events: {len(df[file])}")
        if choice == "VBF_HToTauTau":
            df[file] = df[file][df[file]["jpt_1"] > 0]
        if file == choice:
            df[file]["event"] = 1.
        else:
            df[file]["event"] = 0.
    return pd.concat(df.values(), ignore_index=True)


def correlations(data):
    """Calculate pairwise correlation between features.

    Extra arguments are passed on to DataFrame.corr()
    """
    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr()
    heatmap1 = plt.pcolor(corrmat)  # get heatmap
    plt.colorbar(heatmap1)  # plot colorbar

    plt.title("Correlations")  # set title
    x_variables = corrmat.columns.values  # get variables from data columns

    plt.xticks(np.arange(len(x_variables)) + 0.5, x_variables, rotation=60)  # x-tick for each label
    plt.yticks(np.arange(len(x_variables)) + 0.5, x_variables)  # y-tick for each label

def ANN_model(hp):
    # Creating Artificial Neural Network (with 3 hidden layers)

    input = Input(shape=(NUM_VARS,), name='input')
    hidden = Dense(NUM_VARS * 15, name='hidden1', activation='selu')(input)
    hidden = Dropout(rate=0.1)(hidden)
    hidden = Dense(NUM_VARS * 2, name='hidden2', activation='selu')(hidden)
    hidden = Dropout(rate=0.1)(hidden)
    hidden = Dense(NUM_VARS, name='hidden3', activation='selu')(hidden)
    hidden = Dropout(rate=0.1)(hidden)
    output = Dense(1, name='output', activation='sigmoid')(hidden)
if __name__ == "__main__":
    print("Which file would you process? [ggH o VBF]")
    choice = input()
    print(f"***************** {choice} is our signal channel *****************")
    if choice == "ggH":
        choice = "GluGluToHToTauTau"
        events = read_root(choice)
    elif choice == "VBF":
        choice = "VBF_HToTauTau"
        events = read_root(choice)
    else:
        print("This choice isn't present")
        exit()
    print(f"Number of signal (1) and background events (0):\n{events.event.value_counts()}")

    # Renormalization of events
    sig = events["event"] == 1.
    bgk = events["event"] == 0.
    weight_sig_sum = events["weight"][sig].sum(axis=0)
    weight_bkg_sum = events["weight"][bgk].sum(axis=0)
    events["weight"][sig] = events["weight"][sig] / weight_sig_sum
    events["weight"][bgk] = events["weight"][bgk] / weight_bkg_sum

    ML_VARS = ["nGoodJets", "PV_npvs",
               "muon_pt", "muon_eta", "muon_phi", "muon_m", "muon_iso", "mt_mu",  # muon variables
               "tau_pt", "tau_eta", "tau_phi", "tau_m", "tau_iso", "mt_tau",  # tau variables
               "MET_pt", "MET_phi", "dRmu_tau",  # high level variables
    ]
    correlations(events[ML_VARS])  # plot correlation matrix

    print("\n****************** ARTIFICIAL NEURAL NETWORK ******************")
    NUM_VARS = len(ML_VARS)
    # Generating features' matrix
    features = events.filter(VARS)
    X = np.asarray(features.values).astype(np.float32)
    # Generating target vector
    target = events.filter(['event'])
    y = np.asarray(target.values).astype(np.float32)
    # Generating event weights vector
    weight = events.filter(['weight'])
    W = np.asarray(weight.values).astype(np.float32)

    # Splitting into training and testing dataset
    X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(X, y, W, test_size=0.3, shuffle=True)
    print(f'Number of events for training phase: {len(X_train)}')
    print(f'Number of events for test phase: {len(X_test)}')

    # Performing features scaling
    SC = StandardScaler()
    X_train = SC.fit_transform(X_train)
    X_test = SC.transform(X_test)



    ANN = Model(inputs=input, outputs=output, name='ANN')  # Initialising ANN
    ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],
                weighted_metrics=['accuracy'])  # Compiling ANN
    ANN.summary()  # Printing the model summary