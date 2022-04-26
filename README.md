![GitHub](https://github.com/DomRic98/Searches-for-the-Higgs-boson-in-the-tau-tau-decay-channel/actions/workflows/python-app.yml/badge.svg)
![GitHub](https://img.shields.io/github/license/DomRic98/Searches-for-the-Higgs-boson-in-the-tau-tau-decay-channel?logo=github)
![GitHub](https://img.shields.io/github/languages/count/DomRic98/Searches-for-the-Higgs-boson-in-the-tau-tau-decay-channel?logo=github)
![GitHub last commit](https://img.shields.io/github/last-commit/DomRic98/Searches-for-the-Higgs-boson-in-the-tau-tau-decay-channel?logo=GitHub)
![Lines of code](https://img.shields.io/tokei/lines/github/DomRic98/Searches-for-the-Higgs-boson-in-the-tau-tau-decay-channel?logo=github)
![GitHub top language](https://img.shields.io/github/languages/top/DomRic98/Searches-for-the-Higgs-boson-in-the-tau-tau-decay-channel?logo=github)

# Searches for the Higgs boson in the tau-tau decay channel
<div align="center">
    <img src="https://cds.cern.ch/record/2725256/files/mt2.png?subformat=icon-1440" alt="Image" width="600" height="300" />
    <p >Event display of a p-p collision recorded at CMS with a candidate Higgs boson decaying into two tau leptons.<br>[https://cds.cern.ch/record/2725256]</p>
</div>

### Description

Welcome to this repository! 

Here, you find all the material produced for the final project for Computing Methods for Experimental Physics and Data Analysis exam at the University of Pisa. Running the Python and C++ scripts that you can find in the folders, you can explore a little part of a more complex analysis on the search of Higgs boson decay into two tau leptons.

Thanks to the CMS collaboration is possible to work on Simulated and Real data for the Higgs decay process ([CMS Open Data](http://opendata.web.cern.ch/record/12350)). In order to find Higgs bosons candidates, a selection procedure of events has implemented in the `analysis.cpp` code. Run it to produce the dataset for the Machine Learning algorithms, an Artificial Neural Network and a Random Forest, that execute a signal/background classification. 
Other details are available in pdf format in the Materials directory. 

### Code Documentation

Click on the [link](https://domric98.github.io/Searches-for-the-Higgs-boson-in-the-tau-tau-decay-channel/) for visiting the documentation website (produced via Sphinx generator).

### <img src="https://img.icons8.com/color/32/000000/c-plus-plus-logo.png"/> Instructions

If you have the ROOT framework in your machine, you can run the C++ code, located in the ROOT_workspace foder, to perform the first selection on the data samples:
```bash
$ cd ROOT_framework
$ root -l
$ .L analysis.cpp
$ main()
```

As a precondition, it is necessary to download the Open Data from [CMS Open Data link](http://opendata.web.cern.ch/record/12350) or their reduced version from this [link](https://root.cern/files/HiggsTauTauReduced/). In the Download directory, you can find some Python scripts to download automatically same reduced files via a sequential procedure and parallel ones.
After these steps, you will produce the `_selected.root` files and executing the command
```bash
$ root -l
$ .L plotting.cpp
$ main()
```
you will produce the distribution plots of a lot of variables.

### <img src="https://img.icons8.com/color/32/000000/python--v2.png"/> Instructions

To execute the ML programs, it is necessary to install some Python libraries, as Pandas, Uproot, Keras, etc. (if you haven't installed them yet). Otherwise you can run `requirements.txt`, downloading them automatically.

In the main directory, you find `ML_Higgs.py` file and run it with the command:
```bash
$ python ML_Higgs.py
```
Through this step, you will train two ML algorithms with supervised procedure and you will produce the predictions on a test dataset (all plots will be saved in ML_plots folder).
