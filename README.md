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
Here, we find all the material produced for the final project (CMEPDA course at the University of Pisa). Running the python and C++ scripts you will find in the folders, you can explore a little part, of a more complex analysis, on the search of Higgs boson decay into two tau leptons.
Other details are available in pdf format in the Abstracts directory. 

### Instructions

Thanks to the CMS collaboration is possible to work on Simulated and Real data for the Higgs decay process ([CMS Open Data](http://opendata.web.cern.ch/record/12350)).
If you are instaled on your machine the ROOT framework is possible runs the C++ code, present in this repository, to perform the first selection on the data samples:
```bash
$ root -l
$ .L analysis.cpp
$ main()
```
or you can run the selection procedure directly execute the executable file
```bash
$ ./analysis
```
As a prerogative, it is necessary to download the Open Data from [CMS Open Data link](http://opendata.web.cern.ch/record/12350) or their reduced version from [link](https://root.cern/files/HiggsTauTauReduced/).
With this procedure, you will produce the `_selected.root` files and executing the command
```bash
$ root -l
$ .L plotting.cpp
$ main()
```
or
```bash
$ ./plotting
```
you will produce the distribution plots of a lot of variables.
