# Searches for the Higgs boson in the tau-tau decay channel

### Description
===============
Welcome to this repository! 
Here, we find all the material produced for the final project (CMEPDA course at the University of Pisa). Running the python and C++ scripts you will find in the folders, you can explore a little part, of a more complex analysis, on the search of Higgs boson decay into two tau leptons.
Other details are available in pdf format in the Abstracts directory. 

### Instructions
================
Thanks to the CMS collaboration is possible to work on Simulated and Real data for the process $` H\rightarrow\tau_{\mu}\tau_{h} `$ ([CMS Open Data](http://opendata.web.cern.ch/record/12350)).
If you are instaled on your machine the ROOT framework is possible runs the C++ code, present in this repository, to perform the first selection on the data samples:
```bash
$ root -l
$ .L analysis.cpp
$ main()
```
or you can run the selection procedure directly execute the executable file
```bash
$./analysis
```
As a prerogative, it is necessary to download the Open Data from [CMS Open Data link](http://opendata.web.cern.ch/record/12350) or their reduced version from [link](https://root.cern/files/HiggsTauTauReduced/).
