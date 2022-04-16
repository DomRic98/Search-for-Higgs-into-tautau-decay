"""
This file contains some information for the plotting procedure of some physical variables.

@ Author: Domenico Riccardi & Viola Floris

@ Creation Date: 09/04/2022

@ Last Update: 16/04/2022
"""

import numpy as np
plot = {
        'nGoodJets': {
                'title': 'Number of jets',
                'xlabel': r'$N_{jets}$',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 5.0, 6)
        },
        'PV_npvs': {
                'title': 'Number of primary vertices',
                'xlabel': 'NPV',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 30.0, 30)
        },
        'muon_pt': {
                'title': 'Muon transverse momentum',
                'xlabel': r'$p_{T}^{\mu}$ [GeV]',
                'ylabel': 'Frequency/GeV',
                'bins': np.linspace(17.0, 70.0, 15)
        },
        'tau_pt': {
                'title': 'Tau transverse momentum',
                'xlabel': r'$p_{T}^{\tau}$ [GeV]',
                'ylabel': 'Frequency/GeV',
                'bins': np.linspace(17.0, 70.0, 15)
        },
        'muon_eta': {
                'title': 'Muon pseudorapidity',
                'xlabel': r'$\eta^{\mu}$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-3.0, 3.0, 15)
        },
        'tau_eta': {
                'title': 'Tau pseudorapidity',
                'xlabel': r'$\eta^{\tau}$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-3.0, 3.0, 15)
        },
        'muon_phi': {
                'title': 'Muon angular distribution',
                'xlabel': r'$\phi^{\mu}$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-3.0, 3.0, 15)
        },
        'tau_phi': {
                'title': 'Tau angular distribution',
                'xlabel': r'$\phi^{\tau}$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-3.0, 3.0, 15)
        },
        'muon_iso': {
                'title': 'Muon isolation',
                'xlabel': 'Muon ISO',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 0.1, 20)
        },
        'tau_iso': {
                'title': 'Tau isolation',
                'xlabel': 'Tau ISO',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 0.1, 20)
        },
        'jbtag_1': {
                'title': 'Leading jet b-tag',
                'xlabel': r'$j^{lead}$ b-tag',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 1.0, 15)
        },
        'jbtag_2': {
                'title': 'Trailing jet b-tag',
                'xlabel': r'$j^{trail}$ b-tag',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 1.0, 15)
        },
        'MET_phi': {
                'title': 'Missing pT (phi)',
                'xlabel': r'$\phi^{MET}$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-3.14, 3.14, 30)
        },
        'MET_pt': {
                'title': 'Missing pT',
                'xlabel': r'$p_{T}^{MET}$',
                'ylabel': 'Frequency',
                'bins': np.linspace(0, 60, 30)
        },
        'm_vis': {
                'title': 'Visible di-tau mass',
                'xlabel': r'$m^{\tau\tau}$ [GeV]',
                'ylabel': 'Frequency/GeV',
                'bins': np.linspace(20.0, 130.0, 15)
        },
        'pt_vis': {
                'title': 'Visible di-tau pT',
                'xlabel': r'$p_{T}^{\tau\tau}$ [GeV]',
                'ylabel': 'Frequency/GeV',
                'bins': np.linspace(0.0, 60.0, 15)
        },
        'ptjj': {
                'title': 'Di-jet pT',
                'xlabel': r'$p_{T}^{jj}$ [GeV]',
                'ylabel': 'Frequency/GeV',
                'bins': np.linspace(20.0, 200.0, 15)
        },
        'jdeta': {
                'title': 'Di-jet Delta eta',
                'xlabel': r'$\Delta\eta$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-8.0, 8.0, 15)
        },
}
