"""
This file contains some information for the plotting procedure of some physical variables

@ Authors: Domenico Riccardi & Viola Floris

@ Creation Date: 09/04/2022

@ Last Update: 22/04/2022
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
                'xlabel': 'Number PV',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 30.0, 30)
        },
        'muon_pt': {
                'title': 'Muon transverse momentum',
                'xlabel': r'$p_{T}(\mu)$ [GeV]',
                'ylabel': 'Frequency/(1.7 GeV)',
                'bins': np.linspace(17.0, 70.0, 31)
        },
        'tau_pt': {
                'title': 'Tau transverse momentum',
                'xlabel': r'$p_{T}(\tau)$ [GeV]',
                'ylabel': 'Frequency/(1.7 GeV)',
                'bins': np.linspace(17.0, 70.0, 31)
        },
        'muon_eta': {
                'title': 'Muon pseudorapidity',
                'xlabel': r'$\eta(\mu)$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-2.1, 2.1, 31)
        },
        'tau_eta': {
                'title': 'Tau pseudorapidity',
                'xlabel': r'$\eta(\tau)$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-2.1, 2.1, 31)
        },
        'muon_phi': {
                'title': 'Muon angular distribution',
                'xlabel': r'$\phi(\mu)$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-3.14, 3.14, 31)
        },
        'tau_phi': {
                'title': 'Tau angular distribution',
                'xlabel': r'$\phi(\tau)$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-3.14, 3.14, 31)
        },
        'muon_iso': {
                'title': 'Muon isolation',
                'xlabel': 'Muon ISO',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 0.1, 21)
        },
        'tau_iso': {
                'title': 'Tau isolation',
                'xlabel': 'Tau ISO',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 0.1, 21)
        },
        'jbtag_1': {
                'title': 'Leading jet b-tag',
                'xlabel': r'$j^{leading}$ b-tag',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 1.0, 31)
        },
        'jbtag_2': {
                'title': 'Trailing jet b-tag',
                'xlabel': r'$j^{trailing}$ b-tag',
                'ylabel': 'Frequency',
                'bins': np.linspace(0.0, 1.0, 31)
        },
        'MET_phi': {
                'title': 'Missing pT (phi)',
                'xlabel': r'$\phi^{MET}$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-3.14, 3.14, 31)
        },
        'MET_pt': {
                'title': 'Missing pT',
                'xlabel': r'$p_{T}^{MET} [GeV]$',
                'ylabel': 'Frequency/(0.31 GeV)',
                'bins': np.linspace(0, 100, 31)
        },
        'm_vis': {
                'title': 'Visible di-tau mass',
                'xlabel': r'$m^{\tau_{\mu}\tau_{h}}$ [GeV]',
                'ylabel': 'Frequency/(3.8 GeV)',
                'bins': np.linspace(20.0, 140.0, 31)
        },
        'pt_vis': {
                'title': 'Visible di-tau pT',
                'xlabel': r'$p_{T}^{\tau_{\mu}\tau_{h}}$ [GeV]',
                'ylabel': 'Frequency/(1.9 GeV)',
                'bins': np.linspace(0.0, 60.0, 31)
        },
        'jj_pt': {
                'title': 'Di-jet pT',
                'xlabel': r'$p_{T}^{jj}$ [GeV]',
                'ylabel': 'Frequency/(6.4 GeV)',
                'bins': np.linspace(0.0, 200.0, 31)
        },
        'jj_delta': {
                'title': 'Di-jet Delta eta',
                'xlabel': r'$\Delta\eta$',
                'ylabel': 'Frequency',
                'bins': np.linspace(-9.4, 9.4, 31)
        },
}
