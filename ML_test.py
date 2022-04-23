import unittest
import os
from ML_Higgs import read_root


class TestMLfiles(unittest.TestCase):
    def setUp(self):
        self.root_files_list = ["GluGluToHToTauTau", "VBF_HToTauTau", "DYJetsToLL",
                                "TTbar", "W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu"]
        self.signal_ggH = 4492
        self.signal_VBF = 5326
        self.bkg_TTbar = 13630

        self.plots_list = ["nGoodJets_VBF", "PV_npvs_ggH", "MET_phi_ggH", "pt_vis_VBF",
                           "muon_pt_VBF", "muon_eta_ggH", "muon_phi_VBF", "muon_iso_VBF",
                           "tau_pt_ggH", "tau_eta_VBF", "tau_phi_ggH"]

    def test_files_exist(self):
        """
        Check on the existence of the file comparing with those in the root files list.

        :return: If it passed ok, if it failed: "There aren't all ROOT files".
        """
        print("TEST 1 - Called...")
        PATH = "ROOT_workspace/"
        results = []
        for file in self.root_files_list:
            results.append(os.path.isfile(PATH + file + "_selected.root"))
        self.assertTrue(all(results), "There aren't all ROOT files")

    def test_read_root_ggH(self):
        """
        Check on the number of signal events in the ggH channel.

        :return: If it passed ok, if it failed: "Must be 4492".
        """
        print("TEST 2 - Called...")
        dataframe = read_root("GluGluToHToTauTau")
        value_counts = dataframe['event'].value_counts().to_numpy()
        self.assertEqual(value_counts[1], self.signal_ggH, "Must be 4492")

    def test_read_root_VBF(self):
        """
        Check on the number of signal events in the VBF channel.

        :return: If it passed ok, if it failed: "Must be 5326".
        """
        print("TEST 3 - Called...")
        dataframe = read_root("VBF_HToTauTau")
        value_counts = dataframe['event'].value_counts().to_numpy()
        self.assertEqual(value_counts[1], self.signal_VBF, "Must be 5326")

    def test_read_root_TTbar(self):
        """
        Check on the number of background events, e.g. in TTbar channel.

        :return: If it passed ok, if it failed: "Must be 13630".
        """
        print("TEST 4 - Called...")
        dataframe = read_root("TTbar")
        value_counts = dataframe['event'].value_counts().to_numpy()
        self.assertEqual(value_counts[1], self.bkg_TTbar, "Must be 13630")

    def test_plots(self):
        """
        Check on the names of the saved plots comparing with those in plots list.

        :return: If it passed ok, if it failed: "name_file not found!".
        """
        print("TEST 5 - Called...")
        for file in self.plots_list:
            self.assertTrue(os.path.exists(f"ML_plots/{file}.pdf"),
                            f"{file} not found!")


if __name__ == '__main__':
    unittest.main(verbosity=2)
