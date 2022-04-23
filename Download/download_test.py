import os
import unittest
from download_sequential import download


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.url = "https://root.cern/files/HiggsTauTauReduced/"
        self.files_list = ["VBF_HToTauTau.root", "GluGluToHToTauTau.root"]
        self.files_dim = [24184554, 20460281]  # bytes

    def test_download_files(self):
        """
        Check on the name of the downloaded files compared to the expected one.

        :return: If it passed ok, if it failed: "The files aren't the same".
        """
        print("TEST 1 - Called...")
        result = []
        dim_result = []
        call = 0
        for file in self.files_list:
            call += 1
            result.append(download(self.url + file, call))
            dim_result.append(os.stat(file).st_size)
        self.assertEqual(result, self.files_list, "The files aren't the same")
        self.assertEqual(dim_result, self.files_dim, "The dimensions aren't the same")


if __name__ == '__main__':
    unittest.main(verbosity=2)
