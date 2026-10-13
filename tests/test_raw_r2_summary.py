import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pfm_nsi import cli


class RawR2SummaryTests(unittest.TestCase):
    def test_diagnostics_survive_summary_export(self):
        qc = {'NSI': {'MedianScore': .3, 'Ridge': {'Lambda10': {
            'R2': np.array([.2, .4, np.nan]),
            'RawR2Diagnostics': {'negative_count': 1, 'negative_fraction': 1/3,
                                 'nonfinite_count': 0, 'raw_min': -.2}}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = cli._save_nsi_summary_json(directory, 'example', qc)
            result = json.loads(Path(path).read_text())
        self.assertEqual(result['raw_r2_negative_count'], 1)
        self.assertEqual(result['raw_r2_negative_fraction'], 1/3)
        self.assertEqual(result['raw_r2_min'], -.2)
        self.assertAlmostEqual(result['mean_nsi'], .3)
        self.assertEqual(result['median_nsi'], .3)


if __name__ == '__main__':
    unittest.main()
