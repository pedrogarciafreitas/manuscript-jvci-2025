"""Scientific invariants for the scoped review calculations."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review_analysis as review


class ReviewStatisticsTests(unittest.TestCase):
    def test_weighted_metrics_equal_explicit_cluster_duplicates_with_ties(self):
        y=np.array([1,1,2,3,3,4.],float)
        p=np.array([2,1,3,3,4,4.],float)
        weights=np.array([[2,2,0,3,3,0],[0,0,4,1,1,2]])
        actual=review.metric_vectors(y,p,weights)
        for i,w in enumerate(weights):
            indices=np.repeat(np.arange(len(y)),w)
            expected=[spearmanr(y[indices],p[indices]).statistic,pearsonr(y[indices],p[indices]).statistic,np.sqrt(np.mean((y[indices]-p[indices])**2))]
            np.testing.assert_allclose(actual[i],expected,atol=1e-14)

    def test_constant_predictions_have_undefined_correlations(self):
        result=review.scalar_metrics(np.array([1.,2,3]),np.ones(3))
        self.assertTrue(np.isnan(result[:2]).all())
        self.assertAlmostEqual(result[2],np.sqrt(5/3))

    def test_holm_uses_whole_family_and_preserves_input_order(self):
        np.testing.assert_allclose(review.holm([.04,.01,.03]),[.06,.03,.06])

    def test_identical_methods_cannot_be_significant(self):
        with patch.object(review,'REPEATS',64):
            p,invalid=review.swapped_tests(np.arange(6.),np.array([0,1,2,4,3,5.]),np.array([0,1,2,4,3,5.]),np.array(['a','a','b','b','c','c']))
        np.testing.assert_array_equal(p,np.ones(3))
        np.testing.assert_array_equal(invalid,np.zeros(3))

    def test_alignment_rejects_missing_samples_and_changed_mos(self):
        base=pd.DataFrame({'SIGNAL':['a','b'],'REF':['r','s'],'SCORE':[1.,2.],'REF_NUM_POINTS':[5,6],'TEST_NUM_POINTS':[4,5],'CLASS':[1,2]})
        np.testing.assert_array_equal(review.align(base,base.iloc[::-1]).SIGNAL,base.SIGNAL)
        with self.assertRaises(AssertionError):
            review.align(base,base.iloc[:1])
        changed=base.copy();changed.loc[0,'SCORE']=9
        with self.assertRaises(AssertionError):
            review.align(base,changed)

    def test_fold_reader_rejects_content_leakage(self):
        base=pd.DataFrame({'SIGNAL':list('abcdef'),'REF':['r','r','s','t','u','v']})
        folds=base.copy();folds['dataset']='APSIPA';folds['fold']=[0,1,2,3,4,0]
        with patch.object(review,'read',return_value=folds),self.assertRaises(AssertionError):
            review.assignments('APSIPA',base)


if __name__=='__main__':
    unittest.main()
