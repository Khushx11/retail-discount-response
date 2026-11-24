"""Regression checks for the calendar leakage and prediction-time availability defects."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import build_features
from pipeline import fit_schema, transform, NUM, CAT, wape

class TimingTests(unittest.TestCase):
    def setUp(self):
        self.s=pd.DataFrame(dict(external_code=[1,1,2],retail=[1,2,1],release_date=['2018-10-01','2018-11-05','2019-05-06'],season=['AW18','AW18','SS19'],category=['a']*3,color=['red']*3,fabric=['cotton']*3))
        self.d=self.s[['external_code','retail']].copy();self.d['price']=.5
        for w in map(str,range(12)):self.s[w]=2.;self.d[w]=.1
        dates=pd.to_datetime(self.s.release_date).dt.isocalendar()
        self.r=pd.DataFrame(dict(external_code=self.s.external_code,retail=self.s.retail,year=dates.year.to_numpy(),week=dates.week.to_numpy(),qty=[30,30,30]))
        self.g=pd.DataFrame({'date':pd.date_range('2017-01-02',periods=150,freq='W-MON'),'a':np.arange(150.)})
    def build(self):
        with patch.object(build_features,'load',return_value=(self.s,self.d,self.r,self.g)):return build_features.build()
    def test_target_and_peer_sales_not_in_features(self):
        before=self.build().query('row == 0 and t == 2')[NUM+CAT].reset_index(drop=True)
        self.s.loc[0,'2']=999.;self.s.loc[1,[str(i) for i in range(12)]]=777.
        after=self.build().query('row == 0 and t == 2')[NUM+CAT].reset_index(drop=True)
        pd.testing.assert_frame_equal(before,after)
    def test_target_week_deliveries_not_in_features(self):
        before=self.build().query('row == 0 and t == 2')[NUM+CAT].reset_index(drop=True)
        self.r.loc[len(self.r)]=[1,1,2018,42,100]
        after=self.build().query('row == 0 and t == 2')[NUM+CAT].reset_index(drop=True)
        pd.testing.assert_frame_equal(before,after)
    def test_strict_label_boundaries(self):
        L=self.build()
        for a,b in [('train','val'),('val','test')]:self.assertLessEqual(L[L.split==a].label_end.max(),L[L.split==b].forecast_date.min())
    def test_train_vocab_unknown_and_finite_transform(self):
        L=self.build();tr=L[L.split=='train'];schema=fit_schema(tr);x=L[L.split=='test'].copy();x['category']='unseen'
        X,C=transform(x,schema);self.assertTrue(np.isfinite(X).all());self.assertTrue((C[:,0]==0).all());self.assertIsNone(wape([0,0],[1,2]))

if __name__=='__main__':unittest.main()
