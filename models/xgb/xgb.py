#coding=utf8
import sys
import os
sys.path.append('../../')
from utils import log
import utils
import logging
logger = logging.getLogger('app.xgb')

import pandas as pd
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
import pickle
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from utils.util import Model
from utils import util, handle_data
from sklearn.cross_validation import StratifiedKFold
import numpy as np

from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score







class Xgb(Model):
    def __init__(self):
        self.p =  {'base_score': 0.5,
             'colsample_bylevel': 1,
             'colsample_bytree': 0.3,
             'gamma': 0,
             'learning_rate': 0.04,
             'max_delta_step': 0,
             'max_depth': 8,
             'min_child_weight': 5,
             'missing': None,
             'n_estimators': 1500,
             'nthread': 4,
             'objective': 'binary:logistic',
             'reg_alpha': 0,
             'reg_lambda': 700,
             'scale_pos_weight': 15,
             'seed': 0,
             'silent': False,
             'subsample': 0.7}
        #self.p['n_estimators'] = 100
        self.model = xgb.XGBClassifier(**self.p)
        print 'data done!'


    def do_submit(self, CALIB=False):
        if CALIB:

            tp = {}
            tp['base_estimator'] = self.model
            tp['method'] = 'sigmoid'
            tp['cv'] = 5
            self.model = CalibratedClassifierCV(**tp)

        X = self.data['train_x']
        y = self.data['train_y']
        test_x = self.data['test_x']
        test_id = self.data['test_id']
        '''
        self.model.fit(X, y)
        pre = self.model.predict_proba(test_x)[:,1]
        logger.info(self.model)

        
        self.write_submission(test_id, pre, 'xgb_test_l1.csv')
        '''


        #retrain

        kf = StratifiedKFold(y, 5, random_state=555)
        sub_col = handle_data.get_col()

        lr_train = pd.DataFrame(index=range(len(X)), columns=sub_col)

        index_cv = 0


        for tr_ind, te_ind in kf:
            train_x  = X.iloc[tr_ind]
            train_y = y.iloc[tr_ind]

            test_x = X.iloc[te_ind]
            test_y = y.iloc[te_ind]
            self.model.fit(train_x, train_y)

            pred = self.model.predict_proba(test_x)[:,1].reshape(-1,1)
            lr_train.iloc[te_ind] = list(pred)
            score = roc_auc_score(test_y,pred)

            logger.info('index %s, this score %s'%(index_cv, score))
            index_cv += 1

        lr_train.to_csv('xgb_train_l1.csv', index=False)
        lr_train = np.clip(lr_train, 1e-15, 1 - 1e-15)
        score = roc_auc_score(y, lr_train.values)

        logger.info('cv socre %s' %score)

        return self   
def main(para):
        #data = DH.load_data()
        data = para['data']

        MODEL = para['do_what'] #hold_out  submission  cv tune
        CALIB = para['callib']
        BAG = para['bag']

        xgb_para = {'base_score': 0.5,
             'colsample_bylevel': 1,
             'colsample_bytree': 0.3,
             'gamma': 0,
             'learning_rate': 0.04,
             'max_delta_step': 0,
             'max_depth': 8,
             'min_child_weight': 5,
             'missing': None,
             'n_estimators': 2,
             'nthread': 4,
             'objective': 'binary:logistic',
             'reg_alpha': 0,
             'reg_lambda': 700,
             'scale_pos_weight': 15,
             'seed': 0,
             'silent': True,
             'subsample': 0.7}
       

        xgb_para['n_estimators'] = 1500

        clf = xgb.XGBClassifier(**xgb_para)

        if MODEL == 'tune':

            def objective(args):
                print args
                rf_para = {'bootstrap': True,
                 'class_weight': None,
                 'criterion': 'gini',
                 'max_depth': None,
                 'max_features': 'auto',
                 'max_leaf_nodes': None,

                 'min_samples_leaf': 1,
                 'min_samples_split': 2,
                 'min_weight_fraction_leaf': 0.0,

                 'n_estimators': 10,
                 'n_jobs': 1,
                 'oob_score': False,
                 'random_state': None,
                 'verbose': 0,
                 'warm_start': False}
                n_estimators,max_depth, max_features= args
                rf_para['n_estimators'] = int(n_estimators) + 1
                rf_para['max_depth'] = int(max_depth)
                rf_para['max_features'] = max_features
                rf_para['random_state'] = 4
                rf_para['n_jobs'] = 4
                rf_para['class_weight'] = 'balanced'
                clf = RandomForestClassifier(**rf_para)

                clf.fit(data['val_train_x'],data['val_train_y'])
                pred = clf.predict_proba(data['val_test_x'])
                score = roc_auc_score(data['val_test_y'],pred[:,1])*-1.

                rsv = {
                    'loss': score,
                    'status': STATUS_OK,
                    # -- store other results like this
                    'n_estimators': n_estimators,
                    'max_depth':max_depth,
                    }
                return rsv
            space = (
                hp.quniform('n_estimators',100,1000,100),
                hp.quniform('max_depth', 4, 10, 1),
                hp.quniform('max_features', 0.1,1., 0.2)
            )
            trials = Trials()

            best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=40,trials=trials)
            rinfo = trials.results
            df = pd.DataFrame(rinfo)
            df.to_csv('./tune.csv',index=False)
            print best_sln

        if MODEL == 'submission':
            print 'working'
            if CALIB:

                tp = {}
                tp['base_estimator'] = clf
                tp['method'] = 'sigmoid'
                tp['cv'] = 5
                clf = CalibratedClassifierCV(**tp)


            if BAG:
                bagp = {'base_estimator': None,
                         'bootstrap': True,
                         'bootstrap_features': False,
                         'max_features': 1.0,
                         'max_samples': 1.0,
                         'n_estimators': 10,
                         'n_jobs': 1,
                         'oob_score': False,
                         'random_state': None,
                         'verbose': 0,
                         'warm_start': False}
                bagp['random_state'] = 4
                bagp['base_estimator'] = clf
                bagp['n_jobs'] = 4
                clf = BaggingClassifier(**bagp)
            
            print 'fiting'

            clf.fit(data['train_x'],data['train_y'])
            test_x = data['test'].drop('id',axis=1)
            p = clf.predict_proba(test_x)[:,0]
            test = data['test']
            test['score'] = p
            
            utils.save_result('xgb_pre.csv', test[['id','score']])
            logger.info('submission done!')



        if MODEL == 'hold_out':


            if CALIB:
                tp = {}
                tp['base_estimator'] = clf
                tp['method'] = 'sigmoid'
                tp['cv'] = 5
                tp['cv'] = StratifiedKFold(data['val_train_y'], n_folds=5,shuffle=True,\
                 random_state=4)

                clf = CalibratedClassifierCV(**tp)

            if BAG:
                bagp = {'base_estimator': None,
                         'bootstrap': True,
                         'bootstrap_features': False,
                         'max_features': 1.0,
                         'max_samples': 1.0,
                         'n_estimators': 10,
                         'n_jobs': 1,
                         'oob_score': False,
                         'random_state': None,
                         'verbose': 0,
                         'warm_start': False}
                bagp['random_state'] = 4
                bagp['base_estimator'] = clf
                bagp['n_jobs'] = 4
                clf = BaggingClassifier(**bagp)

            if not CALIB:
                clf.fit(data['val_train_x'],data['val_train_y'],verbose=True,
                eval_set=[(data['val_test_x'],data['val_test_y'])],eval_metric='auc',early_stopping_rounds=2100)
                print clf.best_score, clf.best_iteration
                rs = 'best_score %f, best_iteration %d ' %(clf.best_score, clf.best_iteration)
                logger.info(rs)
                logger.info('score %f, model: %s'%(clf.best_score, clf))

            else:
                clf.fit(data['val_train_x'],data['val_train_y'])
                p1 = clf.predict_proba(data['val_test_x'])[:,1]
                score = roc_auc_score(data['val_test_y'],p1)
                logger.info('score %f, model: %s'%(score,clf))


def get_train_data():


   
    train = pd.read_csv('../../data.csv')
    train_x = train.drop(['y'], axis=1)
    train_y = train.y


    test = pd.read_csv('../../test.csv')
   
    rsv = { }
    rsv['train_x'] = train_x
    rsv['train_y'] = 1-train_y
    rsv['test'] = test

    return rsv

def  get_val_data():
    
    rsv = {}
    val_train = pd.read_csv('../../data.csv')
    X = val_train.drop('y', axis=1)
    y = val_train.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rsv['val_train_x'] = X_train
    rsv['val_train_y'] = y_train

    rsv['val_test_x'] = X_test
    rsv['val_test_y'] = y_test

    return rsv

if __name__ == '__main__':
   m = Xgb()
   #m.data = handle_data.get_X_y_test()
   m.data = handle_data.get_X_y()
   m.do_submit()

