#coding=utf8
import pandas as pd
import sys
sys.path.append('../../')
from sklearn.ensemble import RandomForestClassifier
from utils.util import Model
import numpy as np
from utils import util
import logging
from utils import log
logger = logging.getLogger('app.rf')
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
import numpy as np

from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from utils import handle_data
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score






class LR(Model):
    def __init__(self, is_load_data=False):
        if is_load_data:
            self.data = util.get_x_y()

        self.p = {'C': 1.5,
                 'class_weight': None,
                 'dual': False,
                 'fit_intercept': True,
                 'intercept_scaling': 1,
                 'max_iter': 200,
                 'multi_class': 'ovr',
                 'n_jobs': -1,
                 'penalty': 'l2',
                 'random_state': None,
                 'solver': 'liblinear',
                 'tol': 0.0001,
                 'verbose': False,
                 'warm_start': False}


        #after tune  c=1.8
        self.p['C'] = 1.7
        self.model = LogisticRegression(**self.p)

    def do_hold_out(self):
        pass
        index = 1
        self.data = handle_data.get_hold_out_X_y(index)

        train_x = self.data['train_x']
        train_y = self.data['train_y']


        test_x = self.data['test_x']
        test_y = self.data['test_y']


        logger.info('start fit')
        self.model.fit(train_x, train_y)
        pred = self.model.predict_proba(test_x)
        
        score = roc_auc_score(test_y,pred)
        logger.info('index %s score %s'%(index,score))

        self.data = handle_data.get_test_X_y()

        test_x = self.data['test_x']
        test_id = self.data['test_id']


        
        p = self.model.predict_proba(test_x)
        self.write_submission(test_id, p)

    def do_tune(self):
        data = handle_data.get_hold_out_X_y(1)
        test_x = data['test_x']
        test_y = data['test_y']

        train_x  = data['train_x']
        train_y = data['train_y']



        def objective(args):

            
            para = {'C': 1.0,
                 'class_weight': None,
                 'dual': False,
                 'fit_intercept': True,
                 'intercept_scaling': 1,
                 'max_iter': 100,
                 'multi_class': 'ovr',
                 'n_jobs': 1,
                 'penalty': 'l2',
                 'random_state': None,
                 'solver': 'liblinear',
                 'tol': 0.0001,
                 'verbose': 0,
                 'warm_start': False}

             
            para.update(args)

            self.model =  LogisticRegression(**para)
            self.model.fit(train_x, train_y)

            pred = self.model.predict_proba(test_x)

            score = log_loss(test_y,pred,eps=1e-15, normalize=True)

            logger.info('score %s'%(score))
            
            rsv = {
                'loss': score,
                'status': STATUS_OK,
                # -- store other results like this
                }
            rsv.update(args)
            print args


            return rsv

        space = {
        'C' : hp.quniform('C', 0.1,3, 0.1),
        'penalty':  hp.choice('penalty', ['l1', 'l2'])
        }
        trials = Trials()

        best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=40,trials=trials)
        rinfo = trials.results
        df = pd.DataFrame(rinfo)

        df.to_csv('./tune.csv',index=False)
        print best_sln
        

    def do_submit(self, CALIB=False):
        #self.data = handle_data.get_x_y()
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
        self.model.fit(X, y)
        pre = self.model.predict_proba(test_x)[:,1]
        logger.info(self.model)

        
        self.write_submission(test_id, pre, 'lr_test_l1.csv')



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

            pred = self.model.predict_proba(test_x)[:,1]

            lr_train.iloc[te_ind] = pred.reshape(-1,1)
            score = roc_auc_score(test_y,pred)

            logger.info('index %s, this score %s'%(index_cv, score))
            index_cv += 1

        lr_train.to_csv('lr_train_l1.csv', index=False)
        #lr_train = np.clip(p, 1e-15, 1-1e-15)
        return y, lr_train
        
        score = roc_auc_score(y, lr_train.values)

        logger.info('cv socre %s' %score)


        return self
    def do_cv(self):
        from sklearn.metrics import precision_score


        data  = util.get_x_y_for_cv()
        X = data['x']
        y = data['y']['Survived']
        skf = StratifiedKFold(y, n_folds=5)
        e_test = []
        e_train = []
        for train_idx, test_idx in skf:
            train_x = X.iloc[train_idx]
            train_y = y.iloc[train_idx]

            test_x = X.iloc[test_idx]
            test_y = y.iloc[test_idx]

            self.model.fit(train_x, train_y)
            yhat = self.model.predict(test_x)
            e_test.append(precision_score(test_y, yhat))


            yhat_train = self.model.predict(train_x)
            e_train.append(precision_score(train_y, yhat_train))

        print np.mean(e_train)
        print np.mean(e_test)



if __name__ == '__main__':
    pass
    
    lr = LR()
    lr.data = handle_data.get_X_y()
    m = lr.do_submit()
    
    

    

