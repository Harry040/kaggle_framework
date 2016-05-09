import pandas as pd
import sys
sys.path.append('../../')
from sklearn.ensemble import RandomForestClassifier
from utils.util import Model
from utils import util, handle_data

import logging
from utils import log
logger = logging.getLogger('app.svm')
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold




class SVM(Model):
	def __init__(self, is_load_data=False):
		if is_load_data:
			self.data = util.get_x_y()

		self.p = {'C': 1.0,
				 'cache_size': 200,
				 'class_weight': None,
				 'coef0': 0.0,
				 'decision_function_shape': None,
				 'degree': 3,
				 'gamma': 'auto',
				 'kernel': 'rbf',
				 'max_iter': -1,
				 'probability': True,
				 'random_state': None,
				 'shrinking': True,
				 'tol': 0.001,
				 'verbose': True}


		
		self.model = SVC(**self.p)

	def do_hold_out(self):
		pass

		self.data = handle_data.get_hold_out_X_y()

		train_x = self.data['train_x']
		train_y = self.data['train_y']


		test_x = self.data['test_x']
		test_y = self.data['test_y']


		logger.info('start fit')
		self.model.fit(train_x, train_y)
		pred = self.model.predict_proba(test_x)
		score = log_loss(test_y,pred,eps=1e-15, normalize=True)

		logger.info('score %s'%score)






		self.data = handle_data.get_test_X_y()

		test_x = self.data['test_x']
		test_id = self.data['test_id']

		
		p = self.model.predict_proba(test_x)
		self.write_submission(test_id, p)
		



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
		self.model.fit(X, y)
		pre = self.model.predict_proba(test_x)
		logger.info(self.model)

		
		self.write_submission(test_id, pre, 'svm_test_l1.csv')



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

			pred = self.model.predict_proba(test_x)

			lr_train.iloc[te_ind] = pred
			score = log_loss(test_y,pred,eps=1e-15, normalize=True)

			logger.info('index %s, this score %s'%(index_cv, score))
			index_cv += 1

		lr_train.to_csv('svm_train_l1.csv', index=False)
		score = log_loss(y, lr_train.values, eps=1e-15, normalize=True)


		logger.info('cv socre %s' %score)




	def do_tune(self):
		data = util.get_x_y_for_cv()
		X = data['x']
		y = data['y']['Survived']


		def objective(args):

			print args
			para = {'C': 1.0,
				 'cache_size': 200,
				 'class_weight': None,
				 'coef0': 0.0,
				 'decision_function_shape': None,
				 'degree': 3,
				 'gamma': 'auto',
				 'kernel': 'rbf',
				 'max_iter': -1,
				 'probability': False,
				 'random_state': 4,
				 'shrinking': False,
				 'tol': 0.001,
				 'verbose': False}

			
			
			para.update(args)

			

			clf = SVC(**para)

			score = cross_val_score(clf, X, y, n_jobs=4) * -1
			score = np.mean(score)

			rsv = {
				'loss': score,
				'status': STATUS_OK,
				# -- store other results like this
				
				}
			rsv.update(args)
			
			return rsv

		space = {

		'C':hp.quniform('C', 0.1,3, 0.1),
		#'tol':hp.quniform('tol', 0.001, 0.005, 0.001),
		#'gamma': hp.quniform('gamma', 0.1, 1, 0.1),
		'kernel': hp.choice('kernal',[ 'linear','rbf','sigmoid'])
		}
		trials = Trials()

		best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=40,trials=trials)
		rinfo = trials.results
		df = pd.DataFrame(rinfo)

		df.to_csv('./tune.csv',index=False)
		print best_sln

if __name__ == '__main__':
	svm = SVM()
	svm.data = handle_data.get_X_y_test()
	m = svm.do_submit()
