import pandas as pd
import sys
sys.path.append('../../')
from sklearn.ensemble import RandomForestClassifier
from utils.util import Model
from utils import util, handle_data
import logging
from utils import log
logger = logging.getLogger('app.rf')
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
import numpy as np

from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss

class RF(Model):
	def __init__(self, is_load_data=False):
		if is_load_data:
			self.data = util.get_x_y()

		self.p = {'bootstrap': True,
				 'class_weight': None,
				 'criterion': 'gini',
				 'max_depth': None,
				 'max_features': 'auto',
				 'max_leaf_nodes': None,
				 'min_samples_leaf': 1,
				 'min_samples_split': 6,
				 'min_weight_fraction_leaf': 0.0,
				 'n_estimators': 500,
				 'n_jobs': 4,
				 'oob_score': False,
				 'random_state': 10,
				 'verbose': None,
				 'warm_start': False}
		self.model = RandomForestClassifier(**self.p)


	def do_tune(self):
		data = util.get_x_y_for_cv()
		X = data['x']
		y = data['y']['Survived']


		def objective(args):

			
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
			
			score = cross_val_score(clf, X, y, n_jobs = 4) * -1
			score = np.mean(score)

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

		
		self.write_submission(test_id, pre, 'rf_test_l1.csv')



		#retrain

		kf = StratifiedKFold(y, 5, random_state=555)
		sub_col = handle_data.get_col()

		lr_train = pd.DataFrame(index=range(len(X)), columns=sub_col)

		index_cv = 0


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

		lr_train.to_csv('lr_train_l1.csv', index=False)
		score = log_loss(y, lr_train.values, eps=1e-15, normalize=True)


		logger.info('cv socre %s' %score)





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
	rf = RF()
	rf.data = handle_data.get_X_y_test()
	m = rf.do_submit(True)

