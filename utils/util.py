
import pandas as pd 
from sklearn import preprocessing
import logging
import log
logger = logging.getLogger('app.util')
import cPickle
import os
from  sklearn.preprocessing import Imputer




def get_submit_data():
    pass

class Model(object):
    def __init__(self):
        pass





    def write_submission(self,ID, pred, fname='submission.csv'):
        #open('submission.csv', 'w').write('PassengerId,Survived\n')
        col = ['TARGET']
        df = pd.DataFrame(pred, columns=col)
        df['ID'] = ID
        df['ID'] = df['ID'].astype(int)
        df = df[['ID'] + col]
        df.to_csv(fname, index=False)


def pre_handle_data():
    """
    get train_x, test_x
    """
    train = pd.read_csv('../train.csv')
    x_col =  [u'Pclass', u'Name', u'Sex', u'Age',u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked']
    y_col = [u'Survived']


   
def get_x_y_for_cv():

    fp = os.path.abspath(__file__)
    dfp = os.path.dirname(os.path.dirname(fp))
    fp = dfp + Config['train_data']
    
    df = cPickle.load(open(fp))
    

    tmp = {}
    tmp['x'] = df.drop(u'Survived', axis=1)
    tmp['y'] = df[[u'Survived']]
    return tmp
    fp = dfp + Config['cv_train_data']
    cPickle.dump(tmp, open(fp, 'w'))
    print 'data for cv done!'




def preprocess():
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')

    #drop name ticket Cabin

    df = train.append(test)
    new_col = [u'PassengerId', u'Survived', u'Pclass', u'Sex', u'Age',u'SibSp', u'Parch',  u'Fare',  u'Embarked']
    df = df[new_col]


    # label the Sex to 0, 1
    label = preprocessing.LabelEncoder()

    df['Sex'] = label.fit_transform(df['Sex'])
    logger.info(label.classes_)


    #process Embarked feature
    df['Embarked'] = label.fit_transform(df['Embarked'])
    logger.info(label.classes_)


    #process missing value
    im = Imputer(strategy='most_frequent', verbose=True)
    mv = im.fit_transform(df[['Embarked', 'Age']])
    df['Embarked'] = mv[:,0]
    df['Age'] = mv[:,1]

    im = Imputer(strategy='median', verbose=True)
    mv = im.fit_transform(df[['Fare']])
    df['Fare'] = mv[:,0]




    train = df[~df.Survived.isnull()]
    logger.info('train %s'%len(train))

    test = df[df.Survived.isnull()]
    logger.info('test %s'%len(test))




    cPickle.dump(train, open('../data/c_train_v1', 'w'))
    cPickle.dump(test, open('../data/c_test_v1', 'w'))


    


def preprocess_v2():
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')
    train = train.dropna()
    test = test.dropna()
    #drop name ticket Cabin

    df = train.append(test)
    new_col = [u'PassengerId', u'Survived', u'Pclass', u'Sex', u'Age',u'SibSp', u'Parch',  u'Fare',  u'Embarked']
    df = df[new_col]

    

    
    train = df[~df.Survived.isnull()]
    logger.info('train %s'%len(train))

    test = df[df.Survived.isnull()]
    logger.info('test %s'%len(test))




    cPickle.dump(train, open('../data/c_train_v1', 'w'))
    cPickle.dump(test, open('../data/c_test_v1', 'w'))


def value_type(series):
    """
    get value type
    
    Parameters
    ---------
    series : pandas Series

    Returns
    -------
    
    rsv : list, types of value
    """
    ul = series.unique()
    tl = []
    for e in ul:
        tl.append(type(e))
    tl = set(tl)
    tl = list(tl)
    tl = map(lambda e: str(e), tl)
    
    return tl


    
def missing_value_info(df):
    
    rsv = []
    
    lng = len(df)
    for e in df.columns:
        tmp = {}
        tmp[u'feature'] = e
        tmp[u'missing_per'] = sum(df[e].isnull()*1./lng)
        tmp['value_type'] = value_type(df[e])
        rsv.append(tmp)

    info = pd.DataFrame(rsv)
    return info




if __name__ == '__main__':
    pass
    #preprocess()
    #get_x_y_for_cv()



