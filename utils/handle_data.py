import os
import cPickle
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler

Config = {
'submit_data_test': '/data/c_submit_data_test',
'submit_data': '/data/submit_data',
}


def get_X_y_test(reload=False):
    '''
    fp = os.path.abspath(__file__)
    dfp = os.path.dirname(os.path.dirname(fp))
    fp = dfp + Config['submit_data_test']
    if not reload and os.path.exists(fp):
        print 'ok, I am aready'
        return cPickle.load(open(fp))
    '''

    fp = os.path.abspath(__file__)
    dfp = os.path.dirname(os.path.dirname(fp))

    train = pd.read_csv(dfp+'/train.csv')

    train_x = train.drop(['ID', 'TARGET'], axis=1)
    train_y = train['TARGET']

    #X_train, X_test, y_train, y_test = train_test_split(  train_x, train_y, test_size=0.33, random_state=42)

    kf = StratifiedKFold(train_y, 9, random_state=555)

    for tr, te in kf:
        break


    kf = StratifiedKFold(train_y)
    train_x = train_x.iloc[te]
    train_y = train_y.iloc[te]



    test = pd.read_csv(dfp + '/test.csv')
    test_x = test.drop('ID', axis=1)
    test_id = test['ID']
    #lng = int(len(test_x) * 0.2) 
    #test_x = test_x[:lng]
    #test_id = test_id[:lng]

    tmp = { }
    tmp['train_x'] = train_x
    tmp['test_x'] = test_x
    tmp['train_y'] = train_y
    tmp['test_id'] = test_id

    return tmp



def get_X_y():
    fp = os.path.abspath(__file__)
    dfp = os.path.dirname(os.path.dirname(fp))

    train = pd.read_csv(dfp + '/train.csv')

    train_x = train.drop(['ID', 'TARGET'], axis=1)
    train_y = train['TARGET']


    test = pd.read_csv(dfp +'/test.csv')
    test_x = test.drop('ID', axis=1)
    test_id = test['ID']

    # standerScaler
    st = StandardScaler()
    train_x = st.fit_transform(train_x)
    train_x = pd.DataFrame(train_x)

    test_x = st.transform(test_x)
    test_x = pd.DataFrame(test_x)
    tmp = { }
    tmp['train_x'] = train_x
    tmp['test_x'] = test_x
    tmp['train_y'] = train_y
    tmp['test_id'] = test_id

    return tmp




def get_col():
    '''
    fp = os.path.abspath(__file__)
    dfp = os.path.dirname(os.path.dirname(fp))
    '''

    return ['TARGET']









if __name__ == '__main__':
    pass

    get_X_y_test(True)

