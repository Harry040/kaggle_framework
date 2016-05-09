import pandas as pd

def ensemble():



    ms = ['../lr/lr_test_l1.csv', '../rf/rf_test_l1.csv', '../svm/svm_test_l1.csv', '../xgb/xgb_test_l1.csv']
    ms = ['../lr/lr_test_l1.csv', '../xgb/xgb_test_l1.csv']


    test = []
    for m in ms:
        df = pd.read_csv(m)
        ID = df['ID']
        df = df.drop('ID', axis=1)
        col = df.columns
        test.append(df.values)
    
    df = sum(test)
    df = df/2.0
    df = pd.DataFrame(df, index=ID, columns=col)

    df.to_csv('submission.csv', index_label='ID')

    

def get_level_1():

    ms = ['../lr/lr_test_l1.csv', '../rf/rf_test_l1.csv', '../svm/svm_test_l1.csv', '../xgb/xgb_test_l1.csv']
    test = []
    for m in ms:
        df = pd.read_csv(m)
        
        df = df.drop('id', axis=1)
        test.append(df.values)
    return test
    test = pd.concat(test, axis=1)
    return test


    test.to_csv('../../data/test_level_1.csv', index=False)





    ms = ['../lr/lr_train_l1.csv', '../rf/rf_train_l1.csv', '../svm/svm_train_l1.csv', '../xgb/xgb_train_l1.csv']
    train = []
    for m in ms:
        df = pd.read_csv(m)
        train.append(df)
    train = pd.concat(train, axis=1)
    train.to_csv('../../data/train_level1.csv', index=False)

    



ensembal()
#df = get_level_1()
