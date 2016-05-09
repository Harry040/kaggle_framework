
def save_result(path, df):
    with open(path, 'w') as f:
        f.write('"ID","TARGET"\n')
    df.to_csv(path,mode='a',index=False,header=False)
    print path, 'success!'
   



