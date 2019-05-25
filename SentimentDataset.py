import pandas as pd
import numpy as np
import random

class SentimentDataset():
    def __init__(self,csv_path):
        dataframe=pd.read_csv(csv_path)

        X=dataframe.to_numpy()
        Y=X[:,1]
        X=np.delete(X,1,1)
        new_X=[]
        k=0
        for row in X:
            try:
                nums=str(row[0]).split()
                new_row=[int(x) for x in nums]
                new_X.append(new_row)
                k+=1
            except:
                Y=np.delete(Y,k,0)

        new_X=np.array(new_X)
        seed=random.randint(0,100)
        np.random.seed(seed)
        np.random.shuffle(new_X)
        self.X =new_X
        np.random.seed(seed)
        np.random.shuffle(Y)
        self.Y =Y
    def get_dataset(self,validation_percentage=0.2):
        index=int(self.X.shape[0]*(1-validation_percentage))
        return self.X[:index],self.Y[:index],self.X[index:],self.Y[index:]


if __name__=='__main__':
    s=SentimentDataset('./preprocessed_dataset/kita.csv')