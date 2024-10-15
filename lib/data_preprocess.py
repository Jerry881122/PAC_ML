import pandas as pd
import numpy as np
import torch





class data_preprocess():
    def __init__(self):
        self.feature_nor_obj = normalization()
    
    def train_process_LSTM(self,df,normalized=False):
        feature = torch.tensor(df.iloc[:,0:-1].to_numpy(dtype=np.float32))
        if normalized == True:                                              # normalization
            feature = self.feature_nor_obj.training_normalization(feature)           
        feature = feature.view(len(feature) , len(feature[0]) , 1)          # change shape to (10000,8,1)
        
        return feature

    def test_process_LSTM(self,df_test,normalized=False):
        feature_test = torch.tensor(df_test.iloc[:,0:-1].to_numpy(dtype=np.float32))
        if normalized == True:                                                          # normalization
            feature_test = self.feature_nor_obj.testing_normalization(feature_test)
        feature_test = feature_test.view(len(feature_test) , len(feature_test[0]) , 1)

        return feature_test

class normalization():
    def __init__(self):
        pass

    def training_normalization(self,feature):
        feature_min, _ = feature.min(dim=0)
        feature_max, _ = feature.max(dim=0)
        self.feature_min = feature_min
        self.feature_max = feature_max

        feature_normalized = (feature - self.feature_min) / (self.feature_max - self.feature_min)
        return feature_normalized

    def testing_normalization(self,feature):
        feature_normalized = (feature - self.feature_min) / (self.feature_max - self.feature_min)

        return feature_normalized
