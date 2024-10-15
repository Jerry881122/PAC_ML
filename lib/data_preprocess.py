import torch

class normalization():
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
