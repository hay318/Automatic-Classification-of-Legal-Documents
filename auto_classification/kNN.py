import legal_case_sets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighbor(obj):
    
    def __init__(self, weights, train, label, k=5, metric='euclidean', dWeight='uniform', algorithm='auto', metricKW={}):
        self.knc = KNeighborsClassifier(n_neighbors=k,metric=metric, weights=dWeight, algorithm='auto',**metricKW)
        self.weights = np.array(weights)
        self.train = np.array(train)
        self.label = np.array(label)
        self.labels = sorted(set(self.label))
        self.knc.fit(self.train * weights, self.label)

def calculate_accuracy(self, test_tracks, labels):
    test = np.array(test_tracks) * self.weights
    return self.knc.score(test, labels)
    
    def calculate_accuracy_predict(self, data, labels):
        data = np.array(data) * self.weights
        p = self.knc.predict(data)
        t = np.array(labels)
        accuracy = sum(p == t) / float(len(p))
        return accuracy
