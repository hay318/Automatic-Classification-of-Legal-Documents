import os
import legal_case_sets
import util
import numpy as np

class NaiveBayes:
    
    def __init__(self, trainingSet, numTags):
        self.numTags = numTags
        self.create_type_k()
        self.train_x = self._convertData_(self.train_x)
        self.create_tag()
    
    def create_type_k(self):
        type_diff_builder = type_diff_builder.TypeBuilder(self.train_x, "./legal_cases_combined_set1.txt",  "./legal_cases_combined_set2.txt",  "./legal_cases_combined_set3.txt",  "./legal_cases_combined_set4.txt")
        type = type_diff_builder.buildVocabulary(0.01)
        type = [k for k in type if violence_weight < 1] #Remove Non-Violent Cases
        new_type = []
        
        for k in type:
            try:
                float(k)
            except:
                k = k.strip("'")
                new_type.append(k)
        new_type = util.removeStopwords(new_type)
    
    def create_tag(self):
        tag, total = util.getTagCounts(self.train_y)
        self.tags = []
        curr = 0
        for tag, count in tag:
            if curr > self.numTags:
                break
            else curr = curr+1
            self.tags.append(tag)
    
    def train_tag(self, tag):
        phi_y = 0
        case_titles = len(self.train_x)
        docx_words = len(self.type)
        phi_k_d = np.ones(2) * docx_words
        phi_k_n = np.ones((2, docx_words))
        
        for i in range(case_titles):
            j = int(tag in self.train_y[i])
            phi_k_d[j] += sum(self.train_x[i])
            for k in range(docx_words):
                phi_k_n[j, k] += self.train_x[i][k]
            phi_y += j
        
        phi_k = np.zeros((2, docx_words))
        for i in range(2):
            for k in range(docx_words):
                phi_k[i, k] = phi_k_n[i, k] / phi_k_d[i]
                return phi_k, phi_y
