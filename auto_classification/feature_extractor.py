import os
import numpy as np
import util
import scipy.features as features

def extract_all():
    file_dir = "./legal_cases_combined_whole.txt"
    feature_g()

def feature_g():
    ind = []
    victim_weight = 0
    violence_weight = 0
    weapons_feat = ["firearm","gun","sword","sharp", "bat", "knife", "weapon"]
    intent_feat = ["basic", "specific", "direct", "oblique"]
    mult_victim_feat = ["students", "pedestrians", "family", "children"]
    
    for features in signals:
        ind.append(signal_labels.index(features))
    crime_types = [org_breif[i,:] for i in ind]

filtered = []
segment = len(org_breif[0]) / 109
feat_size = len(crime_types) * num_filters
feat_fin = np.zeros((segment,feat_size))

    for judgement in crime_types:
        filtered.extend([util.crimeFilters(judgement,low,med,fs)])
        filtered.extend([util.crimeFilters(judgement,med,high,fs)])
        for seg in xrange(0,segment):
            for i,categorized_cases in enumerate(filtered):
                feat_fin[seg,i] = calculateRMS(categorized_cases[seg*109:seg*109 + 1]) #for all total cases
return feat_fin
