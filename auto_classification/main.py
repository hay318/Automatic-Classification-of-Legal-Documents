import os
import sys
import transform
import baseline
import kNN
import naiveBayes
import svm
import feature_extractor

def main(args):
    weights = None
    metric = args.knnMetric
    metricKW = {}
    knnAlgorithm = "auto"
    
    # Feature Extracting
    feature_extractor = feature_extractor.extract_all()

    # Decision and Classification
    if args.activate_kNN:test_knn()
    if args.activate_naiveBase:test_naiveBase()
    #if args.activate_svm:test_svm()
    if args.activate_svm: test_baseline()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Input: a set of combine text files")
    args = parser.parse_args()

    if args.NORUN:
        args.activate_kNN = True
        #args.activate_naiveBase = False
        #args.activate_svm = False
        #args.activate_baselube = False

try:
    main(args)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()
