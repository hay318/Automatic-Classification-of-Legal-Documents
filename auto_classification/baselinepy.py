import legal_case_sets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df_a = np.zeros([0,109])
for file_name in glob.glob('./txt'):
    curr_df = np.genfromtxt(file_name, delimiter = ",")
    curr_df = curr_df[:,:109]
    df_a = np.vstack((df_a,curr_df))

df_a = np.delete(df_a, (0), axis=0)
baseline_feat = df_a.sum(1)[...,None]
df_b = np.hstack((df_a,baseline_feat))

x = np.delete(df2, 109, axis=1)
y = df2[:,109]
train_y, test_y, train_x, test_x = train_test_split(y, x, random = 0, size=0.9)

#BaseLine - Logistic Regression
baseline_log = LogisticRegression(random = 0)
baseline_log.fit(feat1_train, train_y)
baseline_prd = np.where(feat1_test > feat1_train.mean(), 1, 0)
baseline_m = confusion_matrix(baseline_prd, test_y)










