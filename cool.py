import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Hinge
from sklearn.linear_model import Huber
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import OneSidedSelection
from sklearn.decomposition import PCA
from boruta import boruta_py
from boruta import BorutaPy
from imblearn.under_sampling import RandomUnderSampler
from tpot import TPOTClassifier
data=pd.read_csv('train_data.csv')
print data.head(2)
print data.describe
print(data.isnull().sum())
y=data['target']
print data['target'].value_counts(normalize=True)
print "d"
del data['target']
del data['connection_id']
from sklearn.model_selection import train_test_split
testdata=pd.read_csv('test_data.csv')

d=testdata['connection_id']
del testdata['connection_id']
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(data)

# Apply the random under-sampling
rus = OneSidedSelection(return_indices=True)
X_resampled, y_resampled, idx_resampled = rus.fit_sample(data, y)
# fit model on all training data

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features - 5 features should be selected
feat_selector.fit(X_resampled, y_resampled)

print feat_selector.support_

# check ranking of features
print feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X_resampled)
X_test_filtered=feat_selector.transform(testdata.values)

# my_tpot = TPOTClassifier(generations=10)
# my_tpot.fit(X_filtered, y_resampled)
# pred=my_tpot.predict(X_test_filtered)
model = RandomForestClassifier(n_estimators=501, min_samples_leaf=5, min_samples_split=3)
model.fit(X_filtered, y_resampled)
pred=model.predict(X_test_filtered)

sub = pd.read_csv('sample_submission.csv')
sub['target'] = pred
sub['target'] = sub['target'].astype(int)
sub.to_csv('sub15.csv', index=False)

