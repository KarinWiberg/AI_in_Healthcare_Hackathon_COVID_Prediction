#0.954 recall


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount
from sklearn.impute import SimpleImputer
from tpot.export_utils import set_param_recursive
from sklearn.metrics import accuracy_score, recall_score, fbeta_score

tpot_data = pd.read_csv('cleaned_data_v1-combined.csv', sep=',', dtype=np.float64)
features = tpot_data.iloc[:, : -3]


features = features.drop('UCI_DAYS', axis=1)
features = features.drop('LIN--SISTEMATICODESANGRE', axis=1)


training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['DEATH'], random_state=42)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

exported_pipeline = make_pipeline(
    ZeroCount(),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.4, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
)
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(recall_score(testing_target, results, average='weighted'))

print(accuracy_score(testing_target, results))

print(fbeta_score(testing_target, results, beta=0.9, average='macro'))

