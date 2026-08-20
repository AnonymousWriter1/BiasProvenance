from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import owlready2 as or2

onto = or2.get_ontology("/path/to/ontology/mlprov.owx").load()
with onto:
    class description(or2.DataProperty):
        range = [str]

req_spec = onto["requirement_specification"]
train_data = onto["training_dataset"]
test_data = onto["testing_dataset"]
data_source = onto["data_source"]
method_collection = onto["Method_of_collection"]
data_collector = onto["data_collector"]
preprocessing_step = onto["preprocessing_step"]
exclusion_criteria = onto["exclusion_criteria"]
class_proportion = onto["dataset_class_split"]
onto_model = onto["model"]
performance_metric = onto["performance_metric"]

req1 = req_spec("req1")
req1.description = ["The model shall predict the class <=50K."]
req2 = req_spec("req2")
req2.description = ["The model shall predict the class >50K."]

training_dataset = train_data("adult_train.csv")
training_dataset.description = ["github"]
original_test_dataset = test_data("adult_test.csv")
original_test_dataset.description = ["github"]
sex_test_dataset = test_data("sex_test.csv")
sex_test_dataset.description = ["github"]
race_test_dataset = test_data("race_test.csv")
race_test_dataset.description = ["github"]

source = data_source("1994 Census")
method_of_collection = method_collection("Census survey")
method_of_collection.description = ["Census survey conducted by the US Census Bureau. Year: 1994"]
collector = data_collector("Government Official")

drop_na = preprocessing_step("Drop NA")
excluded = exclusion_criteria("NA")
male = class_proportion("0.67")
female = class_proportion("0.33")
white = class_proportion("0.85")
black = class_proportion("0.09")
api = class_proportion("0.03")
ai = class_proportion("0.009")
other = class_proportion("0.008")

csv_file_path = '/path/to/data/adult_train.csv'
df = pd.read_csv(csv_file_path)
df = df.dropna()
df.head()

X = df.iloc[:, :-1]   # all rows, all columns except the last
y = df.iloc[:, -1]    # all rows, just the last column

x_encoded = pd.get_dummies(X)
y_encoded = pd.get_dummies(y)
y_encoded = y.map({' <=50K': 0, ' >50K': 1})

print(y_encoded)

clf = LogisticRegression(max_iter=10000, random_state=0).fit(x_encoded, y_encoded)
model = onto_model("clf")

"""Validation metrics"""

acc = accuracy_score(y_encoded, clf.predict(x_encoded)) * 100
print(acc)
acc_val = performance_metric("val accuracy")
acc_val.description = [str(acc)]
precision = precision_score(y_encoded, clf.predict(x_encoded))
print(precision)
precision_val = performance_metric("val precision")
precision_val.description = [str(precision)]
recall = recall_score(y_encoded, clf.predict(x_encoded))
print(recall)
recall_val = performance_metric("val recall")
recall_val.description = [str(recall)]
f1 = f1_score(y_encoded, clf.predict(x_encoded))
print(f1)
f1_val = performance_metric("val f1")
f1_val.description = [str(f1)]
auc = roc_auc_score(y_encoded, clf.predict(x_encoded))
print(auc)
auc_val = performance_metric("val auc")
auc_val.description = [str(auc)]

"""Test on original testing dataset"""

df_test = pd.read_csv('/path/to/data/adult_test.csv')
df_test = df_test.dropna()
df_test.head()

x_test_unfiltered = df_test.iloc[:, :-1]
y_test_unfiltered = df_test.iloc[:, -1]
x_test_unfiltered_encoded = pd.get_dummies(x_test_unfiltered)
# Align the columns of the test set with the training set
x_test_unfiltered_encoded = x_test_unfiltered_encoded.reindex(columns=x_encoded.columns, fill_value=0)
y_test_unfiltered_encoded = pd.get_dummies(y_test_unfiltered)
y_test_unfiltered_encoded = y_test_unfiltered.map({' <=50K.': 0, ' >50K.': 1})

y_pred_unfiltered = clf.predict(x_test_unfiltered_encoded)
acc_test_unfiltered = accuracy_score(y_test_unfiltered_encoded, y_pred_unfiltered) * 100
print(acc_test_unfiltered)
acc_test_unfiltered_val = performance_metric("test accuracy")
acc_test_unfiltered_val.description = [str(acc_test_unfiltered)]
precision = precision_score(y_test_unfiltered_encoded, y_pred_unfiltered)
print(precision)
precision_val = performance_metric("test precision")
precision_val.description = [str(precision)]
recall = recall_score(y_test_unfiltered_encoded, y_pred_unfiltered)
print(recall)
recall_val = performance_metric("test recall")
recall_val.description = [str(recall)]
f1 = f1_score(y_test_unfiltered_encoded, y_pred_unfiltered)
print(f1)
f1_val = performance_metric("test f1")
f1_val.description = [str(f1)]
auc = roc_auc_score(y_test_unfiltered_encoded, y_pred_unfiltered)
print(auc)
auc_val = performance_metric("test auc")
auc_val.description = [str(auc)]

"""Minority sex test set"""

df_test_filtered = pd.read_csv('/path/to/data/sex_test.csv')
df_test_filtered = df_test_filtered.dropna()
df_test_filtered.head()

x_test_filtered = df_test_filtered.iloc[:, :-1]
y_test_filtered = df_test_filtered.iloc[:, -1]
x_test_filtered_encoded = pd.get_dummies(x_test_filtered)
# Align the columns of the test set with the training set
x_test_filtered_encoded = x_test_filtered_encoded.reindex(columns=x_encoded.columns, fill_value=0)
y_test_filtered_encoded = pd.get_dummies(y_test_filtered)
y_test_filtered_encoded = y_test_filtered.map({' <=50K.': 0, ' >50K.': 1})

y_pred_filtered = clf.predict(x_test_filtered_encoded)
acc_test_filtered = accuracy_score(y_test_filtered_encoded, y_pred_filtered) * 100
print(acc_test_filtered)
acc_test_filtered_val = performance_metric("minority sex accuracy")
acc_test_filtered_val.description = [str(acc_test_filtered)]
precision = precision_score(y_test_filtered_encoded, y_pred_filtered)
print(precision)
precision_val = performance_metric("minority sex precision")
precision_val.description = [str(precision)]
recall = recall_score(y_test_filtered_encoded, y_pred_filtered)
print(recall)
recall_val = performance_metric("minority sex recall")
recall_val.description = [str(recall)]
f1 = f1_score(y_test_filtered_encoded, y_pred_filtered)
print(f1)
f1_val = performance_metric("minority sex f1")
f1_val.description = [str(f1)]
auc = roc_auc_score(y_test_filtered_encoded, y_pred_filtered)
print(auc)
auc_val = performance_metric("minority sex auc")
auc_val.description = [str(auc)]

"""Minority race test"""

df_test_race = pd.read_csv('/path/to/data/race_test.csv')
df_test_race = df_test_race.dropna()
df_test_race.head()

x_test_race = df_test_race.iloc[:, :-1]
y_test_race = df_test_race.iloc[:, -1]
x_test_race_encoded = pd.get_dummies(x_test_race)
# Align the columns of the test set with the training set
x_test_race_encoded = x_test_race_encoded.reindex(columns=x_encoded.columns, fill_value=0)
y_test_race_encoded = pd.get_dummies(y_test_race)
y_test_race_encoded = y_test_race.map({' <=50K.': 0, ' >50K.': 1})

y_pred_race = clf.predict(x_test_race_encoded)
acc_test_race = accuracy_score(y_test_race_encoded, y_pred_race) * 100
print(acc_test_race)
acc_test_race_val = performance_metric("minority race accuracy")
acc_test_race_val.description = [str(acc_test_race)]
precision = precision_score(y_test_race_encoded, y_pred_race)
print(precision)
precision_val = performance_metric("minority race precision")
precision_val.description = [str(precision)]
recall = recall_score(y_test_race_encoded, y_pred_race)
print(recall)
recall_val = performance_metric("minority race recall")
recall_val.description = [str(recall)]
f1 = f1_score(y_test_race_encoded, y_pred_race)
print(f1)
f1_val = performance_metric("minority race f1")
f1_val.description = [str(f1)]
auc = roc_auc_score(y_test_race_encoded, y_pred_race)
print(auc)
auc_val = performance_metric("minority race auc")
auc_val.description = [str(auc)]

"""Save provenance graph"""

onto.save(file="/path/to/provenance/graph/mlprov_selection_bias_instances.owl")