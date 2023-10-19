#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

import warnings
warnings.filterwarnings("ignore")


# In[4]:


# Utility Functions
def checking_overfitting_undefitting(y_train, y_train_pred, y_test, y_test_pred):
    """
    Print whether the model is underfit, overfit or good fit.
    
    y_train = training data
    y_train_pred = predictions on training data
    y_test = testing data
    y_test_pred = predictions on testing data
    """
    training_accuracy = accuracy_score(y_train, y_train_pred)
    testing_accuracy = accuracy_score(y_test, y_test_pred)
    if training_accuracy<=0.65:
        print("Model is underfitting.") 
    elif training_accuracy>0.65 and abs(training_accuracy-testing_accuracy)>0.15:
        print("Model is overfitting.")
    else:
        print("Model is not underfitting/overfitting.")

def calculate_classification_metrics(y_true, y_pred, algorithm):
    """
    Return the classification Metrics
    
    y_true = actual values
    y_pred = predicted values
    y_pred_probability = probability values
    algorithm = algorithm name
    """
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred, average='weighted'), 3)
    recall = round(recall_score(y_true, y_pred, average='weighted'), 3)
    f1 = round(f1_score(y_true, y_pred, average='weighted'), 3)
    print("Algorithm: ", algorithm)
    print()
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print()
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Overcast', 'Clear','Foggy']
    plt.figure(figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return accuracy, precision, recall, f1


# In[6]:


# Fetching data
data = pd.read_csv("C:\\Users\\Administrator\\Downloads\\archive\\IRIS.csv")
data.head()


# In[7]:


# Checking datatypes/shape of data
data.info()


# In[8]:


# Checking Statistical Summary of data
data.describe(include='all')


# In[9]:


# Checking null values
data.isnull().sum()


# In[10]:


# Checking duplicated values
data.duplicated().sum()


# In[11]:


# Removing duplicated values
data.drop_duplicates(inplace=True)
# Again checking duplicated values
data.duplicated().sum()


# In[12]:


# Checking for outliers
sns.boxplot(data=data)
plt.title("Boxplot")
plt.show()


# In[13]:


# Pairplot to visualize relationships between variables
sns.pairplot(data, hue="species", markers=["o", "s", "D"])


# In[14]:


# Violin plot to visualize distribution and density by species
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.violinplot(x="species", y="petal_length", data=data)
plt.subplot(2, 2, 2)
sns.violinplot(x="species", y="petal_width", data=data)
plt.subplot(2, 2, 3)
sns.violinplot(x="species", y="sepal_width", data=data)
plt.subplot(2, 2, 4)
sns.violinplot(x="species", y="sepal_length", data=data)
plt.tight_layout()


# In[15]:


# Correlation heatmap
df = data.copy()
df.drop("species", axis=1, inplace=True)
correlation = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[16]:


# Input features dataset
input_df = data.drop(columns="species", axis=1).values


# In[17]:


# Target variable 
# Applying mapping
encoder = LabelEncoder()
y = data["species"]
y = encoder.fit_transform(y)

# Checking the mapping of the classes
class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
for class_label, class_number in class_mapping.items():
    print(f"Class '{class_label}' is labeled as {class_number}")


# In[18]:


# Splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(input_df, y, test_size=0.20, random_state=42)


# In[19]:


# Apply scaling on the input_df DataFrame
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
joblib.dump(scaler, "scaler.pkl")


# In[20]:


# Hyperparameter tuning
parameters = {'solver': ['liblinear', 'saga'], 
              'multi_class':['ovr', 'multinomial'],
              'C':[0.001, 0.01, 10.0],
              'penalty': ['l1', 'l2']}
# Model Creation and Training
model_lr = LogisticRegression(n_jobs=-1)
models_lr = GridSearchCV(estimator=model_lr, param_grid=parameters, cv=4)
models_lr.fit(x_train, y_train)
best_parameters = models_lr.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions for train
best_model_lr = models_lr.best_estimator_
y_pred_lr = best_model_lr.predict(x_train)
# Predictions for test
y_pred_lr_new = best_model_lr.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_lr, y_test, y_pred_lr_new)


# In[21]:


# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_lr, precision_lr, recall_lr, f1_lr = calculate_classification_metrics(y_test, y_pred_lr_new, "Logistic Regression")


# In[22]:


# Hyperparameter tuning
parameters = {'criterion':['gini', 'entropy', 'log_loss'], 
              'max_depth': [None, 5, 10],
              'min_samples_split': [None, 2, 5],
              'splitter':['best','random']}
# Model Creation and Training
model_dt = DecisionTreeClassifier()
models_dt = GridSearchCV(estimator=model_dt, param_grid=parameters, cv=4)
models_dt.fit(x_train, y_train)
best_parameters = models_dt.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on train data
best_model_dt = models_dt.best_estimator_
y_pred_dt = best_model_dt.predict(x_train)
# Predictions on test data
y_pred_dt_new = best_model_dt.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_dt, y_test, y_pred_dt_new)


# In[23]:


# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_dt, precision_dt, recall_dt, f1_dt = calculate_classification_metrics(y_test, y_pred_dt_new, "Decision Tree")


# In[24]:


# Hyperparameter tuning
parameters = {'max_depth': [None, 5],
            'class_weight': [None, 'balanced'],
            'min_samples_split': [None, 2, 5],
            'criterion':['gini','log_loss','entropy']}
# Model Creation and Training
model_et = ExtraTreesClassifier()
models_et = GridSearchCV(estimator=model_et, param_grid=parameters, cv=4)
models_et.fit(x_train, y_train)
best_parameters = models_et.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on train data
best_model_et = models_et.best_estimator_
y_pred_et = best_model_et.predict(x_train)
# Predictions on test data
y_pred_et_new = best_model_et.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_et, y_test, y_pred_et_new)


# In[25]:


# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_et, precision_et, recall_et, f1_et = calculate_classification_metrics(y_test, y_pred_et_new, "Extra Trees")


# In[29]:


# Hyperparameter tuning
parameters = {'var_smoothing':[1e-9, 1e-8, 1e-10]}
# Model Creation and Training
model_nb = GaussianNB()
models_nb = GridSearchCV(estimator=model_nb, param_grid=parameters, cv=4)
models_nb.fit(x_train, y_train)
best_parameters = models_nb.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on training data
best_model_nb = models_nb.best_estimator_
y_pred_nb = best_model_nb.predict(x_train)
# Predictions on test data
y_pred_nb_new = best_model_nb.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_nb, y_test, y_pred_nb_new)


# In[30]:


# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_nb, precision_nb, recall_nb, f1_nb = calculate_classification_metrics(y_test, y_pred_nb_new, "Gaussian NB")


# In[31]:


# Results
print("Testing Performances for Machine Learning Algorithms")
result = pd.DataFrame({"Algorithms":['Logistic Regression', "Decision Tree", "Extra Trees Classifier", "Gaussian Naive Bayes"],
                       "Accuracy":[accuracy_lr,  accuracy_dt, accuracy_et, accuracy_nb],
                       "Precision":[precision_lr,  precision_dt,  precision_et, precision_nb],
                       "Recall":[recall_lr,  recall_dt,  recall_et, recall_nb],
                       "F1 Score":[f1_lr,  f1_dt,  f1_et, f1_nb]}).set_index('Algorithms')
result


# In[32]:


# Saving sklearn machine learning models
models = [best_model_dt, best_model_lr, best_model_et, best_model_nb]
names = ["dt","lr","et","nb"]
for i in range(len(models)):
    joblib.dump(models[i],names[i]+".pkl")


# In[ ]:




