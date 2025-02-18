# Vertebral Column Dataset Analysis

## 📌 Overview
This project analyzes the Vertebral Column dataset using Python, exploring biomechanical features and applying machine learning models for classification.

## a) Downloading the dataset

### 🔧 Code Example
```python
!pip3 install -U ucimlrepo 
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import seaborn as sns

```

### 🔧 Code Example
```python
vertebral_column = fetch_ucirepo(id=212) 
```

### 🔧 Code Example
```python
print(vertebral_column.variables) 

```

### 🔧 Code Example
```python
print(vertebral_column)
```

### 🔧 Code Example
```python
type(vertebral_column)

```

## pre processing and exploratory data analysis

Let's first make scatterplots of the independent variables in the dataset by using color to show Classes 0 and 1

### 🔧 Code Example
```python
data = vertebral_column.data.features 
target = vertebral_column.data.targets

```

### 🔧 Code Example
```python
target = target.replace({"Normal": 0, "Hernia": 1, "Spondylolisthesis": 1})
```

### 🔧 Code Example
```python
print(target)
```

### 🔧 Code Example
```python
data["Class"] = target
```

### 🔧 Code Example
```python
print(data["Class"].unique())
```

### 🔧 Code Example
```python
sns.pairplot(data, hue="Class", diag_kind="hist", palette="coolwarm");

```

### 🔧 Code Example
```python
print(data)

```

### 🔧 Code Example
```python
print(target)
```

### Now let's make boxplots for each of the independent variables. Use color to show Classes 0 

### 🔧 Code Example
```python
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.boxplot(data = data.loc[:, 'pelvic_incidence':'degree_spondylolisthesis'], orient = 'h', width=0.6)
plt.title("Horizontal Boxplot for independent variables", fontsize=16, weight='bold')
plt.xlabel("Values", fontsize=14)
plt.ylabel("biomechanical attributes", fontsize=14)
```

### As we need to split train/test data let's select the first 70 rows of Class 0 and the  first 140 rows of Class 1 as the training set and the rest of the data as the test set.

### 🔧 Code Example
```python
normal = data[data['Class'] == 0]
abnormal= data[data['Class'] == 1]
```

### 🔧 Code Example
```python
train_class_normal = normal.head(70)
train_class_abnormal = abnormal.head(140)
```

### 🔧 Code Example
```python
train_set = pd.concat([train_class_normal, train_class_abnormal])

```

### 🔧 Code Example
```python
test_class_normal = normal.iloc[70:]
test_class_abnormal = abnormal.iloc[140:]


test_set = pd.concat([test_class_normal, test_class_abnormal])
```

### 🔧 Code Example
```python
print(train_set)
```

### 🔧 Code Example
```python
print(test_set)
```

## Applying K Nearest Neighbors to classify

### Let's use Euclidean distance first

### 🔧 Code Example
```python
X_train = train_set.loc[:, 'pelvic_incidence':'degree_spondylolisthesis'].values
y_train = train_set['Class']

X_test = test_set.loc[:, 'pelvic_incidence':'degree_spondylolisthesis'].values
y_test = test_set['Class']

k_values = list(range(208,3,-3))

train_errors = []
test_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k, metric = "euclidean")
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    train_errors.append(train_error)
    test_errors.append(test_error)





    
```

### Plotting train and test errors in terms of k for k in 208; 205; : : : ; 7; 4; 1; g (in reverse order). Let's also calculate the confusion matrix, true positive rate, true negative rate, precision, and F1-score when k = k

### 🔧 Code Example
```python
plt.figure(figsize = (10,5))
plt.plot(k_values, train_errors, label = 'Train Error')
plt.plot(k_values, test_errors, label = 'Test Error')

plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error Rate')
plt.title('Train and Test Errors vs. k')
plt.legend()
plt.grid()
plt.show()

```

### 🔧 Code Example
```python
print(min(train_errors))

```

### Lets select k = 25, which is the point where the test error decreases before increasing drastically

### 🔧 Code Example
```python
knn = KNeighborsClassifier(n_neighbors = 25, metric = "euclidean")


knn.fit(X_train, y_train)
y_test_pred = knn.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()



tpr = tp / (tp + fn) 
tnr = tn / (tn + fp) 
precision = precision_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)


print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
print(f"True Positive Rate (TPR/Recall): {tpr:.4f}")
print(f"True Negative Rate (TNR): {tnr:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")
```

### finding the best test error rate when the size of training set is 10; 20; 30 ... 210. For each N, training set will be the  first N/3 rows of Class 0 and the  first N - N/3 rows of Class 1. for each N, we'll select the optimal k from a set starting from k = 1, increasing by 5. For example, if N = 200, the optimal k is selected from f1; 6; 11; : : : ; 196g.

### 🔧 Code Example
```python
train_class_normal = normal.head(70)
train_class_abnormal = abnormal.head(140)

train_set = pd.concat([train_class_normal, train_class_abnormal])

test_class_normal = normal.iloc[70:]
test_class_abnormal = abnormal.iloc[140:]


test_set = pd.concat([test_class_normal, test_class_abnormal])


X_train = train_set.loc[:, 'pelvic_incidence':'degree_spondylolisthesis'].values
y_train = train_set['Class']

X_test = test_set.loc[:, 'pelvic_incidence':'degree_spondylolisthesis'].values
y_test = test_set['Class']

k_values = list(range(208,3,-3))

train_errors = []
test_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k, metric = "euclidean")
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    train_errors.append(train_error)
    test_errors.append(test_error)

```

### 🔧 Code Example
```python

N_values = list(range(10,211,10))

best_test_errors = []
best_train_errors = []

for N in N_values:
    train_class_normal = normal.head(N//3)
    train_class_abnormal = abnormal.head(N-(N//3))
    train_set = pd.concat([train_class_normal, train_class_abnormal])

    X_train = train_set.loc[:, 'pelvic_incidence':'degree_spondylolisthesis'].values
    y_train = train_set['Class']

    
    X_test = test_set.loc[:, 'pelvic_incidence':'degree_spondylolisthesis'].values
    y_test = test_set['Class']

    
    k_values = list(range(1, min(N, 200), 5))
    
    test_errors = []
    train_errors = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_train, y_train)
        y_test_pred = knn.predict(X_test)
        y_train_pred = knn.predict(X_train)

        test_error = 1 - accuracy_score(y_test, y_test_pred)
        train_error = 1 - accuracy_score(y_train,y_train_pred)
        
        test_errors.append(test_error)
        train_errors.append(train_error)
    #print(train_errors)

    
    best_test_errors.append(min(test_errors))
    best_train_errors.append(min(train_errors))
    

```

### 🔧 Code Example
```python
#plotting the learning curve 
plt.figure(figsize=(8, 5))
plt.plot(N_values, best_test_errors, marker='o', linestyle='-', label='Best Test Error Rate')

plt.xlabel('Training Set Size (N)')
plt.ylabel('Best Test Error Rate')
plt.title('Learning Curve')
plt.legend()
plt.grid()
plt.show()
```

### 🔧 Code Example
```python
plt.figure(figsize=(8, 5))
plt.plot(N_values, best_train_errors, marker='o', linestyle='-', label='Best Train Error Rate')

plt.xlabel('Training Set Size (N)')
plt.ylabel('Best Train Error Rate')
plt.title('Best Training Error vs Sample size')
plt.legend()
plt.grid()
plt.show()
```

### It is expected to have a training error of 0 when k = 1 no matter what the N is -> (overfitting)

## Now we have results with Euclidean metric, let's change it and analyze the results

### Minkowski Distance, which becomes Manhattan Distance with p = 1.

### 🔧 Code Example
```python

X_train = train_set.loc[:, 'pelvic_incidence':'degree_spondylolisthesis'].values
y_train = train_set['Class']

X_test = test_set.loc[:, 'pelvic_incidence':'degree_spondylolisthesis'].values
y_test = test_set['Class']

k_values = list(range(1, 197, 5))


test_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=1) 
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)

    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_error)


results_df = pd.DataFrame({"k": k_values, "Test Error": test_errors})


print(results_df)

```

### 🔧 Code Example
```python

log10_p_values = np.arange(0.1, 1.1, 0.1)  
p_values = 10 ** log10_p_values  


test_errors = []


k = 26

for p in p_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=p)
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)

    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_error)


results_df = pd.DataFrame({"log10(p)": log10_p_values, "p": p_values, "Test Error": test_errors})


best_index = results_df["Test Error"].idxmin()
best_log10_p = results_df.loc[best_index, "log10(p)"]
best_p = results_df.loc[best_index, "p"]
best_test_error = results_df.loc[best_index, "Test Error"]



print(results_df)



```

## chebyshev distance 

### 🔧 Code Example
```python
k_values = list(range(1, 197, 5))


test_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="chebyshev") 
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)

    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_error)


results_df = pd.DataFrame({"k": k_values, "Test Error": test_errors})


print(results_df)
```

####  Mahalanobis Distance.

### 🔧 Code Example
```python
cov_matrix = np.cov(X_train, rowvar=False)

inv_cov_matrix = np.linalg.pinv(cov_matrix)


k_values = list(range(1, 197, 5))
test_errors = []


for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="mahalanobis", metric_params={"VI": inv_cov_matrix}) 
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)

    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_error)


results_df = pd.DataFrame({"k": k_values, "Test Error": test_errors})


print(results_df)

```

### The majority polling decision can be replaced by weighted decision, in which the weight of each point in voting is inversely proportional to its distance from the query/test data point. In this case, closer neighbors of a query point will have a greater influence than neighbors which are further away. Let's use weighted voting with Euclidean, Manhattan, and Chebyshev distances and report the best test errors when k 2 f1; 6; 11; 16; : : : ; 196g.

### 🔧 Code Example
```python
k_values = list(range(1, 197, 5))

distance_metrics = ["euclidean", "manhattan", "chebyshev"]

results = {metric: [] for metric in distance_metrics}

for metric in distance_metrics:
    test_errors = []
    
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights="distance")  
        knn.fit(X_train, y_train)
        y_test_pred = knn.predict(X_test)

        test_error = 1 - accuracy_score(y_test, y_test_pred)
        test_errors.append(test_error)

    results[metric] = test_errors  


results_df = pd.DataFrame({"k": k_values, 
                           "Euclidean Test Error": results["euclidean"], 
                           "Manhattan Test Error": results["manhattan"], 
                           "Chebyshev Test Error": results["chebyshev"]})


print(results_df)


best_results = results_df.iloc[:, 1:].min()
print("\nBest Test Errors:")
print(best_results)

```

Lowest training rate is 0 no matter we are using the full sample or any N, because in case of k=1 the training error is 0

