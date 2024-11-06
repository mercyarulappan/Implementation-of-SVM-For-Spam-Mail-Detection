# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program. 

## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MERCY A
RegisterNumber: 212223110027
*/
import  pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.tail()
data.info()
```
![Screenshot 2024-11-06 112920](https://github.com/user-attachments/assets/ef7d825a-79b0-4b13-871a-011053f5ad6b)
![Screenshot 2024-11-06 112926](https://github.com/user-attachments/assets/74e59eea-a73e-46a9-9230-4a12c3ac8417)
![Screenshot 2024-11-06 112932](https://github.com/user-attachments/assets/6e872c5e-558c-4177-807e-edd9adc5edd7)

```python
data.isnull().sum()
```
![Screenshot 2024-11-06 112938](https://github.com/user-attachments/assets/be0c8dd2-bd63-445c-a75a-652622691eac)

```python
x = data['v2'].values
y = data['v1'].values
print(x.shape)
print(y.shape)
```
![Screenshot 2024-11-06 112944](https://github.com/user-attachments/assets/ea69bed4-2e2a-4692-b867-49e7eaaf2f12)

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
print(x_train.shape)
print(x_test.shape)
```
![Screenshot 2024-11-06 112951](https://github.com/user-attachments/assets/44c5768c-abfd-4dd2-abd6-c18cdfe719ec)

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
print(x_train.shape)
print(x_test.shape)
```
![Screenshot 2024-11-06 112956](https://github.com/user-attachments/assets/3b06a991-cd0a-42c6-8b40-15fca16c8a69)
![Screenshot 2024-11-06 113000](https://github.com/user-attachments/assets/46fca495-a9cf-4180-869d-6575466bfcff)

```python
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy
```
![Screenshot 2024-11-06 113005](https://github.com/user-attachments/assets/6fbc0662-1276-4ef0-b014-2163bd8c225e)

## Output:

![Screenshot 2024-11-06 113009](https://github.com/user-attachments/assets/87936fc8-ab22-45d6-bdf7-92938e90ebe4)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
