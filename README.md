# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the Required module from sklearn.
## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KISHORE.S
RegisterNumber: 212224230130
*/
import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
HEAD AND INFO:

![Screenshot 2025-04-12 224622](https://github.com/user-attachments/assets/6740fe39-7410-4e5e-bafc-4c26a0e70a4c)
![Screenshot 2025-04-12 224635](https://github.com/user-attachments/assets/b52af287-b46d-47a0-9336-5caf8e709d8d)

NULL VALUES:

![Screenshot 2025-04-12 224710](https://github.com/user-attachments/assets/bc847b39-cc2d-4895-a18e-ceb9b521bca5)

COUNT VALUES:

![Screenshot 2025-04-12 224720](https://github.com/user-attachments/assets/2d099a50-bb79-45bc-a8f9-3cafe89009da)

ENCODED VALUES:

![Screenshot 2025-04-12 224738](https://github.com/user-attachments/assets/f6cbf705-ed9f-48e6-870e-1e4c5148103d)

ACCURACY:

![Screenshot 2025-04-12 224750](https://github.com/user-attachments/assets/14ef1820-3d6a-4e8c-b26e-d114b2fe34c6)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
