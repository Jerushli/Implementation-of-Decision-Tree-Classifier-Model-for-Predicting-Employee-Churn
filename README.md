# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

step 1:start the program

step 2: Import the standard Libraries.

stap 3:Set variables for assigning dataset values.

step 4:Import linear regression from sklearn.

step 5:Assign the points for representing in the graph.

step 6:Predict the regression for marks by using the representation of the graph.

step 7: Compare the graphs and hence we obtained the linear regression for the given datas.

step 8:End the program

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JERUSHLIN JOSE JB
RegisterNumber: 212222240039
```
```python
08 employee churn
import pandas as pd
data = pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

### Data:

![Screenshot 2024-09-20 135323](https://github.com/user-attachments/assets/28c5a398-7f32-4b79-acd0-ec36a2c76c60)

### Accuracy:

![image](https://github.com/user-attachments/assets/b9b1dd85-7b53-4299-beda-b04dd0b7b9d3)

### Predict:

![image](https://github.com/user-attachments/assets/1b084a37-90af-4668-8285-51097e9ec346)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
