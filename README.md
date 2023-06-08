## EXPERIMENT 01:IMPLEMENTATION OF UNIVARIATE LINEAR REGRESSION
```
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing Input data
x=np.array(eval(input("Enter array X: ")))
y=np.array(eval(input("Enter array Y: ")))

# Mean
x_mean=np.mean(x)
y_mean=np.mean(y)

# For slope
num,denom=0,0  

# To find sum of (xi-x') & (yi-y') & (xi-x')^2
for i in range(len(x)):
  num+=((x[i]-x_mean)*(y[i]-y_mean))
  denom+=(x[i]-x_mean)**2

# Calculate slope
m=num/denom
print("Slope: ",m)

# Calculate intercept
b=y_mean-m*x_mean
print("Intercept: ",b)

# Line equation
y_predicted=m*x+b
print(y_predicted)

# To plot graph
plt.scatter(x,y,color='pink')
plt.plot(x,y_predicted,color='blue')
plt.show()

# Predict y if x=3
y_3=m*3+b
print("If x=3 then y=",y_3)
```
```
# input:
8,2,11,6,5,4,12,9,6,1
3,10,3,6,8,12,1,4,9,14
```

## EXPERIMENT 02: IMPLEMENTATION OF SIMPLE LINEAR REGRESSION MODEL FOR PREDICTING THE MARKS SCORED
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#Displaying the content in datafile
df.head()

#Last five rows
df.tail()

#Segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#Splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#Displaying predicted values
Y_pred

#Displaying actual values
Y_test

#Graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="orange")
plt.title("Hours VS Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#MSE
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
#MAE
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
#RMSE
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
[student_scores.csv](https://github.com/KeerthikaNagarajan/ml/files/11688315/student_scores.csv)

## EXPERIMENT 03:IMPLEMENTATION OF LINEAR REGRESSION USING GRADIENT DESCENT
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit prediction")

def computeCost(X,y,theta): 
    #Take in an numpy array X,y,theta and generate the cost function in a linear regression model
    m=len(y)#Length of training data
    h=X.dot(theta)#Hypothesis
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err)#returning J

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)#Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
    #Take in numpy array X,y and theta and update theta by taking num_iters with learning rate of alpha
    #return theta and list of the cost of theta during each iteration
    m=len(y)
    J_history=[]
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*1/m*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

#Graph with Line of Best Fit
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="red")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

#Making predictions using optimized theta values
def predict(x,theta):
    #Takes in numpy array of x and theta and return the predicted value of y based on theta
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population= 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population= 70,000, we predict a profit of $"+str(round(predict2,0)))
```
[ex1.txt](https://github.com/KeerthikaNagarajan/ml/files/11688320/ex1.txt)

## EXPERIMENT 04: IMPLEMENTATION OF LOGISTIC REGRESSION MODEL TO PREDICT THE PLACEMENT STATUS OF STUDENT
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
[Placement_Data.csv](https://github.com/KeerthikaNagarajan/ml/files/11688323/Placement_Data.csv)

## EXPERIMENT 05:IMPLEMENTATION OF LOGISTIC REGRESSION USING GRADIENT DESCENT
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt('ex2data1.txt',delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

#Visualizing the data

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

#Sigmoid Function
def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J
def gradient(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
   xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train=np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
[ex2data1.txt](https://github.com/KeerthikaNagarajan/ml/files/11688326/ex2data1.txt)

## EXPERIMENT 06: IMPLEMENTATION OF DECISION TREE CLASSIFIER MODEL FOR PREDICTING EMPLOYEE CHURN 
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing  import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
data.head()

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
[Employee.csv](https://github.com/KeerthikaNagarajan/ml/files/11688333/Employee.csv)

## EXPERIMENT 07: IMPLEMENTATION OF DECISION TREE REGRESSOR MODEL FOR PREDICTING THE SALARY OF THE EMPLOYEE
```
import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]
x.head()

y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
[Salary.csv](https://github.com/KeerthikaNagarajan/ml/files/11688337/Salary.csv)

## EXPERIMENT 08:IMPLEMENTATION OF K MEANS CLUSTERING FOR CUSTOMER SEGMENTATION
```
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[] #Within-Cluster Sum of Square

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="yellow",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="pink",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="purple",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```
[Mall_Customers (1).csv](https://github.com/KeerthikaNagarajan/ml/files/11688341/Mall_Customers.1.csv)

## EXPERIMENT 09: IMPLEMENTAION OF SVM FOR SPAM MAIL DETECTION
```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

data.head()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.fit_transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
[spam.csv](https://github.com/KeerthikaNagarajan/ml/files/11688344/spam.csv)
