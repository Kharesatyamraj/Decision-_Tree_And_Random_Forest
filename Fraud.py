# -*- coding: utf-8 -*-
"""


@author: khare
"""

#3)Build a Decision Tree & Random Forest model on the fraud data. 
#Treat those who have taxable_income <= 30000 as Risky and others as
# Good (discretize the taxable_income column)

data = pd.read_csv(r'\Fraud_check.csv')

####Data Preprocessing

data.isna().sum() # no null values
data.duplicated().sum() # no duplicates 

##Discretization of Sales attribute
bin_Tax = ['Risky','Good']# list of labels under which countinuos data grouped
#Creating new cols TaxBin and dividing 'Taxable.Income' cols on the basis of [10002,30000,99620] for Risky and Good
data["TaxBin"] = pd.cut(data["Taxable.Income"], bins = [10002,30000,99620], labels = bin_Tax)
count= data['TaxBin'].value_counts()
count
data.drop(["Taxable.Income"],1,inplace = True)
#EDA
data.info()
data.describe()
data.skew()
data.kurt()
pd.DataFrame.hist(data)
#sns.pairplot(p)

#One HotEncoding 
lb = LabelEncoder()
data.head()
data["Undergrad"] = lb.fit_transform(data["Undergrad"])
data["Marital.Status"] = lb.fit_transform(data["Marital.Status"])
data["Urban"] = lb.fit_transform(data["Urban"])


x = data.iloc[:,:-1].values
y = data.iloc[:,5:].values


# Splitting the dataset into the Training set and Test set

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

# feature scaling

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test =  std.fit_transform(x_test)

# Training the Decision Tree on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion= "entropy",random_state=0)
classifier.fit(x_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(x_test)

#Accuracy test and Confusion matrix
print(accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print(cm)


#Testing Train dataset
Train_pred = classifier.predict(x_train)

#Accuracy test and Confusion matrix
print(accuracy_score(y_train,Train_pred))

cm = confusion_matrix(y_train,Train_pred)
print(cm)

#overfitting ----Need to Optimize
decision_new = DecisionTreeClassifier(max_depth=10,min_samples_split=250,criterion='entropy')
decision_new .fit(x_train,y_train)

# Predicting the Test set results
y_pred = decision_new.predict(x_test)

#Accuracy test and Confusion matrix
print(accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print(cm)


#Testing Train dataset
Train_pred = decision_new.predict(x_train)

#Accuracy test and Confusion matrix
print(accuracy_score(y_train,Train_pred))
cm = confusion_matrix(y_train,Train_pred)
print(cm)
# Training the Random Forest on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,criterion= "entropy",random_state=0)
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

#Accuracy test and Confusion matrix
print(accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print(cm)


#Testing Train dataset
Train_pred = classifier.predict(x_train)

#Accuracy test and Confusion matrix
print(accuracy_score(y_train,Train_pred))

cm = confusion_matrix(y_train,Train_pred)
print(cm)

#overfitting ----Need to Optimize
forest_new = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')
forest_new .fit(x_train,y_train)

# Predicting the Test set results
y_pred = forest_new.predict(x_test)

#Accuracy test and Confusion matrix
print(accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print(cm)


#Testing Train dataset
Train_pred = forest_new.predict(x_train)

#Accuracy test and Confusion matrix
print(accuracy_score(y_train,Train_pred))
cm = confusion_matrix(y_train,Train_pred)
print(cm)

