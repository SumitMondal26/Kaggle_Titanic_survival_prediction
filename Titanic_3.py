#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

#setting visual table in pycharm
plt.style.use('ggplot')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#to control randomness of randomness
np.random.seed(1)

#importing train and test files
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#creating function to clean test and train data
def clean_Data(train):
    
    #categorical data is transformed by first converting it into category data type then performing data[column].cat.codes
    train.Sex=train.Sex.astype('category')
    train.Sex=train.Sex.cat.codes

    train.Embarked=train.Embarked.astype('category')
    train.Embarked=train.Embarked.cat.codes
    
    #to fill unknoun tickets with "U" and then performing string extraction of given pattern
    #thus eg :A3554346 yeild A which is deck of which ticket is bought 
    train.Cabin.fillna("U",inplace=True)
    train.Cabin=train.Cabin.str.extract("([A-Za-z])")
    train.Cabin=train.Cabin.astype('category')
    train.Cabin=train.Cabin.cat.codes
    
    #to fill unknown ticket fare with median 
    #not mean because data may contain outliers which can affect mean value
    train.Fare.fillna(train.Fare.median(),inplace=True)
    train.Fare=train.Fare.astype(int)
    #filled train fare is then binned based on frequency of fare price of tickets  
    train.Fare=pd.cut(train.Fare,bins=[-100,7,14,25,35,50,100,200,1000],labels=[i for i in range(1,9)])

    #Now to fill age simple mean is not sufficient , so an another array of size of missing age rows is created and filled with 
    #random age between mean and using std deviance 
    mean=train.Age.mean()
    std=train.Age.std()
    size=train.Age.isna().sum()
    rand=np.random.randint(mean-std,mean+std,size)

    age=train.Age.copy()
    age[np.isnan(age)]=rand
    train.Age=age

    train.Age=pd.cut(train.Age,bins=[-100,5,10,14,18,30,40,60,200],labels=[i for i in range(1,9)])
    
    #new features have been created to better take advantage of features
    
    # Age*Class to depict importance of person to be saved during incident
    train['Age*Class']=train.Age.values*train.Pclass.values
    
    #to extract title from names and downsizing them to group of only important titles
    train['Title']=train.Name.str.extract("([A-Za-z]+\.)")

    train.Title.replace(['Rev.','Dr.','Major.','Sir.','Col.','Capt.'],'Rare.',inplace=True)
    train.Title.replace(['Don.','Jonkheer.'], 'Mr.', inplace=True)
    train.Title.replace(['Dona.', 'Mme.','Lady.','Countess.'], 'Mrs.', inplace=True)
    train.Title.replace(['Ms.','Mlle.'], 'Miss.', inplace=True)

    train.Title=train.Title.astype('category')
    train.Title=train.Title.cat.codes

    train['Age*Title']=train.Age.values*train.Title.values
    
    #feature created to denote if passenger is alone or with family
    train['is_NotAlone']=train.SibSp.values+train.Parch.values
    train.loc[train.is_NotAlone>0,'is_NotAlone']=1
    
    #drop unwanted features
    train.drop(['Ticket','Name'],axis=1,inplace=True)


clean_Data(test)
clean_Data(train)

#just for backup of passenger ID during output file
from copy import deepcopy
test_copy=deepcopy(test)

test.drop('PassengerId',axis=1,inplace=True)

#X and y data defined to fit
X=train.drop(['PassengerId','Survived'],axis=1)
y=train.Survived

#min max scaler to transform data
array=MinMaxScaler( )
X_s=array.fit_transform(X)
X_s=pd.DataFrame(X_s,columns=X.columns)

test_s=array.fit_transform(test)
test_s=pd.DataFrame(test_s,columns=test.columns)

#test train split during model testing
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Random forest classifier is being used
model=RandomForestClassifier(n_estimators=2000,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=20,bootstrap=True,n_jobs=-1)
model.fit(X_s,y)
# print('score:',(model.score(X_test,y_test)*100),'%')
y_pred=model.predict(test_s)

#to plot feature importance after fitting data

important=model.feature_importances_*100
df=pd.DataFrame({'Features':X.columns,'Impotance':important},index=X.columns)
df=df.sort_index(by='Impotance')
sns.barplot('Impotance','Features',data=df,color='red',alpha=0.8)
plt.show()

#output file for submission
out=pd.DataFrame({'PassengerId':test_copy.PassengerId,'Survived':y_pred})#
out.to_csv('Survived_pred_2.csv',index=False)


#For getting best parameters for given data and model usinf Random Search CV cause grid cv Sucks on my computer😂

# print("pram is use :\n",model.get_params())
#
# params={'n_estimators':[50,100,300,500,1000,2000],'max_features':['auto','sqrt'],'max_depth':[5,7,10,20,50,100],
#         'bootstrap':[True,False],'min_samples_leaf': [1, 2, 4],'min_samples_split':[2, 5, 10]}
#
# rfCV=RandomizedSearchCV(model,n_iter=100,param_distributions=params,scoring='accuracy',cv=5)
# best_model=rfCV.fit(X_train,y_train)
# print("best parameters :\n",best_model.best_params_)

