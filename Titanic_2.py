import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize,MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix



np.random.seed(1)

plt.style.use('ggplot')
fig,ax=plt.subplots()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# sns.heatmap(train.isna(),cmap='viridis')
# plt.show()

def clean_data(train):


    train['title']=train.Name.str.extract("([A-Za-z]+\.)")
    train['title']=train['title'].replace(['Don.','Rev.','Dr.','Major.','Sir.','Col.','Capt.','Jonkheer.'],'Rare.')
    train['title']=train['title'].replace(['Mme.','Lady.','Countess.','Dona.'],'Mrs.')
    train['title'] = train['title'].replace(['Ms.', 'Mlle.'],'Miss.')

    train['title']=train['title'].map({'Mr.':1,'Mrs.':2,'Master.':3,'Miss.':4,'Rare.':5})

    train['Sex']=train['Sex'].map({'male':1,'female':0})

    train['relavtives']=train.SibSp+train.Parch

    train.loc[train['relavtives']>0,'is_alone']=0
    train['is_alone']=train.is_alone.fillna(1).astype(int)

    train.loc[train['relavtives']>4,'big_family']=1
    train['big_family']=train.big_family.fillna(0).astype(int)

    train.Cabin.fillna('U0',inplace=True)
    train.Cabin=train.Cabin.str.extract("([A-Za-z])")
    train.Cabin=train.Cabin.map({'U':0,'C':3,'E':5,'G':7,'D':4,'A':1,'B':2,'F':6,'T':8})

    train.Embarked.fillna('S', inplace=True)
    train.Embarked=train.Embarked.map({'S':1,'Q':2,'C':3})

    mean=train.Age.mean()
    std_dev=train.Age.std()
    miss_no=train.Age.isna().sum()
    rand_age=np.random.randint(mean-std_dev,mean+std_dev,miss_no)
    age=train.Age.copy()
    age[np.isnan(age)]=rand_age
    train.Age=age.astype(int)

    train.Fare.fillna(train.Fare.median(),inplace=True)
    train['Fare']=round(train.Fare)
    train.loc[train['Fare'] <= 7, 'Fare'] = 0
    train.loc[(train['Fare'] > 7) & (train['Fare'] <= 14), 'Fare'] = 1
    train.loc[(train['Fare'] > 14) & (train['Fare'] <= 31), 'Fare'] = 2
    train.loc[train['Fare'] > 31, 'Fare'] = 3
    train['Fare'] = train['Fare'].astype(int)

    train.loc[train.Age<15,'is_young']=1
    train.is_young.fillna(0,inplace=True)

    bin=[-100,5,10,16,18,35,50,65,100]
    train['Age']=pd.cut(train.Age,bin,labels=[1,2,3,4,5,6,7,8])

    train['Age*Class']=train.Age.values*train.Pclass.values

    train.drop(columns=['Ticket','Name'],inplace=True)
    train=train.values.astype(int)


clean_data(train)
clean_data(test)

X=train.drop(['Survived','PassengerId','Parch','is_young','big_family'],axis=1)
y=train['Survived']
test=test.drop(['PassengerId','Parch','is_young','big_family'],axis=1)

scaler=MinMaxScaler()
array=scaler.fit_transform(X)
X=pd.DataFrame(array,columns=X.columns)
array=scaler.fit_transform(test)
test=pd.DataFrame(array,columns=test.columns)


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=20,max_features=0.2, min_samples_leaf=8,random_state=20)
#randomforest = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
randomforest.fit(X, y)
y_pred = randomforest.predict(X)


# feature=randomforest.feature_importances_*100
# important=pd.DataFrame({'importance':feature},index=X.columns)
# important=important.sort_index(by='importance')
# print(important)
# important.plot(kind='barh')
# plt.show()