import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import linear_model,preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r'C:\Users\Ariya Rayaneh\Desktop\Car_Data_Set_5582ae97ba\Car Data Set\car.data')

le = preprocessing.LabelEncoder()
dff=[]
buying=le.fit_transform(df.buying)
dff.append(buying)
maint=le.fit_transform(df.maint)
dff.append(maint)
door=le.fit_transform(df.door)
dff.append(door)
persons=le.fit_transform(df.persons)
dff.append(persons)
lug_boot=le.fit_transform(df.lug_boot)
dff.append(lug_boot)
safety=le.fit_transform(df.safety)
dff.append(safety)
class_=le.fit_transform(df['class'])
dff.append(class_)


dff= {'buying':le.fit_transform(df.buying),'maint':le.fit_transform(df.maint) ,'door':le.fit_transform(df.door), 'persons':le.fit_transform(df.persons), 'lug_boot':le.fit_transform(df.lug_boot), 'safety':le.fit_transform(df.safety),'class':le.fit_transform(df['class'])}

title=['buying', 'maint', 'door', 'persons', 'lug_boot', 'safety', 'class']
dff=pd.DataFrame(dff,columns=title)

print(dff)

x=dff.drop('class',axis=1)
y=dff['class']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

scores=[]
for i in range(3,30):
 model=KNeighborsClassifier(n_neighbors=i)

 model.fit(x_train,y_train)
 y_predict=model.predict(x_test)
 score=model.score(x_test,y_test)
 scores.append(score)

print(scores)

i=[i for i in range(3,30)]
font1 = {'family':'serif','color':'blue','size':20}
plt.figure(figsize=(10,10))
plt.plot(i,scores,color='r')
plt.xlabel('n_neighbors(K)',fontdict=font1)
plt.xticks(fontsize=10,rotation=30)
plt.yticks(fontsize=10,rotation=30)
plt.ylabel('score',fontdict=font1)
plt.title('score VS n_neighbors(K) Factor',fontdict=font1)
plt.show()