import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('Decision-Tree-Classification-Data.csv')
print(df.head)
x=df.drop('diabetes',axis=1)
y=df.diabetes

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

model=DecisionTreeClassifier()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy = metrics.accuracy_score(y_test,y_pred)
print('Accuracy: ',accuracy)
print('Confusion Matrix:')
cm = confusion_matrix(y_test,y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['0','1'])
cmd.plot()
#print(confusion_matrix(y_test,model.predict(x_test)))
x_new=[43,85]
x_new=pd.DataFrame([x_new])
print('New Sample:\n',x_new)
y_new=model.predict(x_new)
print('Predicted class: ',y_new)