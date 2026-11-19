### AIM : 







Train RandomForest and AdaBoost classifiers on the breast cancer dataset and compare 
accuracies. 







### ALGORITHM : 









RandomForest = bagging many trees; AdaBoost = sequential boosting of weak learners 
focusing on hard examples.









### THEORY: 








Bagging reduces variance (averaging), boosting reduces bias (sequential focus on errors). 







## INPUT :



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

x,y = load_breast_cancer(return_X_y=True)

xtr, xte, ytr, yte = train_test_split(x,y,test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(xtr,ytr)

ab = AdaBoostClassifier(n_estimators=50, random_state=42).fit(xtr,ytr)

print('RF accuracy:', round(accuracy_score(yte, rf.predict(xte)),4))

print('AdaBoost accuracy:', round(accuracy_score(yte, ab.predict(xte)),4))




## OUTPUT:


RF accuracy: 0.9708

AdaBoost accuracy: 0.9708






### RESULT:







Students compare bagging vs boosting; performance depends on data and hyperparameters.
