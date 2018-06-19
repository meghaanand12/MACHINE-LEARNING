#! usr/bin/python3

import os 


from sklearn import tree

data = [[100,0],[130,0],[135,1],[150,1]]

#target
output=["apple","apple","orange","orange"]

#decision tree algo call
algo = tree.DecisionTreeClassifier()

#train data
train_algo = algo.fit(data,output)

#now testing phase
predict = train_algo.predict([[100,1]])

#printing output
print(predict)
