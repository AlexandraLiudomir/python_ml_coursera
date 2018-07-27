import numpy as np
import pandas

data = pandas.read_csv('_ea07570741a3ec966e284208f588e50e_titanic.csv', index_col = 'PassengerId')

men = data['Sex'].value_counts()
survived = data['Survived'].mean(axis = 0)
meanAge = data['Age'].mean(axis=0)
medAge =  data['Age'].median(axis=0)
cl = data['Pclass'].value_counts()
#print(cl)

siblings = data['SibSp']
parch = data['Parch']

x = data.corr('pearson',1)
#print(x)

name = data['Name']

def getFemaleName(inp):
    if str(inp).find('Mrs.')!= -1:
        if str(inp).__len__()>1:
            return str(inp).split('(')[1].split(' ')[0]
        else:
            return 'NO BRACKET in'+str(inp)
    elif str(inp).find('Miss.')!= -1:
        return str(inp).split(' ')[2]
    else:
        return '-'



print(name.apply(getFemaleName))