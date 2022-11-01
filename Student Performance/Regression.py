import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#   Read the csv file which has student data. We split this file using ';' and store it in dataframe 'data'
data = pd.read_csv("student_mat.csv", sep=";")
# print(data.head())

# Removing the columns which wont help us in guessing the final grade G3 of a student
data = data[["G1", "G2" , "G3" , "studytime" , "failures" , "absences"]]
# print(data.head())

# Our aim is to find the G3 attribute while all other attribute values are known
predict = "G3"

# x is the df which has all other attributes except the one to be predicted i.e. G3
x = np.array(data.drop([predict], 1))
# y is the df which has only the G3 attribute/column
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# The below lines were used to generate a model by fitting the model to 30 different types of train and test data. and storing the model which has the best accuracy out of the 30 iterations.
# The model is stored using pickle, which is an in-built python library. Alternatively, joblib library can be used to save a model in local file system.
'''
best = 0
for i in range(30):
    # splitting x and y dataframes into train and test sets. The training set is 10% of the data while the train is 90%
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy is ",acc)
    if(acc>best):
        best = acc
        with open("StudentModel.pickle","wb") as f:
            pickle.dump(linear,f)
        # coefficient = linear.coef_
        # intercept = linear.intercept_
        # predictions = linear.predict(x_test)
    print(f"The best accuracy was {best}")
    '''
#The below lines help us in loading the already pickled model StudentModel.pickle and using it to predict x_test data.
file = "StudentModel.pickle"
# The StudentModel.pickle is loaded as "linear"
linear = pickle.load(open(file,'rb'))

# We can get the coefficients and intercept values of the loaded model.
print(f"Coefficients of the pickled model {file}: \n",linear.coef_)
print(f"Intercept of the pickled model {file}: \n",linear.intercept_)

# We find the score of the model linear against x_test and y_test
print(f"Score: {linear.score(x_test,y_test)}")
# We use 'linear' model to predict the values for the x_test values and store them in list "predictions" 
predictions = linear.predict(x_test)

# The predictions, their input x_test value, and the correct/expected y_test value are printed.
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
# print("Accuracy is ",best)
# print("X_TEST")
# print(x_test)
# print("Y_TEST")
# print(y_test)

# To store the input values got from user
user_input= []
print("Enter the prompted values, so we can predict value of G3:")
variables=["G1","G2","studytime","failures","absences"]
# G1_input = int(input("Enter G1: "))
# G2_input = int(input("Enter G2: "))
# studytime_input = int(input("Enter studytime: "))
# failures_input = int(input("Enter failures: "))
# absences_input = int(input("Enter absences: "))
j = 0
for i in range(5):
    print(f"Enter {variables[j]}:")
    j += 1
    item = int(input())
    user_input.append(item)
np_user_input = np.array(user_input)
np_user_input = np_user_input.reshape(1,-1)
print(np_user_input)
# user_prediction = linear.predict([G1_input, G2_input, studytime_input, failures_input, absences_input])

user_prediction = linear.predict(np_user_input)
print(f"The predicted value for final score for the given user inputs is: {user_prediction}")
#We store the name of the column for whose correlation with col G3 need to be found.
p = "G1"
style.use("ggplot")

# We plot G1 column against x-axis and G3(final score) column against y-axis.
pyplot.scatter(data[p],data[predict])

# We give the labels which we desire to see in the scatter plot.
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
