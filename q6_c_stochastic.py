'''
@author : Shyam Tayal 
Predicting House price using stochastic Gradient Descent and comparing it's performance with and without 
'''

import numpy as np
import csv

filename = 'Housing_Price_data_set.csv'
rows = []
Price = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    fields = next(csvreader)
    for row in csvreader:
        rows.append([1.0, float(row[2]), float(row[3]), float(row[4])])
        Price.append(float(row[1]))
num = len(rows)
training_size = (num*7) // 10

X_train = rows[:training_size]                                              # Training Data
Y_train = Price[: training_size]                                          # Cost in Training Data

test_data = rows[training_size:num]                                   # Testing Data
Y_test = Price[training_size:num]

lotsize = []
for r in rows :
    lotsize.append(r[1])

# Using Stochastic gradient descent with feature scaling without Regularisation

scaled_train = np.array(X_train, dtype='double')
mean = np.mean(lotsize[:training_size])
max_min = np.max(lotsize[:training_size]) - np.min(lotsize[:training_size])

for i in range(training_size) :
    scaled_train[i][1] = (X_train[i][1] - mean)/max_min

theta = np.array([0,0,0,0],dtype='double')
alpha = 0.000777

# print(f"Initial Theta: {theta} ")
for k in range(100) :

      for i in range(training_size) :
            temptheta = theta.copy()
            for l in range(4) :
                  temp = 0
                  for j in range(4):
                        temp += theta[j]*scaled_train[i][j]

                  temptheta[l] -= alpha * (temp - Y_train[i]) * scaled_train[i][l]
            theta = temptheta.copy()

# print(f"Final Theta: {theta} ")

error = 0
testing_size = len(test_data)
for i in range(testing_size) :
    y_pred = 0
    for j in range(4) :
        if j != 1 :
            y_pred += test_data[i][j] * theta[j]
        else :
            y_pred += ((test_data[i][j] - mean)/max_min)*theta[j]

    error += abs(Y_test[i] - y_pred)/Y_test[i]

error /= testing_size

print(f'\nLearning Rate : {alpha}')
print('Percent Error in Stochastic gradient descent without Regularisation : %.2f %% \n'%(error*100))

# Using Stochastic gradient descent with feature scaling with Regularisation
lmda = 0.0001
theta = np.array([1,1,1,1],dtype='double')
alpha = 0.0007

# print(f"Initial Theta: {theta} ")
for k in range(100) :

      for i in range(training_size) :
            temptheta = theta.copy()
            for l in range(4) :
                  temp = 0.0
                  for j in range(4):
                        temp += temptheta[j]*scaled_train[i][j]

                  if(l == 0) :
                        temptheta[l] -= alpha * (temp - Y_train[i]) * scaled_train[i][l]
                  else :
                        temptheta[l] *= (1 - alpha * lmda )
                        temptheta[l] -= (temp - Y_train[i])*scaled_train[i][l]*alpha

            theta = temptheta.copy()
# print(f"Final Theta: {theta} ")

error = 0
testing_size = len(test_data)
for i in range(testing_size) :
    y_pred = 0
    for j in range(4) :
        if j != 1 :
            y_pred += test_data[i][j] * theta[j]
        else :
            y_pred += ((test_data[i][j] - mean)/max_min)*theta[j]

    error += abs(Y_test[i] - y_pred)/Y_test[i]

error /= testing_size

print(f'\nLearning Rate : {alpha} \nLamba : {lmda}')
print('Percent Error in Stochastic gradient descent with Regularisation :  %.2f %% \n'%(error*100))
