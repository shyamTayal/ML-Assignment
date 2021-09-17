'''
@author : Shyam Tayal 
Predicting House price using Batch Gradient Descent and comparing it's performance with and without 
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
        rows.append([1, int(row[2]), int(row[3]), int(row[4])])
        Price.append(int(row[1]))
num = len(rows)
training_size = (num*7) // 10

X_train = rows[:training_size]                                              # Training Data
Y_train = Price[: training_size]                                          # Cost in Training Data

test_data = rows[training_size:num]                                   # Testing Data
Y_test = Price[training_size:num]

lotsize = []
for r in rows :
    lotsize.append(r[1])

# Using batch gradient descent with feature scaling without Regularisation
scaled_train = X_train.copy()

mean = np.mean(lotsize[:training_size])
max_min = np.max(lotsize[:training_size]) - np.min(lotsize[:training_size])

for i in range(training_size) :
    scaled_train[i][1] = (X_train[i][1] - mean)/max_min

theta = [0,0,0,0]
alpha = 0.001

# print(f"Initial Theta: {theta} ")

for rep in range(3000) :                                                # Reapeating for 3000 times

        for j in range(4) :
            
            temp = 0
            for i in range(training_size) :
                diff = 0
                for k in range(4) :
                    diff += (theta[k]*scaled_train[i][k])

                temp += (diff - Y_train[i])*scaled_train[i][j]

            theta[j] -= (alpha*temp)/training_size                  # Updating theta

# print(f"Final theta:{theta}")


mean = np.mean(lotsize[training_size:num])
max_min = np.max(lotsize[training_size:num]) - np.min(lotsize[training_size:num])


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
print(f'Percent Error in batch gradient descent without regularisation : %.2f %% \n'%(error*100))



# Using batch gradient descent with feature scaling and Regularisation

lmda = -100
theta = [0,0,0,0]
alpha = 0.001

# print(f"Initial Theta: {theta} ")
for rep in range(3000) :                                                # Reapeating for 3000 times

      for j in range(4) :
            temp = 0
            for i in range(training_size) :
                diff = 0
                for k in range(4) :
                    diff += (theta[k]*scaled_train[i][k])

                temp += (diff - Y_train[i])*scaled_train[i][j]
            if(j == 0) :
                  theta[j] -= (alpha*temp)/training_size 
            else :
                  theta[j] = (1 - alpha * lmda / training_size)*theta[j] - ((alpha*temp)/training_size)            

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
# print(f"Final theta:{theta}")
print(f'\nLearning Rate : {alpha} \nLamba : {lmda}')
print(f'Percent Error in batch gradient descent with regularisation : %.2f %% \n'%(error*100))
