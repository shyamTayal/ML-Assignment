'''
@author : Shyam Tayal 
Predicting House price using Mini Batch Gradient Descent and comparing it's performance with and without 
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
training_size = 380

X_train = rows[:training_size]                                              # Training Data
Y_train = Price[: training_size]                                          # Cost in Training Data

test_data = rows[training_size:num]                                   # Testing Data
Y_test = Price[training_size:num]

lotsize = []
for r in rows :
    lotsize.append(r[1])

# Using Mini batch gradient descent with feature scaling with batch size = 20
# here as we see in the data no. of bedrooms and bathrooms don't need scaling as they have values in smaller ranges
# To scale lotsize in smaller ranges

theta = [1,1,1,1]
alpha = 0.3
batchsize =20
print(f'Taking Mini Batch Size : {batchsize}\n')
nofbatch = training_size//batchsize
scaled_train = np.array(X_train, dtype='double')

mean = np.mean(lotsize[:training_size])
max_min = np.max(lotsize[:training_size]) - np.min(lotsize[:training_size])

for i in range(training_size) :
    scaled_train[i][1] = (X_train[i][1] - mean)/max_min

# print(f"Initial Theta: {theta} ")
for x in range(150) :
    for batch in range(nofbatch) :

        for j in range(4) :

            temp = 0
            for i in range(batch*batchsize , (batch+1)*batchsize) :
                diff = 0.0
                for k in range(4) :
                    diff += (theta[k]*scaled_train[i][k])

                temp += (diff - Y_train[i])*scaled_train[i][j]

            theta[j] -= (temp/training_size)*alpha

# print(f"Final theta:{theta}")

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
print('Percent Error in Mini batch gradient descent without Regularisation : %.2f %% \n'%(error*100))


theta = np.array([0,0,0,0],dtype='double')
alpha = 0.0007
lmda = -400

# print(f"Initial Theta: {theta} ")
for x in range(150) :
    for batch in range(nofbatch) :

        for j in range(4) :

            temp = 0
            for i in range(batch*batchsize , (batch+1)*batchsize) :
                diff = 0.0
                for k in range(4) :
                    diff += (theta[k]*scaled_train[i][k])

                temp += (diff - Y_train[i])*scaled_train[i][j]

            if( j== 0) :
                theta[j] -= (temp/training_size)*alpha
            else :
                theta[j] = (1 - alpha * lmda / training_size) * theta[j] - (temp / batchsize) * alpha

# print(f"Final theta:{theta}")

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
print('Percent Error in Mini batch gradient descent with Regularisation : %.2f %% \n'%(error*100))