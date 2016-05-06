import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import KFold
from cvxopt import matrix,solvers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import generate_dataset
from numpy import linalg
from sklearn.metrics import confusion_matrix


# ---------- Iris dataset code -------------------
iris = datasets.load_iris()
X_temp = iris.data[0:100, :2]  # we only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
y_temp = iris.target[0:100]

X_List = []
y_List = []

for i in range(len(y_temp)):
    if( y_temp[i] == 0 or y_temp[i] == 1):
        X_List.append(X_temp[i])
        if y_temp[i] == 0:
            y_List.append(-1.0)
        else:
            y_List.append(1.0)

data_matrix = np.array(X_List)
target_matrix = np.array(y_List)

#print data_matrix
#print target_matrix

#exit()
# ---------------- Iris dataset code ends----------------

# ------------- Code for data generation ------------ #

"""

print " ~~~~~ Enter necessary parameters ~~~~~ "

print "Type of data: [linear, non_linear]"
print "Number of Example: : [50]"
print "Type of Margin : [hard, soft]"

response1 = raw_input("Please enter type of data: ")
response2 = int(raw_input("Please enter number of example: "))

while not response1:
    response1 = raw_input("Please enter data: ")

while not response2:
    response2 = int(raw_input("Please number of example: "))

obj_dataset = generate_dataset.DataSet(response2, response1)
obj_Generate_Dataset = obj_dataset.GenerateDataset()

data2 = open(response1+'_dataset.txt', 'r')
files = np.loadtxt(data2,dtype=str, delimiter=',')

data_matrix = np.array(files[:,0:-1], dtype='float')
target_matrix = np.array(files[:,-1], dtype='float')

# --------------- code for data generation ends --------- #
"""
def Find_W_Value(x_temp,y_temp,alpha_value, C=None):

    W_list = []

    if C is None:
        for i in range(len(alpha_value)):
            if (alpha_value[i]>0):
                value  = (alpha_value[i] * y_temp[0, i] * x_temp[i])
                W_list.append(value)
    else:
        for i in range(len(alpha_value)):
            if (0<alpha_value[i]<C):
                value  = (alpha_value[i] * y_temp[0, i] * x_temp[i])
                W_list.append(value)

    W_array = np.sum(np.array(W_list), axis=0)
    return W_array

def Find_W0_Value(x_train,y_train,w_value,alpha_value, C=None):

    W0_list = []
    count =0
    if C is None:
        for i in range(len(alpha_value)):
            if(alpha_value[i]>0):
                W0_list.append(y_train[0,i] - (w_value * x_train[i].T))
                count+=1
    else:
        for i in range(len(alpha_value)):
            if (0<alpha_value[i]<C):
                W0_list.append(y_train[0,i] - (w_value * x_train[i].T))
                count+=1

    W0_Val = np.sum(np.array(W0_list))/count
    return W0_Val


def project(W_Val_,W0_Val_,X_Test_Mat_):
    return np.dot(W_Val_, X_Test_Mat_.T) + W0_Val_

def predict(projected_arr):
    return np.sign(projected_arr)

def discriminant_function(W_Val,W0_Val,X_Test_Mat):

    predicted_Y = []
    value = (W_Val * X_Test_Mat.T) + W0_Val
    for i in range(value.shape[1]):
        if (value[0,i] > 0):
            predicted_Y.append(1.0)
        else:
            predicted_Y.append(-1.0)
    return predicted_Y

def linear_kernel(x1, x2):
        return np.dot(x1, x2)

def findQPSolver(X_Train_Data_, Y_Train_Data_, C=None):

    no_of_samples = len(X_Train_Data_)
    # Gram matrix
    K = np.zeros((no_of_samples, no_of_samples))
    for i in range(no_of_samples):
        for j in range(no_of_samples):
            K[i,j] = linear_kernel(X_Train_Data_[i], X_Train_Data_[j])

    P = matrix(np.outer(Y_Train_Data_,Y_Train_Data_) * K)
    q = matrix(np.ones(no_of_samples) * -1)
    A = matrix(Y_Train_Data_, (1,no_of_samples))
    b = matrix(0.0)

    if C is None:
        tmp1 = np.diag(np.ones(no_of_samples) * -1)
        tmp2 = np.diag(np.ones(no_of_samples))
        G = matrix(np.vstack((tmp1, tmp2)))

        tmp1 = np.zeros(no_of_samples)
        tmp2 = np.ones(no_of_samples) * 1
        h = matrix(np.hstack((tmp1, tmp2)))
    else:
        tmp1 = np.diag(np.ones(no_of_samples) * -1)
        tmp2 = np.diag(np.ones(no_of_samples))
        G = matrix(np.vstack((tmp1, tmp2)))

        tmp1 = np.zeros(no_of_samples)
        tmp2 = np.ones(no_of_samples) * C
        h = matrix(np.hstack((tmp1, tmp2)))

    sol = solvers.qp(P, q, G, h, A, b)

    return sol['x']

def plot_points(X1_train, X2_train, w, w0, sv_1, sv_2):

    w = np.ravel(w)
    def f(x, w, b, c=0):
        return (-w[0] * x - b + c) / w[1]

    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "bo")
    plt.scatter(sv_1, sv2, s=100, c="g", marker="*")

    """
    # w.x + b = 0
    a0 = -4; a1 = f(a0, w, w0)
    b0 = 4; b1 = f(b0, w, w0)
    plt.plot([a0,b0], [a1,b1], "k")

    # w.x + b = 1
    a0 = -4; a1 = f(a0, w, w0, 1)
    b0 = 4; b1 = f(b0, w, w0, 1)
    plt.plot([a0,b0], [a1,b1], "k--")

    # w.x + b = -1
    a0 = -4; a1 = f(a0, w, w0, -1)
    b0 = 4; b1 = f(b0, w, w0, -1)
    plt.plot([a0,b0], [a1,b1], "k--")
    """
    plt.axis("tight")
    plt.show()

def findOtherParameters(confusion_mat):

    list_diagonal = np.zeros(confusion_mat.shape[0])
    list_row_sum = np.zeros(confusion_mat.shape[0])
    list_column_sum=np.zeros(confusion_mat.shape[1])

    precision_value = []
    recall_value = []
    f_measure_value = []

    total = np.sum(confusion_mat)
    confuse_diagonal = 0

    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            list_row_sum[i] += confusion_mat[i][j]
            list_column_sum[i] += confusion_mat[j][i]
            if(i==j):
                list_diagonal[i] = confusion_mat[i][j]
                confuse_diagonal +=  confusion_mat[i][j]

    accuracy = float(confuse_diagonal)/total

    for index in range(len(list_row_sum)):
        if list_row_sum[index]==0:
            precision_value.append(0.0)
        else:
            precision_value.append((float)(list_diagonal[index]) / list_row_sum[index])

        if list_column_sum[index]==0:
            recall_value.append(0)
        else:
            recall_value.append((float)(list_diagonal[index]) / list_column_sum[index])

        if precision_value[index]==0 or recall_value[index]==0:
            f_measure_value.append(0)
        else:
            f_measure_value.append((float) (2 * precision_value[index] * recall_value[index]) / (precision_value[index] + recall_value[index]))

    return accuracy, precision_value, recall_value, f_measure_value

print "\n K-Fold"
kf = KFold(data_matrix.shape[0], n_folds= 10, shuffle=False)
print len(kf)

for train_index, test_index in kf:

    X_Train_Data, X_Test_Data = data_matrix[train_index], data_matrix[test_index]
    Y_Train_Data, Y_Test_Data = target_matrix[train_index], target_matrix[test_index]

    X_Train_Matrix = np.matrix(X_Train_Data)
    Y_Train_Matrix = np.matrix(Y_Train_Data)

    X_Test_Matrix = np.matrix(X_Test_Data)
    Y_Test_Matrix = np.matrix(Y_Test_Data)

    sv1 = []
    sv2 = []
    support_vector_List =[]


    alpha_value = findQPSolver(X_Train_Data, Y_Train_Data, C=None)
    W_Value = Find_W_Value(X_Train_Matrix,Y_Train_Matrix,alpha_value, C=None)
    W0_Value = Find_W0_Value(X_Train_Matrix,Y_Train_Matrix,W_Value,alpha_value, C=None)

    for i in range(len(alpha_value)):
        if(alpha_value[i] > 0):
            support_vector_List.append(alpha_value[i])
            sv1.append(X_Train_Data[i][0])
            sv2.append(X_Train_Data[i][1])


    # calculate support vector
    projected_array = np.ravel(project(W_Value,W0_Value,X_Test_Matrix))
    predictedy = predict(projected_array)


    correct = np.sum(predictedy == Y_Test_Data)

    print "%d out of %d predictions correct" % (correct, len(predictedy))
    print "Accuracy = ", str(accuracy_score(Y_Test_Data,predictedy) * 100) + " % "

    confusion_mat = confusion_matrix(Y_Test_Data, predictedy)
    print confusion_mat

    # find precision, recall , f-measure
    accuracy, precision_val, recall_val, f_measure_val = findOtherParameters(confusion_mat)
    print "Precision = ", precision_val
    print "Recall = ", recall_val
    print "F-Measure", f_measure_val
    print "\n Fold Completed \n"

    plot_points(X_Train_Data[Y_Train_Data==1], X_Train_Data[Y_Train_Data==-1], W_Value, W0_Value, sv1, sv2)
