from pylab import rand,plot,show,norm
import numpy as np
import random

class DataSet:

    def __init__(self,no_of_records, type):
        self.no_of_records = no_of_records
        self.type = type

    def generateLinearData(self, n):
        """
        generates a 2D linearly separable dataset with n samples, where the third element is the label
        """
        inputs = []
        Y = []

        w = np.array([0.5,0.5])
        w0 = np.array([0])
        X = np.append((np.random.randn(n,2)+1.5),(np.random.randn(n,2)-1.5),axis =0)

        for i in range(len(X)):
            if (w0 + np.dot(w,X[i])) > 0:
                inputs.append([X[i][0], X[i][1], 1])
            elif (w0 + np.dot(w,X[i])) < 0:
                inputs.append([X[i][0], X[i][1], -1])
        Y = np.array(Y)

        return inputs

    def generateNonLinearData(self, n):

        """
        generates a 2D non linearly separable dataset with n samples, where the third element is the label
        """

        w = np.array([0.5,0.5])
        w0 = np.array([0])

        inputs = []
        X = np.append((np.random.randn(n,2)+0.1),(np.random.randn(n,2)-0.1),axis =0)
        Y = []

        counter = 0
        print bool(random.getrandbits(1))
        for i in range(len(X)):

            if (w0 + np.dot(w,X[i])) > 0:
                if(bool(random.getrandbits(1))):
                    inputs.append([X[i][0], X[i][1], 1])
                else:
                    inputs.append([X[i][0], X[i][1], -1])
            elif (w0 + np.dot(w,X[i])) < 0:
                if(bool(random.getrandbits(1))):
                    inputs.append([X[i][0], X[i][1], -1])
                else:
                    inputs.append([X[i][0], X[i][1], 1])
        Y = np.array(Y)

        return inputs

    def GenerateDataset(self):

        if self.type == 'linear':
            linear_dataset = self.generateLinearData(self.no_of_records)

            # Write linear dataset
            f = open('linear_dataset.txt', 'w')
            for i in range(len(linear_dataset)):
                line = linear_dataset[i]
                updateLine=str(line[0])+','+str(line[1])+','+str(line[2])
                f.write(str(updateLine)+'\n')
            f.close()
            print "Linear dataset is generated"
        else:
            non_linear_dataset = self.generateNonLinearData(self.no_of_records)
            # Write non linear dataset
            f = open('non_linear_dataset.txt', 'w')
            for i in range(len(non_linear_dataset)):
                line = non_linear_dataset[i]
                updateLine=str(line[0])+','+str(line[1])+','+str(line[2])
                f.write(str(updateLine)+'\n')
            f.close()
            print "Non linear dataset is generated"

