# Divide fer2013 into three different .csv files for easy utilization
# ferTrain and ferDev will be used for Training
# ferTest will be used for test accuracy for final accuracy

""" 
    Types of Data : Training , Public Test , Private Test
    Here Training Data is taken as train data , Public Test as dev test , Private Test as Testing set

    You can also divide Public Test into 2 parts for dev and test set and use PrivateTest as productionTest
"""

from __future__ import print_function
import numpy as np

# get the data
filname = 'fer2013.csv'

ferTrain = open("ferTrain.csv","w")
ferDev = open("ferDev.csv","w")
ferTest = open("ferTest.csv","w")

Y = []
X = []
first = True
for line in open(filname):
    if first:
        first = False
    else:
        row = line.split(',')
        
        # get type of data 
        row = row[2].split('\n')
        type_of_data = row[0]

        if type_of_data == "Training" :
        	print( line , file=ferTrain , end="")
        elif type_of_data == "PublicTest" :
        	print( line , file=ferDev , end="")
        elif type_of_data == "PrivateTest" :
        	print( line , file=ferTest , end="")

ferTrain.close()
ferDev.close()
ferTest.close()

# Now ferTrain.csv contains 27809 images data
#     ferTrain.csv contains 3589  images data
#     ferTrain.csv contains 3589  images data