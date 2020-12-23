import doctest
import itertools
import random
import math
import numpy as np
from Line_Iris import Line_Iris
from H_Iris import H_Iris
from Point_Iris import Point_for_Iris

""" 
Names: Cohen Yarden 207205972
       Shalel Shani 206134033
we worked together on this Submission assignment.
"""

"""  version of python is 3.8"""

"""
Results for Iris of 100 runs :
Train: the rate of success for 1 is 95.79007048540973 percent 
Test: the rate of success for 1 is 88.84525071391849 percent 

Train: the rate of success for 2 is 95.32296656547611 percent 
Test: the rate of success for 2 is 88.05921138189647 percent 

Train: the rate of success for 3 is 96.85653606387723 percent 
Test: the rate of success for 3 is 88.97728654484678 percent 

Train: the rate of success for 4 is 96.89628077640391 percent 
Test: the rate of success for 4 is 88.86257659566128 percent 

Train: the rate of success for 5 is 98.13148312725687 percent 
Test: the rate of success for 5 is 88.7424177297434 percent 

Train: the rate of success for 6 is 98.15840722841033 percent 
Test: the rate of success for 6 is 88.87828530048694 percent 

Train: the rate of success for 7 is 98.50733885438872 percent 
Test: the rate of success for 7 is 88.81806022974797 percent 

Train: the rate of success for 8 is 98.66936544627275 percent 
Test: the rate of success for 8 is 88.57381834591479 percent 
"""

""" Do you see overfitting? no 
To get overffiting, the learning about the points of the train must be very good.
In addition - given a new point from the test set we will not know how to classify it. 
Its mean that the classification of the test points will be poor and will yield very low results relative to the train.
We do not see this in Iris result because for the points of the test- we got good results that are close to the results
of the train ,(the difference is 10 between the highest(train) and the lowest(test) and we think that its not to much). 
"""


# rank the error of the line (if we ronge about the label we add the weight to the rate)
def line_error(line, points):
    rate = 0
    for p in points:
        if (line.is_right(p) == False):  # made an error
            rate += p.weight
            if (p.weight == 0):
                print("the w is :{}".format(p.weight))
    if (rate == 0):
        return 0.0000000000000000001
    else:
        return rate


# this function return the best rule (line) on the points
def best_line(points):
    best = Line_Iris((points[0], points[1]))
    best_rate = line_error(best, points)
    for p in itertools.combinations(points, 2):
        temp = Line_Iris(p,True)
        temp_rate = line_error(temp, points)
        if (temp_rate < best_rate):
            best = temp
            best_rate = temp_rate

    for p in itertools.combinations(points, 2):
        temp = Line_Iris(p,False)
        temp_rate = line_error(temp, points)
        if (temp_rate < best_rate):
            best = temp
            best_rate = temp_rate
    return best


def adaboost(points, rules=8):
    n = len(points)
    w = 1 / n
    for p in points:
        p.weight = w
    best_rules = []
    weight_rules = []  # weight=alpha_t
    alpha_t = 0.5
    for i in range(0, rules):
        sum = 0
        best_r = best_line(points)  # find the best rule
        best_rules.append(best_r)
        et = line_error(best_r, points)  # getting the error on the points
        num = (1 - et) / et
        alpha_t = 0.5 * (np.log(num))  # np=numpy
        weight_rules.append(alpha_t)
        for p in points:
            if (best_r.is_right(p) == False):
                p.weight = p.weight * (np.math.e ** (alpha_t))
            else:
                p.weight = p.weight * (np.math.e ** (-alpha_t))
            sum += p.weight
        for p in points:  # normalized the point weight
            p.weight = p.weight / sum

    ans = H_Iris(best_rules, weight_rules, 8)

    for p in points:
        p.weight = 1

    return ans


def run_train(points, rules=8, times=100):
    #for i in range(1, rules + 1):
        avg1=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        avg2=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        for j in range(0,times):
            #print("j:{}".format(j))
            learn = []
            test = []
            # filling the 2 list above with the points we get in random way
            for p in points:
                rand = random.randint(0, 1)
                if (len(learn) >= 65):
                    test.append(p)
                else:
                    if (len(test) >= 65):
                        learn.append(p)
                    else:
                        if (rand == 0):
                            test.append(p)
                        else:
                            learn.append(p)

            ans_learn = adaboost(learn)


            for i in range(1, rules + 1):
                rate = 0
                rate1 = 0  # shani add
                for p in learn:
                    if ans_learn.is_right_H(p,i):
                        rate += 1
                #print ("rate {} ".format((rate /len(learn) )*100))
                avg1[i]+=(rate /len(learn) )* 100


                for p1 in test:
                    if ans_learn.is_right_H(p1,i):
                        rate1 += 1
                avg2[i]+=(rate1 / len(test) )* 100

        for i in range (1,9):
            avg1[i] /= (times)
            print("Train: the rate of success for {} is {} percent ".format(i, avg1[i]))
            #shani add
            avg2[i] /= times
            print("Test: the rate of success for {} is {} percent ".format(i, avg2[i]))
            print("")


if __name__ == '__main__':

    print("Iris:")

    f = open("iris.txt", "r")
    points = []
    for x in f:
        points.append(Point_for_Iris(x))
    run_train(points, 8, 100)
