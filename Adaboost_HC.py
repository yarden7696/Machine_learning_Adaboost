# got some help from github user we found https://github.com/eliahusatat/Machine-Learning
import doctest
import itertools
import random
import math
import numpy as np
from Line_HC import Line
from Point_HC import Point_for_HC
from H_HC import H


""" 
Names: Cohen Yarden 207205972
       Shalel Shani 206134033
we worked together on this Submission assignment.
"""

"""  version of python is 3.8"""

"""
HC_Body_Temperature :
Train: the rate of success for 1 is 68.75384615384611 percent 
Test: the rate of success for 1 is 50.753846153846155 percent 

Train: the rate of success for 2 is 67.36923076923077 percent 
Test: the rate of success for 2 is 50.75384615384617 percent 

Train: the rate of success for 3 is 75.92307692307696 percent 
Test: the rate of success for 3 is 50.30769230769229 percent 

Train: the rate of success for 4 is 73.46153846153847 percent 
Test: the rate of success for 4 is 50.86153846153847 percent 

Train: the rate of success for 5 is 79.99999999999999 percent 
Test: the rate of success for 5 is 50.96923076923078 percent 

Train: the rate of success for 6 is 78.39999999999998 percent 
Test: the rate of success for 6 is 50.815384615384616 percent 

Train: the rate of success for 7 is 83.43076923076924 percent 
Test: the rate of success for 7 is 51.87692307692309 percent 

Train: the rate of success for 8 is 81.98461538461535 percent 
Test: the rate of success for 8 is 50.69230769230771 percent
"""

""" Do you see overfitting? yes 
To get overffiting, the learning about the points of the train must be very good.
In addition - given a new point from the test set we will not know how to classify it. 
Its mean that the classification of the test points will be poor and will yield very low results relative to the train.
We see this in HC_Body_Temperature result because for the points of the test- we got bad results that are not 
close to the results of the train,(the difference is 30 between the highest(train) and the lowest(test) and we
think that its to much). 
"""


# rank the error of the line (if we ronge about the label we add the weight to the rate)
# compute the empirical error
def line_error(line, points):
    rate=0
    for p in points:
        if(line.is_right(p)==False):  #made an error
            rate+=p.weight
            if(p.weight==0):
                print("the w is :{}".format(p.weight))
    if(rate==0):
        return 0.000000001
    else:
        return rate



#this function return the best rule (line) on the points
def best_line(points):
    best = Line((points[0], points[1]))
    best_rate =line_error(best, points)
    for p in itertools.combinations(points, 2):
        temp = Line(p,True)
        temp_rate = line_error(temp, points)
        if (temp_rate < best_rate):
            best = temp
            best_rate = temp_rate

    for p in itertools.combinations(points, 2):
        temp = Line(p,False)
        temp_rate = line_error(temp, points)
        if (temp_rate < best_rate):
            best = temp
            best_rate = temp_rate
    return best



def adaboost(points,rules=8):
    n=len(points)
    w=1/n
    for p in points:
        p.weight=w
    best_rules=[]
    weight_rules=[]  #weight=alpha_t
    alpha_t=0.5
    for i in range (0,rules) :
        sum = 0
        best_r=best_line(points)  #find the best rule
        best_rules.append(best_r)
        et=line_error(best_r,points)  #getting the error on the points
        num=(1-et)/et
        alpha_t=0.5*(np.log(num))  #np=numpy
        weight_rules.append(alpha_t)
        for p in points:
            if(best_r.is_right(p)==False):
                p.weight=p.weight*(np.math.e ** (alpha_t))
            else:
                p.weight=p.weight*(np.math.e ** (-alpha_t))
            sum+=p.weight
        for p in points: #normalized the point weight
            p.weight=p.weight/sum

    ans= H(best_rules,weight_rules,8)


    for p in points:
        p.weight=1

    return ans


def run_train(points, rules=8,times=100):
    #for i in range( 1,rules+1):
    avg1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    avg2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for j in range(times):
        learn = []
        test = []
        #filling the 2 list above with the points we get in random way
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

        for i in range (1,rules+1):
            rate = 0.0
            rate1=0.0

            for p in learn:
                if ans_learn.is_right_H(p,i):
                    rate += 1
            #print("avg :{}".format((rate / len(learn)) * 100))
            avg1[i] += (rate / len(learn)) * 100


            for p in test:
                if ans_learn.is_right_H(p,i):
                    rate1 += 1
            avg2[i]+=(rate1 / len(test) )* 100

    for i in range(1, 9):
        avg1[i] /= (times)
        print("Train: the rate of success for {} is {} percent ".format(i, avg1[i]))
        # shani add
        avg2[i] /= times
        print("Test: the rate of success for {} is {} percent ".format(i, avg2[i]))
        print("")



if __name__ == '__main__':

    print("HC_Body_Temperature :")

    f = open("HC_Body_Temperature.txt", "r")
    points = []
    for x in f:
        points.append(Point_for_HC(x))
    run_train(points, 8, 100)