import doctest
from Line_HC import Line
from Point_HC import Point_for_HC

class H():

    def __init__(self,best_rules,weight_rules, rules=8 ):
        self.best_rules=best_rules
        self.weight_rules=weight_rules
        self.rules=rules


    def sign(self, p,k):
        sum = 0
        for i in range(k):
            if (self.best_rules[i].up):

                if (self.best_rules[i].is_low(p)):
                    sum -= self.weight_rules[i]
                else:
                    sum += self.weight_rules[i]
            else:
                if (self.best_rules[i].is_low(p)):
                    sum += self.weight_rules[i]
                else:
                    sum -= self.weight_rules[i]

        if (sum >= 0):
            return 1
        else:
            return -1


    def is_right_H(self,p,k):
        if(self.sign(p,k)==p.gender):
            return True
        else:
            return False
