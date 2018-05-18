import numpy as np
#
#    AT  |  AF
# ET 0,0 | 0,1
# -------+-------
# EF 1,0 | 1,1
#        |
#
#     AT  |  AF
# ET  TP  |  FN
# --------+-------
# EF  FP  |  TN
#         |
def confusion_matrix(target,actual,expected):
   arr = [[0,0],[0,0]]
   for i in range(len(actual)):
      y = 0 if actual[i] == target else 1
      x = 0 if expected[i] == target else 1
      arr[x][y] += 1
   return arr

# TP/(TP+FP)
def precision(cmatrix):
   TP = cmatrix[0][0]
   FP = cmatrix[1][0]
   return float(TP)/(TP+FP)

#TP/(TP+FN)
def recall(cmatrix):
   TP = cmatrix[0][0]
   FN = cmatrix[0][1]
   return float(TP)/(TP+FN)

def fmeasure(cmatrix, beta=None):
   if beta == None:
      beta = 1
   p = precision(cmatrix)
   r = recall(cmatrix)
   return (1+beta**2)*((p*r)/((beta**2)*p+r))
