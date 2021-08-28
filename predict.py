import numpy as np
from numpy.core.fromnumeric import clip
from numpy.core.function_base import linspace
from numpy.lib import index_tricks
import pandas
import math
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy


def IsSideWayUp(clipp, features):
    delta = 0.2 * (clipp[-1][0] - clipp[0][0])
    point = -1
    
    for i in range(2, len(clipp)):
        clippp = clipp[:i+1]
        line1 = [((clipp[i][0]+delta)-(clipp[0][0]+delta)), (features.index(clipp[0])-features.index(clipp[i]))] 
        line1.append(-1*((line1[0]*features.index(clipp[0])) + (line1[1]*(clipp[0][0]+delta))))
        line2 = [((clipp[i][0]-delta)-(clipp[0][0]-delta)), (features.index(clipp[0])-features.index(clipp[i]))]
        line2.append(-1*((line1[0]*features.index(clipp[0])) + (line1[1]*(clipp[0][0]-delta))))
        for day in clippp:
            y1 = ((features.index(day) * line1[0])+line1[2]) / (-1*line1[1])
            y2 = ((features.index(day) * line2[0])+line2[2]) / (-1*line2[1])
            if not ((day[0] < y1 and day[0] > y2)):
                point = i - 1
                break
        
        if point == i-1: break  
        point = i

    lines = [line1, line2]
    line3 = [((clipp[-1][0]+delta)-(clipp[point][0]+delta)), (features.index(clipp[point])-features.index(clipp[-1]))]  
    line3.append(-1*((line3[0]*features.index(clipp[point])) + (line3[1]*(clipp[point][0]+delta))))

    lines.append(line3)

    for i in range(point+1, len(clipp)):
        clippp = clipp[point+1:]
        for day in clippp:
            y1 = ((features.index(day) * line3[0])+line3[2]) / (-1*line3[1])
            if day[0] < y1:
                return False
    
    return True

def IsSideWayDown(clipp, features):
    delta = 0.2 * (clipp[-1][0] - clipp[0][0])
    point = -1
    
    for i in range(2, len(clipp)):
        clippp = clipp[:i+1]
        line1 = [((clipp[i][0]+delta)-(clipp[0][0]+delta)), (features.index(clipp[0])-features.index(clipp[i]))] 
        line1.append(-1*((line1[0]*features.index(clipp[0])) + (line1[1]*(clipp[0][0]+delta))))
        line2 = [((clipp[i][0]-delta)-(clipp[0][0]-delta)), (features.index(clipp[0])-features.index(clipp[i]))]
        line2.append(-1*((line1[0]*features.index(clipp[0])) + (line1[1]*(clipp[0][0]-delta))))
        for day in clippp:
            y1 = ((features.index(day) * line1[0])+line1[2]) / (-1*line1[1])
            y2 = ((features.index(day) * line2[0])+line2[2]) / (-1*line2[1])
            if not ((day[0] > y1 and day[0] < y2)):
                point = i - 1
                break
        
        if point == i-1: break 
        point = i 

    lines = [line1, line2]
    line3 = [((clipp[-1][0]+delta)-(clipp[point][0]+delta)), (features.index(clipp[point])-features.index(clipp[-1]))]  
    line3.append(-1*((line3[0]*features.index(clipp[point])) + (line3[1]*(clipp[point][0]+delta))))

    lines.append(line3)

    for i in range(point+1, len(clipp)):
        clippp = clipp[point+1:]
        for day in clippp:
            y1 = ((features.index(day) * line3[0])+line3[2]) / (-1*line3[1])
            if day[0] > y1:
                return False
    
    return True

#--------------------------read data from file--------------------------
data = pandas.read_csv('ADANIPORTS.csv')
close, open, high, low, volume = data['Close'], data['Open'], data['High'], data['Low'], data['Volume']

#--------------------------add features to list--------------------------
features = []
for i in range(len(close)):
    features.append([close[i], open[i], high[i], low[i], volume[i]])

#--------------------------create clips with len=18--------------------------
clips = []
for i in range(len(features)):
    if i+18 <= len(features):
        clips.append(features[i:i+18])

#--------------------------compute each clip belong to which class--------------------------
classs, class0, class1, class2, class3 = [], [], [], [], []
# all     UP    unknown  flat    down

for clipp in clips: # class of each clip
    delta = 0.2 * (clipp[-1][0] - clipp[0][0])
    # a,b,c of lines save in list: line1=[a,b,c]
    # up and doxw lines of each clip created
    line1 = [((clipp[-1][0]+delta)-(clipp[0][0]+delta)), (features.index(clipp[0])-features.index(clipp[-1]))] 
    line1.append(-1*((line1[0]*features.index(clipp[0])) + (line1[1]*(clipp[0][0]+delta))))
    line2 = [((clipp[-1][0]-delta)-(clipp[0][0]-delta)), (features.index(clipp[0])-features.index(clipp[-1]))]
    line2.append(-1*((line1[0]*features.index(clipp[0])) + (line1[1]*(clipp[0][0]-delta))))

    flag = True # if true => class is continuous
    for day in clipp:
        # check whether each point is in clip ABCD or not
        y1 = ((features.index(day) * line1[0])+line1[2]) / (-1*line1[1])
        y2 = ((features.index(day) * line2[0])+line2[2]) / (-1*line2[1])
        if not ((day[0] < y1 and day[0] > y2) or (day[0] < y2 and day[0] > y1)):
            flag = False
            break

    sideUp = IsSideWayUp(clipp, features) # if true => class is sideway Up
    sideDown = IsSideWayDown(clipp, features) # if true => class is sideway Down


    #check conditions to set classes
    if flag and clipp[-1][0] > clipp[0][0]: 
        classs.append('Continuous Up') 
        class0.append(clips.index(clipp))
    elif flag and clipp[-1][0] < clipp[0][0]: 
        classs.append('Continuous Down')  
        class3.append(clips.index(clipp))    
    elif not flag and sideUp:
        classs.append('Sideway Up')
        class0.append(clips.index(clipp)) 
    elif not flag and sideDown:
        classs.append('Sideway Down')
        class3.append(clips.index(clipp))     
    elif not flag and not sideDown and not sideUp: 
        classs.append('Unknown')
        class1.append(clips.index(clipp)) 
    else: 
        classs.append('Flat')
        class2.append(clips.index(clipp))

#--------------------------find probability to occurrence each class--------------------------
P_m = []
P_m.append(len(class0) / len(classs))
P_m.append(len(class3) / len(classs))
P_m.append(len(class1) / len(classs))
P_m.append(len(class2) / len(classs))

#--------------------------calculate Entropy--------------------------
tmp = 0
for i in range(len(P_m)):
    if P_m[i] != 0:
        tmp += P_m[i]*math.log2(P_m[i])
Entropy = -1*tmp

#--------------------------calculate D_i--------------------------
D_i = []
tmp = []
for i in range(len(class0)): # for class Up
    sum = 0
    
    xj = clips[class0[i]][0]
    clipp = clips[class0[i]]
    for j in range(len(clipp)):
        sum += math.dist([features.index(clipp[j]) , clipp[j][0]], [features.index(xj), xj[0]])
    tmp.append(1/sum)   
if len(tmp) != 0: 
    tmp, class0 = zip(*sorted(zip(tmp, class0), reverse=True))
    tmp , class0 = list(tmp), list(class0)
D_i.append(tmp)     

tmp = []
for i in range(len(class1)): # for class Unknown
    sum = 0
    
    xj = clips[class1[i]][0]
    clipp = clips[class1[i]]
    for j in range(len(clipp)):
        sum += math.dist([features.index(clipp[j]) , clipp[j][0]], [features.index(xj), xj[0]])
    tmp.append(1/sum)  
if len(tmp) != 0:
    tmp, class1 = zip(*sorted(zip(tmp, class1), reverse=True))
    tmp , class1 = list(tmp), list(class1)  
D_i.append(tmp)  

tmp = []
for i in range(len(class2)): # for class Flat
    sum = 0
    
    xj = clips[class2[i]][0]
    clipp = clips[class2[i]]
    for j in range(len(clipp)):
        sum += math.dist([features.index(clipp[j]) , clipp[j][0]], [features.index(xj), xj[0]])
    tmp.append(1/sum)
if len(tmp) != 0:
    tmp, class2 = zip(*sorted(zip(tmp, class2), reverse=True))
    tmp , class2 = list(tmp), list(class2)   
D_i.append(tmp)  

tmp = []
for i in range(len(class3)): # for class Down
    sum = 0
    
    xj = clips[class3[i]][0]
    clipp = clips[class3[i]]
    for j in range(len(clipp)):
        sum += math.dist([features.index(clipp[j]) , clipp[j][0]], [features.index(xj), xj[0]])
    tmp.append(1/sum)
if len(tmp) != 0:
    tmp, class3 = zip(*sorted(zip(tmp, class3), reverse=True))
    tmp , class3 = list(tmp), list(class3)   
D_i.append(tmp) 

#--------------------------calculate Gain--------------------------
n = min(len(class1), len(class0), len(class3))
res = close[1:].tolist()
closeTmp = deepcopy(close[:-1])
closeTmp = closeTmp.tolist()


tmpClip = []
for i in range(n):
    tmpClip.append(clips[class0[i]])
    tmpClip.append(clips[class1[i]])
    tmpClip.append(clips[class3[i]])  


# gain = [] #for close feature

# for i in range(len(closeTmp)):
#     coefficient = closeTmp.count(closeTmp[i]) / (len(closeTmp))
#     tmp = 0
#     for j in range(closeTmp.count(closeTmp[i])):
#         try:
#             t1 = res.count(closeTmp[i])
#             t2 = closeTmp.count(closeTmp[i])
#             coef = t1 / t2
#             tmp += coef * math.log2(coef)
#         except:
#             tmp += 0
#     tmp *= -1
#     gain.append(coefficient * tmp)      


# gain = []

# for i in range(len(clips)):
holdClass = []
for clas in classs:
    if 'Up' in clas:
        holdClass.append(1.0)
    elif 'Down' in clas:
        holdClass.append(2.0)
    elif 'Unknown' in clas:
        holdClass.append(3.0)
    else:
        holdClass.append(4.0)     

holdClass2 = []
for i in range(n):
    holdClass2.append(holdClass[class0[i]])
    holdClass2.append(holdClass[class1[i]])
    holdClass2.append(holdClass[class3[i]])


for clippp in clips:
    for i in range(len(clippp)):
        clippp[i] = clippp[i][0]
holdClass = np.array(holdClass, dtype=np.float)

# B = 10
# randF = RandomForestRegressor(n_estimators=B)

# tree = randF.fit(clips[:3000], holdClass[:3000])
# pred = tree.predict(clips[3000:])
# print(metrics.r2_score(holdClass[3000:], pred))
# sm = difflib.SequenceMatcher(None, pred, holdClass[3000:])
# print(sm.ratio())

# av = 0
# for i in range(len(pred)):
#     if pred[i] == holdClass[3000+i]:
#         av += 1
# print(av/len(pred))
# print('num of matches',av)
# print('num of all test list', len(pred))
print('==============')

B = 10
randF = RandomForestRegressor(n_estimators=B)

tree = randF.fit(tmpClip[:550], holdClass2[:550])
pred = tree.predict(tmpClip[550:])
print(metrics.r2_score(holdClass2[550:], pred))

av = 0
for i in range(len(pred)):
    if pred[i] == holdClass2[550+i]:
        av += 1
print(av/len(pred))
print('num of matches',av)
print('num of all test list', len(pred))