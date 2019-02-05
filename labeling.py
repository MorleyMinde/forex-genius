from os import listdir
import os
from os.path import isfile, join
import re
import csv

# Time;Open;High;Low;Close;Volume;Action
# Time,Open,High,Low,Close,Volume,Action
datapath = 'data/EURUSD/raw'
def getIndexData(index,file):
    retRow = None
    with open(os.getcwd() + "/" + datapath + "/" + file,'r') as csvfile:
        reader = csv.reader(csvfile)
        first = True
        for i, row in enumerate(reader):
            if first:
                first = False
                continue
            if i - 1 == index:
                # print("{},{},{}".format(i,index,row))
                retRow = row
                break
        # print("The row:{}".format(retRow[4]))
    return retRow[4]
print(os.getcwd())
onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
for file in onlyfiles:
    print(file)
    newFile = re.sub(r"BID_[0-9][0-9].[0-9][0-9].", "", file)
    newFile = datapath + "/" + re.sub(r"-[0-9][0-9].[0-9][0-9].20[0-9][0-9]", "labeled", newFile)
    print(newFile)
    with open(os.getcwd() + "/" + datapath + "/" + file,'r') as csvfile,open(newFile,'a') as newcsvfile:#, NamedTemporaryFile(dir="/home/vincent/PycharmProjects/trainstrategy/forexmine/data/hourly/",delete=False) as temp:
        r = csv.reader(csvfile)
        counterRow = 0
        windowSize = 11
        numberOfMinutes = sum(1 for row in r)
        # print("numberOfMinutes:{}".format(numberOfMinutes))
        csvfile.seek(0)
        r = csv.reader(csvfile)
        first = True
        for row in r:
            if first:
                first = False
                continue
            result = 0
            if counterRow > windowSize:
                windowBeginIndex = counterRow - windowSize
                windowEndIndex = windowBeginIndex + windowSize - 1
                windowMiddleIndex = int((windowBeginIndex +windowEndIndex)/2)
                minimum = None
                maximum = None
                maxIndex = None
                minIndex = None
                # print("{}:{}, {}".format(counterRow,windowBeginIndex, windowEndIndex))
                for i in range(windowBeginIndex, windowEndIndex):
                    number = float(getIndexData(i,file))
                    if minimum == None:
                        minimum =number
                        maximum =number
                        maxIndex = i
                        minIndex = i
                    if number < minimum:
                        minimum =number
                        minIndex = i
                    if number > maximum:
                        maximum =number
                        maxIndex = i
                if(maxIndex == windowMiddleIndex):
                    result = 2
                elif (minIndex == windowMiddleIndex):
                    result = 1
                else:
                    result = 0
                # print("{}:{}, {}".format(maxIndex,minIndex, windowMiddleIndex,result))
            newcsvfile.write("{}{}{} {}{}{},{},{},{},{},{},{}\n".format(row[0][6:10],row[0][3:5],row[0][0:2],row[0][11:13],row[0][14:16],row[0][17:19],row[1],row[2],row[3],row[4],row[5],result))
            counterRow += 1
    break