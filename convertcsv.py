from os import listdir
import os
from os.path import isfile, join
import re
import csv

print(os.getcwd())
datapath = 'data/EURUSD'
onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
for file in onlyfiles:
    print(file)
    newFile = re.sub(r"BID_[0-9][0-9].[0-9][0-9].", "", file)
    newFile = datapath + "/" + re.sub(r"-[0-9][0-9].[0-9][0-9].20[0-9][0-9]", "", newFile)
    print(newFile)
    with open(os.getcwd() + "/" + datapath + "/" + file,'r') as csvfile,open(newFile,'a') as newcsvfile:#, NamedTemporaryFile(dir="/home/vincent/PycharmProjects/trainstrategy/forexmine/data/hourly/",delete=False) as temp:
        r = csv.reader(csvfile)
        index = 0
        for row in r:
            if not index == 0:
                dt = row[0].split(",")
                print("{}{}{} {}{}{};{};{};{};{};{}".format(row[0][6:10],row[0][3:5],row[0][0:2],row[0][11:13],row[0][14:16],row[0][17:19],row[1],row[2],row[3],row[4],row[5]))
                newcsvfile.write("{}{}{} {}{}{};{};{};{};{};{}\n".format(row[0][6:10],row[0][3:5],row[0][0:2],row[0][11:13],row[0][14:16],row[0][17:19],row[1],row[2],row[3],row[4],row[5]))
            index += 1
