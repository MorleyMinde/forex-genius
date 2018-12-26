import csv

file = '/home/vincent/PycharmProjects/trainstrategy/forexmine/data/hourly/EURUSD_Candlestick_1_Hour_BID_29.01.2018-30.01.2018.csv'
newFile = '/home/vincent/PycharmProjects/trainstrategy/forexmine/data/hourly/EURUSD_Candlestick_1_Hour_BID_29.01.2018-30.01.2018.csv'
with open(file,'r') as csvfile,open(newFile,'a') as newcsvfile:#, NamedTemporaryFile(dir="/home/vincent/PycharmProjects/trainstrategy/forexmine/data/hourly/",delete=False) as temp:
    r = csv.reader(csvfile)
    index = 0
    for row in r:
        if not index == 0:
            dt = row[0].split(",")
            print("{}{}{} {}{}{};{};{};{};{};{}".format(row[0][6:10],row[0][3:5],row[0][0:2],row[0][11:13],row[0][14:16],row[0][17:19],row[1],row[2],row[3],row[4],row[5]))
            newcsvfile.write("{}{}{} {}{}{};{};{};{};{};{}\n".format(row[0][6:10],row[0][3:5],row[0][0:2],row[0][11:13],row[0][14:16],row[0][17:19],row[1],row[2],row[3],row[4],row[5]))
        index += 1
        #w.writerow(row)
#move(temp.name,"Filename.csv")