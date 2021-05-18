import shutil
import os
import csv

file = "ids.txt"
file_n = open(file)
content = file_n.read()
file_n.close()
content = content.split('\n')
total = int(content[-2][6:12])
trainLimit = int(total*70/100)
valLimit = int(total*90/100)
file ="annotations.txt"
file_n = open(file)
content = file_n.read()
file_n.close()
content = content.split('\n')
result = []
i = 0
for line in content:
    if line is not '':
        line = line.split(',')
        filename = line[0].strip("()\\\' ")
        annotation = line[1].strip("()\\\' ")
        if annotation != 'NoClick':
            result.append(int(filename[6:12]))

csv_file='label.csv'
folderPath = 'imagesDirectory'
trainTD = '/home/nutfruit/Documents/Fathima599_Fall20/AllVideos/Minh-images/train/TouchDetected'
trainND = '/home/nutfruit/Documents/Fathima599_Fall20/AllVideos/Minh-images/train/NotDetected'
testTD = '/home/nutfruit/Documents/Fathima599_Fall20/AllVideos/Minh-images/test/TouchDetected'
testND = '/home/nutfruit/Documents/Fathima599_Fall20/AllVideos/Minh-images/test/NotDetected'
valTD = '/home/nutfruit/Documents/Fathima599_Fall20/AllVideos/Minh-images/validation/TouchDetected'
valND = '/home/nutfruit/Documents/Fathima599_Fall20/AllVideos/Minh-images/validation/NotDetected'
with open(csv_file, mode='r') as infile:
    reader = csv.reader(infile)
    with open('coors_new.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        key_values = {rows[0]: rows[1] for rows in reader}
filesToFind = key_values.keys()
p = -1
for file in filesToFind:
    nme = int(file[6:12])
    p = p+1
    if p%5 != 0:
        continue
    else:
        name = os.path.join(folderPath, file)
        if key_values[file] == '0':
            if not any(x in result for x in range( nme-10,  nme+10)):
                if p<= trainLimit:
                    shutil.copy(name, trainND)
                elif p<= valLimit:
                    shutil.copy(name, valND)
                else:
                    shutil.copy(name, testND)
        else:
            if not any(x in result for x in range( nme-10,  nme+10)):
                if p<= trainLimit:
                    shutil.copy(name, trainTD)
                elif p<= valLimit:
                    shutil.copy(name, valTD)
                else:
                    shutil.copy(name, testTD)

 
