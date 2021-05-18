import pandas as pd
import glob

idlist = []
file = "ids.txt"
file_n = open(file)
content = file_n.read()
file_n.close()
content = content.split('\n')
idlist.append(content[-2])
j = -1
for file in glob.glob("annotations.txt"):
    j += 1
    file_n = open(file)
    content = file_n.read()
    file_n.close()
    content = content.split('\n')
    result = dict()
    for line in content:
        line = line.split(',')
        if not line[0] == '':
            filename = line[0].strip("()\\\' ")
            annotation = line[1].strip("()\\\' ")
            if annotation != 'NoClick':
                result[int(filename[6:12])] = annotation
    final_labels = []
    count1 = 0
    label = '0'
    total = idlist[j][6:12]
    while count1 <= int(total):
        f_name = "SR-%s.jpg" % (idlist[j][3:6] + str(count1).zfill(6))
        temp = [f_name, label]
        count1 += 1
        final_labels.append(temp)
        if count1 in result.keys() and result[count1] == 'OneClick':
            label = '1'
        elif count1-1 in result.keys() and result[count1-1] == 'TwoClick':
            label = '0'
    #f_temp = file.lstrip("annores")
    #f_temp = f_temp[:-4]
    file_output = "label.csv"
    df = pd.DataFrame(final_labels)
    df.to_csv(file_output, index=False, header=False)
