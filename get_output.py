import csv
import time
import os

def get_mean(data):
    sum = 0
    for i in data:
        sum = sum + int(i)
    return sum / len(data)

#clear the file
# with open('/store/avoid_result/reponse.log', 'r+') as empty:
with open('/home/LAB/xuesq/xuesq/response.log', 'r+') as empty:
    empty.seek(0)
    empty.truncate()
    empty.close()

while True:
    time.sleep(10)
    clean_data1 = []
    clean_data2 = []
    clean_data3 = []
    if os.path.exists('/home/LAB/xuesq/xuesq/olc_submit/zxqy-olc.csv'):
        with open('/home/LAB/xuesq/xuesq/olc_submit/zxqy-olc.csv') as csv_file:
            read_csv = csv.reader(csv_file, delimiter=',')
            for row in read_csv:
                if len(row) > 2:
                    if 'chrome' not in row[2]:
                        if row[2].find('/'):
                            if 'e' not in row[1]:
                                clean_data1.append(row[1])
                else:
                    break
    if os.path.exists('/home/LAB/xuesq/xuesq/forecast_submit/2-forecast.csv'):
        with open('/home/LAB/xuesq/xuesq/forecast_submit/2-forecast.csv') as csv_file:
            read_csv = csv.reader(csv_file, delimiter=',')
            for row in read_csv:
                if len(row) > 2:
                    if 'chrome' not in row[2]:
                        if row[2].find('/'):
                            if 'e' not in row[1]:
                                clean_data2.append(row[1])
                else:
                    break


    if os.path.exists('/home/LAB/xuesq/xuesq/normal_submit/2-normal.csv'):
        with open('/home/LAB/xuesq/xuesq/normal_submit/2-normal.csv') as csv_file:
            read_csv = csv.reader(csv_file, delimiter=',')
            for row in read_csv:
                if len(row) > 2:
                    if 'chrome' not in row[2]:
                        if row[2].find('/'):
                            if 'e' not in row[1]:
                                clean_data3.append(row[1])
                else:
                    break
    if len(clean_data1) == 0:
        olc = 0
    else:
        olc = get_mean(clean_data1)
    if len(clean_data2) == 0:
        mp = 0
    else:
        mp = get_mean(clean_data2)
    if len(clean_data3) == 0:
        nmp = 0
    else:
        nmp = get_mean(clean_data3)
    print('OLC = ' + str(olc) + ' MP = ' + str(mp) + ' NMP = ' + str(nmp))
    # with open('/store/avoid_result/reponse.log', 'a') as reponse:
    with open('/home/LAB/xuesq/xuesq/response.log', 'a') as response:
        print('&OLC=' + str(olc) + '&MP=' + str(mp) + '&NMP=' + str(nmp) + '&', file=response)
