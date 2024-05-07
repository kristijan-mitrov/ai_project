# import libaries 
import csv

# get csv data
groundtruthdata = []
with open('data.csv', 'r') as csv_datei:
    csv_reader = csv.reader(csv_datei)
    # skip header
    next(csv_reader)
    for zeile in csv_reader:
        groundtruthdata.append(zeile[:2])