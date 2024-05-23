# import libaries 
import csv
from PIL import Image 

# get csv data
groundtruthdata = []
with open('data.csv', 'r') as csv_datei:
    csv_reader = csv.reader(csv_datei)
    # skip header
    next(csv_reader)
    for zeile in csv_reader:
        groundtruthdata.append(zeile[:2])

#import images
filename = "images/Image1.jpg"
with Image.open(filename) as image:
    width, height = image.size

print(image.size)
