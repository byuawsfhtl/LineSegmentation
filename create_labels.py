import os


files = os.listdir('./images')

label_file = open('labels.csv', 'w')

for f in files:
    line = 'images/' + f + '\tbaselines/' + f + '\n'
    label_file.write(line)

label_file.close()
