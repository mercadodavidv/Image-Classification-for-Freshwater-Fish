###
# Outputs a .csv "import file" for use with the Vertex AI datasets service for data labeling
###

from os import listdir
from os.path import isfile, join

result = ''

for dir in [d for d in listdir('./') if not isfile(join('./', d))]:
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    print(dir + ' ' + str(len(files)))
    for file in files:
        result += 'gs://fish-cv-classification/' + dir + '/' + file + ',' + dir + '\n'

with open('image_classification_single_label.csv', 'w') as f:
    f.write(result)
