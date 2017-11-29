import re
import os
filename = 'results.txt'

with open(filename, 'r') as f:
    lines = f.readlines()

print(str(len(lines)) + " lines")

# videos[videoName][classifierName][det0/fp1/fps2] = float
videos = {}

currentClassifier = {}
classifierNames = []
classifierIndex = 0

useBgRemoval = False

currentClassifierName = ''

def addTestSetData(data):
    for videoName in data.keys():
        classifierVideo = data[videoName]

        if videoName not in videos:
            videos[videoName] = {}
        #print(classifierVideo)
        #print(videoName + ' ' + classifierName)
        videos[videoName][classifierName] = classifierVideo;

for i in range(len(lines)):
    line = lines[i]
    if "Loaded classifier" in line:
        currentClassifierName = line.split('/')[1].split(' ')[0]
        continue
    elif "Set contains" in line:
        if len(currentClassifier):
            print('ADD ' + str(len(currentClassifier.keys())))
            # loop through videos for current classifier
            addTestSetData(currentClassifier)

            currentClassifier = {}
            classifierIndex = classifierIndex + 1
            if i == len(lines) - 1:
                break

        classifierName = currentClassifierName
        useBgRemoval = not useBgRemoval

        if useBgRemoval:
            classifierName += ' BGR'

        if classifierName not in classifierNames:
            classifierNames.append(classifierName)
        continue

    res = re.findall('\[(.*?)\]',line) 
        
    if res and len(res) > 0:
        videoName = line.split()[0]

        for i in range(len(res)):
            res[i] = re.search('[0-9]+[.][0-9]+', res[i]).group(0)

        currentClassifier[videoName] = res

# Add last set
addTestSetData(currentClassifier)

print(str(len(videos)) + ' videos')
#print(videos)
filenames = ['detectionrate.csv', 'fpositives.csv', 'fps.csv']

for i in range(len(filenames)):
    filename = filenames[i]

    try:
        os.remove(filename)
    except:
        pass

    with open(filename, 'a') as file:
        file.write('video;')

        for cn in classifierNames:
            file.write(cn + ';')

        file.write('\n')

        for videoName, classifiers in videos.items():
            file.write(videoName + ';')

           #print(len(classifiers))
            for classifier, result in classifiers.items():
                file.write(result[i] + ';')
            file.write('\n')
    
