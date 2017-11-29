import re
filename = 'results.txt'
pattern = re.compile('a')

with open(filename, 'r') as f:
    lines = f.readlines()

print(str(len(lines)) + " lines")
detArr = []
fpArr = []
fpsArr = []

# [video][classifier][det/fp/fps] = float
videos = []

currentClassifier = []
classifierIndex = 0

for line in lines:
    if "Set contains" in line:
        if(len(currentClassifier)):
            #print(currentClassifier)
            print('len(currentClassifier): ' + str(len(currentClassifier)))
            for i in range (0, len(currentClassifier)):
                if(len(videos) - 1 < i):
                    videos.append([])
                video = currentClassifier[i]
                print(str(i) + ' ' + str(classifierIndex))
                
                if len(videos[i]) - 1 < classifierIndex:
                    videos[i].append([])
                videos[i][classifierIndex].append(video)
      
        currentClassifier = []
        classifierIndex = classifierIndex + 1
        print('Classifier ' + str(classifierIndex))
        continue
    res = re.findall('\[(.*?)\]',line) 
        
    if res:
        currentClassifier.append(res)

print(str(len(videos)) + ' videos')
print(str(len(videos[0])) + ' classifiers')
filenames = ['detectionrate.csv', 'fpositives.csv', 'fps.csv']
for i in range(len(filenames)):
    with open(filenames[i], 'a') as file:
        for j in range(len(videos)):
            print(videos[j])
            for k in range(len(videos[j])):
                #print(videos[j][k][i])
                file.write(videos[j][k][i] + ';')
            file.write('\n')
    
