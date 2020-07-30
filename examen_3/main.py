from sklearn.preprocessing import StandardScaler
from file_utils import read_from_csv
from sklearn import linear_model
from log_regression import MyLogisticRegression


def normalisation(trainData, testData, sample):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]
        sample = [[d] for d in sample]

        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
        normalisedSample = scaler.transform(sample)

        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
        normalisedSample = [el[0] for el in normalisedSample]
    else:
        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
        normalisedSample = scaler.transform(sample)
    return normalisedTrainData, normalisedTestData, normalisedSample


# ============================================================================================================================
# Sa se identifice categoria (barbati sau femei) careia ii corespunde o greutate medie mai mare.

age, height, weight, sex = read_from_csv('personsNew.csv')

avg_male = 0
avg_female = 0
female_contor = 0
male_contor = 0

for i in range(150):
    if sex[i] == 'female':
        avg_female += weight[i]
        female_contor += 1
    else:
        avg_male += weight[i]
        male_contor += 1

print('average male weight : ' + str(avg_male / male_contor))
print('average female weight : ' + str(avg_female / female_contor))

# ============================================================================================================================
# Sa se determine sexul unei persoane care are 47 ani, 65 kilograme si 190 cm inaltime folosind un
# model de clasificare antrenat pe primele 100 de exemple din setul de date din „persons.csv”.

# Ce sex ar prezice pentru aceeasi persoana un model de clasificare antrenat similar modelului de la
# punctul b, dar folosind setul de date „personsNew.csv”? Este mai performant acest model decat cel antrenat
# la punctul b?

trainInputs = [[age[i], height[i], weight[i]] for i in range(100)]
trainOutputs = [0 if sex[i] == 'female' else 1 for i in range(100)]

testInputs = [[age[i], height[i], weight[i]] for i in range(100, 150)]
testOutputs = [0 if sex[i] == 'female' else 1 for i in range(100, 150)]

sample = [[47, 190, 65]]
trainInputs, testInputs, sample = normalisation(trainInputs, testInputs, sample)

'''
classifier = linear_model.LogisticRegression()
classifier.fit(trainInputs, trainOutputs)
computedTestOutputs = classifier.predict([[47,190,65]])
print(['male' if computedTestOutputs[0] == 1 else 'female'][0])

accuracy = 0
for i in range(len(testInputs)):
    if testOutputs[i] == 0:
        realLabel = 'female'
    else:
        realLabel = 'male'
    computedLabel = classifier.predict([testInputs[i]])
    computedLabel = ['male' if computedLabel == 1 else 'female'][0]
    print('real :' + realLabel + '  predicted : ' + computedLabel)
    if realLabel == computedLabel:
        accuracy += 1

print('accuracy : ' + str(accuracy / len(testInputs)))
'''

classifier = MyLogisticRegression()
classifier.fit(trainInputs, trainOutputs)
computedTestOutputs = classifier.predictOneSample(sample[0])
print(computedTestOutputs)

accuracy = 0

for i in range(len(testInputs)):
    if testOutputs[i] == 0:
        realLabel = 'female'
    else:
        realLabel = 'male'
    computedLabel = classifier.predictOneSample(testInputs[i])
    print('real :' + realLabel + '  predicted : ' + computedLabel)
    if realLabel == computedLabel:
        accuracy += 1

print('accuracy : ' + str(accuracy / len(testInputs)))
