import numpy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from file_utils import read_from_csv, read_from_csv2
from sklearn.neural_network import MLPRegressor


def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]
        scaler.fit(trainData)
        scaledTrainData = scaler.transform(trainData)
        scaledTestData = scaler.transform(testData)
        scaledTrainData = [el[0] for el in scaledTrainData]
        scaledTestData = [el[0] for el in scaledTestData]
    else:
        scaler.fit(trainData)
        scaledTrainData = scaler.transform(trainData)
        scaledTestData = scaler.transform(testData)
    return scaledTrainData, scaledTestData


# ==============================================================================================================================
# Sa se stabileasca categoria (femei sau barbati) cu cele mai multe tricouri vandute deja

temp, female, male = read_from_csv('tshirts.csv')

nr_female = 0
nr_male = 0

for i in range(len(temp)):
    nr_female += female[i]
    nr_male += male[i]

if nr_female > nr_male:
    print(' a) Cele mai multe tricouri au fost cumparate de femei')
elif nr_male > nr_female:
    print(' a) Cele mai multe tricouri au fost cumparate de barbati')
else:
    print(' a) egalitate')

# ================================================================================================================================
# Sa se estimeze cate tricouri de barbati trebuie sa comande pentru urmatoarea luna.
trainInputs = numpy.array([temp[i] for i in range(len(temp))]).reshape(-1, 1)
trainOutputs = numpy.array([male[i] for i in range(len(temp))]).reshape(-1, 1)

regr = LinearRegression(normalize=True)

regr.fit(trainInputs, trainOutputs)

testInput = numpy.array([26, 26, 25, 25, 27, 27, 27, 27, 24, 25, 23, 27, 27, 22, 24, 27,
                         24, 23, 25, 27, 26, 22, 24, 24, 24, 25, 24, 25, 23, 25]).reshape(-1, 1)

predicted = regr.predict(testInput)
if sum(predicted) % 1 == 0:
    print(sum(predicted))
else:
    print(' b) ' + str(int((sum(predicted) - sum(predicted) % 1 + 1)[0])) + ' tricouri de cumparat pt barbati')

# ================================================================================================================================
# Administratorul mai primeste inca 2 informatii suplimentare (pentru fiecare din zilele ultimelor 5
# luni) despre concursurile sportive desfasurate in fiecare zi (a se vedea fisierul tshirtsNew.csv). Ce impact
# pot avea aceste informatii asupra numarului de tricouri pe care trebuie sa le comande pentru o noua zi cand
# vor fi 25 de grade si vor avea loc multe concursuri sportive in cadrul liceului din orasul SMART?

temp, female, male, competitions, location = read_from_csv2('tshirtsNew.csv')

for i in range(len(competitions)):
    if competitions[i] == 'veryFew':
        competitions[i] = 1
    elif competitions[i] == 'few':
        competitions[i] = 2
    elif competitions[i] == 'medium':
        competitions[i] = 5
    elif competitions[i] == 'many':
        competitions[i] = 10
    else:
        competitions[i] = 15

for i in range(len(location)):
    if location[i] == 'park':
        location[i] = 1
    elif location[i] == 'high-school':
        location[i] = 2
    else:
        location[i] = 3

trainInputs = [[temp[i], competitions[i], location[i]] for i in range(len(temp))]
trainOutputs = [[female[i], male[i]] for i in range(len(temp))]
testInputs = [[25, 10, 2]]
normalizedTrainInputs, normalizedTestInputs = normalisation(trainInputs, testInputs)

# varianta multi-layer perceptron
# solvers : lbfgs -> an optimizer in the family of quasi-Newton methods
#           adam  -> a stochastic gradient-based optimizer  (DEFAULT)
#           sgd   -> stochastic gradient descent
# regressor = MLPRegressor(hidden_layer_sizes=(30, 50, 20, 60), solver='lbfgs', max_iter=5000)

regressor = MLPRegressor(hidden_layer_sizes=(50, 50, 50), solver='adam', max_iter=4000, verbose=10, n_iter_no_change=10,
                         warm_start=True, learning_rate='adaptive')
regressor.fit(normalizedTrainInputs, trainOutputs)
print(' c) loss : ' + str(regressor.loss_))

'''
# varianta random forest regressor
regressor = RandomForestRegressor(max_depth=1000, random_state=0)
regressor.fit(normalizedTrainInputs, trainOutputs)
'''

predicted = regressor.predict(normalizedTestInputs)
print('    predicted female : ' + str(predicted[0][0]))
print('    predicted male : ' + str(predicted[0][1]))
