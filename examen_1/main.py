from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from file_utils import loadData2, loadData
from normalization_stuff import normalization, normalize_one

# ============================================================================================================================
# Stabiliti pretul mediu al imobilelor din locatia (identificata prin zipcode) in care se afla amplasat imobilul cu cel mai mare pret.
# pt prima cerinta ne intereseaza : pret, zipcode

prices, zipcodes = loadData('homeData.csv', 'price', 'zipcode')

# aflam locatia imobilului cu cel mai mare pret
maxim = max(prices)
maxim_index = [i for i in range(len(prices)) if prices[i] == maxim]
locatie_maxim = zipcodes[maxim_index[0]]

medie = 0
k = 0

# calculam media preturilor imobilelor care au zipcode-ul locatie_maxim
for i in range(len(prices)):
    if zipcodes[i] == locatie_maxim:
        medie += prices[i]
        k += 1

print("media este " + str(medie / k) + "\n")
# ============================================================================================================================

# Sa se - stabileasca eroare medie patratica (mean squared error) a unui model de predictie a pretului
# imobilelor care foloseste 5 atribute (['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors’]), baza de
# antrenare fiind formata din primele 100 de exemple, iar cea de testare din urmatoarele 50 de exemple din
# fisierul homeData.csv.

bedrooms, bathrooms, sqft_living, sqft_lot, floors, prices = loadData2('homeData.csv')
all_data = [[bedrooms[i], bathrooms[i], sqft_living[i], sqft_lot[i], floors[i]] for i in range(150)]

una = [3, 2.5, 1910, 66211, 2]
una = normalize_one(all_data, una)
una = [una]

bedrooms = normalization(bedrooms)
bathrooms = normalization(bathrooms)
sqft_living = normalization(sqft_living)
sqft_lot = normalization(sqft_lot)
floors = normalization(floors)

trainSample = [i for i in range(100)]
testSample = [i for i in range(100, 150)]

# split data
trainInputs = [[bedrooms[i], bathrooms[i], sqft_living[i], sqft_lot[i], floors[i]] for i in trainSample]
trainOutputs = [prices[i] for i in trainSample]
testInputs = [[bedrooms[i], bathrooms[i], sqft_living[i], sqft_lot[i], floors[i]] for i in testSample]
testOutputs = [prices[i] for i in testSample]


regr = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, alpha=3, learning_rate='adaptive', verbose=1, n_iter_no_change=1000))
regr.fit(trainInputs, trainOutputs)

error = 0

pred = regr.predict(testInputs)
for i in range(len(testInputs)):
    print('real : ' + str(testOutputs[i]) + '  predicted : ' + str(pred[i]) + '  eroare : ' + str(
        testOutputs[i] - pred[i]))
    error = error + (testOutputs[i] - pred[i]) ** 2

print("prediction error : {}".format(error / len(testInputs)))

print('Pretul prezis : ' + str(regr.predict(una)[0]))
