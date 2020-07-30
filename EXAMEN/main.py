from file_utils import read_from_csv
from knn import kmeans

# cerinta 1

id, ultima_com, information, averageCommunicLength, clientSpecialisation = read_from_csv('server.csv')

rezultat = []

for i in range(100):
    if clientSpecialisation[i] == 'simple' and information[i] == 'high':
        rezultat.append(id[i])
print('ID-urile clientilor sunt :' + str(rezultat))


# cerinta 2
trainInputs = [[id[i], ultima_com[i], information[i], averageCommunicLength[i], clientSpecialisation[i]] for i in range(100)]
k = kmeans(2)
k.fit(trainInputs)

media = [0,0]

for each in range(len(k.clusters)):
    print('\nCLUSTER ' + str(each) + ' : ')
    for one in k.clusters[each]:
        print(one)
        media[each] += one[3]

print('Durata medie a primului cluster : ' + str(media[0] / len(k.clusters[0])))
print('Durata medie a celui de-al doilea cluster : ' + str(media[1] / len(k.clusters[1])))

# cerinta 3
id, ultima_com, information, averageCommunicLength, clientSpecialisation = read_from_csv('serverNew.csv')
trainInputsNew = [[id[i], ultima_com[i], information[i], averageCommunicLength[i], clientSpecialisation[i]] for i in range(100, 150)]
k.fit_trained_already(trainInputsNew)

print('\n====================== dupa citirea celor 50 de date noi =======================')
media = [0,0]

for each in range(len(k.clusters)):
    print('\nCLUSTER ' + str(each) + ' : ')
    for one in k.clusters[each]:
        print(one)
        media[each] += one[3]

print('Durata medie a primului cluster : ' + str(media[0] / len(k.clusters[0])))
print('Durata medie a celui de-al doilea cluster : ' + str(media[1] / len(k.clusters[1])))