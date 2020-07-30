from file_utils import read_from_csv1, read_from_csv2
from sklearn.cluster import KMeans

# Sa se determine densitatea medie a vinurilor cu grad de alcool peste medie.

density, alcohol = read_from_csv1('wine.csv')

# calculam densitatea medie
avg = sum(density) / len(density)

big_avg = 0
k = 0

for each in density:
    if each > avg:
        big_avg += each
        k += 1

print('grad mediu de alcool : ' + str(avg))
print('densitatea medie a vinurilor cu grad de alcool peste medie : ' + str(big_avg / k))

# producatorii de vin stabilesc preturile de vanzare ale acestor vinuri pe baza a 5 grupe de
# vinuri identificate pe baza datelor avute la dispozitie -> in fct de aciditate

fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,\
    total_sulfur_dioxide, density, pH, sulphates, alcohol = read_from_csv2('wine.csv')

trainInputs = [
    [fixed_acidity[i], volatile_acidity[i], citric_acid[i], residual_sugar[i], chlorides[i], free_sulfur_dioxide[i],
     total_sulfur_dioxide[i], density[i], pH[i], sulphates[i], alcohol[i]] for i in range(len(fixed_acidity))]


# Fixed acidity = 8.1, Volatile acidity = 0.545, Citric acid = 0.18, Residual sugar =
# 1.9, Chlorides = 0.08, Free sulfur dioxide = 13, Total sulfur dioxide = 35, Density = 0.9972, Ph = 3.3,
# Sulphates = 0.59, Alcohol = 9.8

unsupervisedClassifier = KMeans(n_clusters=5, random_state=0)
unsupervisedClassifier.fit(trainInputs)
computedTestIndex = unsupervisedClassifier.predict([[8.1, 0.545, 0.18, 1.9, 0.08, 13, 35, 0.9972, 3.3, 0.59, 9.8]])
print('vinul cu datele respective face parte din grupa ' + str(computedTestIndex[0]))

for each in trainInputs:
    print(str(each) + '   categorie : ' + str(unsupervisedClassifier.predict([each])[0]))

# producatorii doresc sa stabileasca pretul si pentru un vin despre care cunosc doar urmatoarele informatii:
# Fixed acidity = 7.9, Residual sugar = 1.8, , Density = 0.9969, Ph = 3.04, Alcohol = 9.8.
# Din care din cele 5 grupe face parte acest vin?

avg_volatile_acidity = 0
avg_citric_acid = 0
avg_chlorides = 0
avg_free_sulfur_dioxide = 0
avg_total_sulfur_dioxide = 0
avg_sulphates = 0

for each in unsupervisedClassifier.cluster_centers_:
    avg_volatile_acidity += each[1]
    avg_citric_acid += each[2]
    avg_chlorides += each[4]
    avg_free_sulfur_dioxide += each[5]
    avg_total_sulfur_dioxide += each[6]
    avg_sulphates += each[9]

avg_volatile_acidity /= 5
avg_citric_acid /= 5
avg_chlorides /= 5
avg_free_sulfur_dioxide /= 5
avg_total_sulfur_dioxide /= 5
avg_sulphates /= 5

computedTestIndex = unsupervisedClassifier.predict([[7.9, avg_volatile_acidity, avg_citric_acid, 1.8, avg_chlorides,avg_free_sulfur_dioxide,
                                                     avg_total_sulfur_dioxide, 0.9969, 3.04, avg_sulphates, 9.8]])

print('vinul cu datele respective face parte din grupa ' + str(computedTestIndex[0]))
