from heap import Heappy


def find_path(graf, nrNoduri, sursa, destinatie):
    # initializam heap-ul de perechi
    # cheia este costul, iar valoarea este lantul
    heap = Heappy()

    # punem costul de la sursa la toate nodurile adiacente
    # construim lantul de inceput
    for i in range(nrNoduri):
        if i != sursa:
            heap.push((graf[sursa][i], [sursa, i]))

    while True:
        # lantul cu costul cel mai mic
        cost, lant = heap.pop()

        # conditie pentru cerinta 1
        if destinatie == lant[-1] or sursa == lant[-1]:
            return cost, lant

        # adaugam nodul sursa la lantul actual (cerinta 1)
        if len(lant) == nrNoduri:
            copy = lant.copy()
            copy.append(sursa)
            heap.push((cost + graf[lant[-1]][sursa], copy))

        # adaugam in heap orasele care nu au fost vizitate
        for i in range(nrNoduri):
            if i not in lant:
                copy = lant.copy()
                copy.append(i)
                heap.push((cost + graf[lant[-1]][i], copy))
