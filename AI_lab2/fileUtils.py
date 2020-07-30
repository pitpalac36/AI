

def read_from_file(matrix):
    file = open('./data/input.txt', 'r')
    nrOrase = int(file.readline())
    for i in range(0, nrOrase):
        linie = []
        for elem in file.readline().split(","):
            linie.append(int(elem))
        matrix.append(linie)
    orasStart = int(file.readline())
    orasFinal = int(file.readline())
    file.close()
    return nrOrase, orasStart, orasFinal


def create_output(rezCerinta, lines):
    line = ""
    cost, lant = rezCerinta
    lines.append(str(len(lant)) + '\n')
    for i in range(0, len(lant)):
        lant[i] += 1
        line += str(lant[i]) + ","
    line = line[:-1]
    lines.append(line)
    lines.append('\n' + str(cost) + '\n')


def write_to_file(nrOrase, rezCerinta1, rezCerinta2):
    lines = []

    # stergem orasul sursa
    del rezCerinta1[1][-1]

    # afisare cerinta 1
    create_output(rezCerinta1, lines)

    # afisare cerinta 2
    create_output(rezCerinta2, lines)

    file = open('./data/output.txt', 'w')
    file.writelines(lines)
    file.close()
