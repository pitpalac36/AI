import fileUtils
import algorithm


def run():
    graf = []
    nrOrase, start, end = fileUtils.read_from_file(graf)
    fileUtils.write_to_file(
        nrOrase,
        algorithm.find_path(graf, nrOrase, 0, -1),
        algorithm.find_path(graf, nrOrase, start - 1, end - 1)
    )


run()
