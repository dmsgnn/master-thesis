import os
import shutil


def main():
    print("dimension of the matrix: ", end='')

    dim = input()
    if not dim.isnumeric():
        return

    cora = open("data/cora/cora.content", "r")
    cites = open("data/cora/cora.cites", "r")

    folder = "data/cora" + dim + "/"

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.mkdir(folder)
    print("directory " + folder + " created")

    new_cora = open(folder + "cora.content", "a")
    new_cites = open(folder + "cora.cites", "a")

    nodes = []

    idx = 0
    for line in cora:
        new_cora.write(line)
        nodes.append(line.split()[0])
        idx += 1
        if idx == int(dim):
            break

    for line in cites:
        if line.split()[0] in nodes and line.split()[1] in nodes:
            new_cites.write(line)


if __name__ == '__main__':
    main()
