import sys
Diss = {};
def main():

    with open(sys.argv[1],"r") as readFile:
        lines = readFile.read().splitlines()
        # splitContent = lines.splitList()
        for line in lines:
            words = line.split()
            for word in words:
                if word in Diss:
                    Diss[word] += 1
                else:
                    Diss[word] = 1
    with open('Q1.txt',"w+") as writeFile:
        count = 0
        for word in Diss:
            if count < len(Diss) - 1:
                writeFile.write(word+' ' + str(count) + ' ' + str(Diss[word]) + '\n')
            else:
                writeFile.write(word+' ' + str(count) + ' ' + str(Diss[word]))
            count += 1


if __name__ == '__main__':
    main()
