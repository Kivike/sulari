

nextPoints = [4, 2, 1, 0, 7, 3, 5, 6]

def calcBinary(pointCount, startPoint):
    num = 0

    for i in range(pointCount):
        index = (startPoint + i) % 8
        num |= 1 << nextPoints[index]

    return num

for n in range(1, 8):
    print("COUNT: " + str(n))
    for r in range(8):
        bin = calcBinary(n, r)
        print(bin)
