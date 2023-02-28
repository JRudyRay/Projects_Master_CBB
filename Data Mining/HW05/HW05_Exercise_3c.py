COL1A1 = 'GPAGFAGPPGDA'
COL1A2 = 'PRGDQGPVGRTG'
GPR_143= 'GFPNFDVSVSDM'


def kernel_GXY(string1, string2):
    
    x = list(string1)
    y = list(string2)
    length = len(x)
    counter = 0
    for i in range(0, length-2):
        for j in range(0, length-2):

            if x[i] == 'G' and y[j] == 'G': # first value must be a G
                counter += 1

                if x[i+1] == y[j+1]:
                    counter += 1
                
                if x[i+2] == y[j+2]:
                    counter += 1
            else:
                continue
    return counter


result1 = kernel_GXY(COL1A1, COL1A2)
print('k_GXY(COL1A1, COL1A2) =', result1)

result2 = kernel_GXY(COL1A1, GPR_143)
print('k_GXY(COL1A1, GPR_143) =', result2)

