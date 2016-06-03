
ifile = open("../../oanc_SlateTestData.txt","r")
lines = ifile.readlines()
outlines = []
i = 0
for line in lines:
  if len(line.strip()) > 0:
    i += 1
    res = i % 500
    if (res >= 490) and (res <= 499):
      outlines.append(line)

ifile.close()
ofile = open("../../oanc_SlateLambdaTraining.txt","w+")
ofile.writelines(outlines)
ofile.close()



