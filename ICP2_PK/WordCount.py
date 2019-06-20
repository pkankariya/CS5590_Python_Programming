# Word count in file
infile = open('WordCount','r')
line = infile.read().splitlines()
print(line)
dict = {}
numWords = 0
with open('WordCount', 'r') as f:
    for line in f:
        for words in line.split(' '):
            print(words)
            if words in dict:
                value = dict[words]
                value = value+1
                dict.update({words:value})
            else:
                dict.update({words: 1})
print(dict)
file = open('newfile','w')
for i in dict.keys():
    file.write(str(i))
    file.write(str(dict[i]))
    file.write('\n')