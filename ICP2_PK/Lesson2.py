# Finding the average weight of students in the class
N = int(input("How many students are present in class? "))
sum = 0.0
for i in range(N):
    x = int(input("Enter the weight of student in lbs:"))
    sum = sum + x
    print("\nThe average weight of students in the class is ", sum / N)

# Converting the weight of students in the class from lbs to kgs
N = int(input("How many students are present in class? "))
for i in range(N):
    x = int(input("Enter the weight of student in lbs:"))
    wt = int(x / 2.2)
    print("The weight of the student in kgs is ", wt)

# Reading altenate characters of the string
str = "Good Evening"
str_alt = str[0::2]
print(str_alt)

# # Word count in file
# infile = open('WordCount','r')
# dict = {}
# line = infile.read().splitlines()
# print(line)
# sum = 0.0
# count = 0
#
# while line != "":
#     for xStr in line.split(","):
#         sum = sum + len(xStr)
#         count = count + 1
#     line = infile.readline()
# print("\nThe average of the numbers is ", sum / count)
