# Python code to convert into dictionary

def Convert(tup, di):
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di

# Driver Code
tups = [("akash",('Physics',90)), ("gaurav",('Arts', 92)), (("anand", 14)),
        ("suraj",('History',20)), ("akhil",('Chemistry',25)),("akash",('Chemistry',95)), ("ashish",('Maths',30))]
dictionary = {}
print(Convert(tups, dictionary))