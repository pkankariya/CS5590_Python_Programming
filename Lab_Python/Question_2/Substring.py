# String input from the user
word = input("Enter a word of your choice:")

# Assigning variables to hold string and substring
longestTillNow = ""
longestOverall = ""

# Initializing a set for the characters to be identified
se = set()

# Iteration through every character in string
for i in range(0, len(word)):
    # Verifying the character at index position i's presence within the set created
    char = word[i]
    # Character from index position i exists in set
    if char in se:
        longestTillNow = ""
        se.clear()
    # Character from index position i does not exist in set
    longestTillNow = longestTillNow + char
    se.add(char)
    # Verifying the length of the current substring with the previous one
    # When current substring length is greater than the previous substring
    if len(longestTillNow) > len(longestOverall):
        longestOverall = longestTillNow

# Displaying the substring derived with corresponding information
print("The longest substring derived is", longestOverall)
print("The length of the substring derived is", len(longestOverall))
