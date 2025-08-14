import numpy as np

def Distance(s,t,directory):
    ans = [(s[i]-t[i])**2 for i in range(len(s))]
    ans = np.sqrt(np.sum(ans))
    filename = directory + "/distanceanswer.txt"
    with open(filename, "w") as file:
        file.write(str(ans))
    print(ans)
    return ans
