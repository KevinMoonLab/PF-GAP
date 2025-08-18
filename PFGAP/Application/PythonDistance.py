import numpy as np
from geomstats.geometry.hypersphere import Hypersphere

sphere = Hypersphere(dim=2)


# def Distance(s,t,directory):
#     ans = [(s[i]-t[i])**2 for i in range(len(s))]
#     ans = np.sqrt(np.sum(ans))
#     filename = directory + "/distanceanswer.txt"
#     with open(filename, "w") as file:
#         file.write(str(ans))
#     print(ans)
#     return ans

def Distance(s,t,directory):
    ans = sphere.metric.dist(s, t)

    # Required stuff
    filename = directory + "/distanceanswer.txt"
    with open(filename, "w") as file:
        file.write(str(ans))
    print(ans)
    return ans

