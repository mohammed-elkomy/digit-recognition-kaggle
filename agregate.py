from collections import defaultdict

import pandas as pd


def find_majority(k):
    myMap = defaultdict(int)
    maximum = ('', 0)  # (occurring element, occurrences)
    for n in k:
        myMap[n] += 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]:
            maximum = (n, myMap[n])

    return maximum[0]


a = pd.DataFrame(pd.read_csv("subs/sub_a.csv"))
b = pd.DataFrame(pd.read_csv("subs/sub_b.csv"))
c = pd.DataFrame(pd.read_csv("subs/sub_c.csv"))
d = pd.DataFrame(pd.read_csv("subs/sub_d.csv"))  # :D !! added nothing
e = pd.DataFrame(pd.read_csv("subs/sub_e.csv"))  # :D !! added nothing

labels_predicted = []

for i in range(a.shape[0]):
    labels_predicted.append(find_majority([a.iloc[i]['Label'],
                                           b.iloc[i]['Label'],
                                           c.iloc[i]['Label'],
                                           d.iloc[i]['Label'],
                                           e.iloc[i]['Label']]))

df = pd.DataFrame({'ImageId': range(1, a.shape[0] + 1), 'Label': labels_predicted})
df.to_csv("subs/agg.csv", index=False)
