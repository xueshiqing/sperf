import pandas as pd
import numpy as np
np.random.seed(0)

base = np.array(["2013-01-01 00:00:00"], "datetime64[ns]")

a = (np.random.rand(30)*1000000*1000).astype(np.int64)*1000000
t1 = base + a
t1.sort()
print(t1.shape)

b = (np.random.rand(10)*1000000*1000).astype(np.int64)*1000000
t2 = base + b
t2.sort()
print(t2.shape)

idx = np.searchsorted(t1, t2) - 1
mask = idx >= 0

df = pd.DataFrame({"t1":t1[idx][mask], "t2":t2[mask]})

print(df)