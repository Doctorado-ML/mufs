import warnings
import time
from mufs import MUFS
from mufs.Metrics import Metrics
from stree import Stree
import numpy as np
from scipy.io import arff

mufsc = MUFS(discrete=False)

filename = "conn-bench-sonar-mines-rocks.arff"
data, meta = arff.loadarff(filename)
train = np.array([data[i] for i in meta])
X = train.T
X = X[:, :-1].astype("float64")
y = data["clase"]

m, n = X.shape
print("* Differential entropy in X")
for i in range(n):
    print(i, Metrics.differential_entropy(X[:, i], k=10))
print("* Information Gain")
print("- Continuous features")
print(Metrics.information_gain_cont(X, y))
for i in range(n):
    print(i, Metrics.information_gain_cont(X[:, i], y))
# Classification
warnings.filterwarnings("ignore")
print("CFS")
now = time.time()
cfs_f = mufsc.cfs(X, y).get_results()
time_cfs = time.time() - now
print(cfs_f, "items: ", len(cfs_f), f"time: {time_cfs:.3f} seconds")
print("FCBF")
now = time.time()
fcbf_f = mufsc.fcbf(X, y, 0.07).get_results()
time_fcbf = time.time() - now
print(fcbf_f, "items: ", len(fcbf_f), f"time: {time_fcbf:.3f} seconds")
now = time.time()
print("IWSS")
iwss_f = mufsc.iwss(X, y, 0.5).get_results()
time_iwss = time.time() - now
print(iwss_f, "items: ", len(iwss_f), f"time: {time_iwss:.3f} seconds")
print("X.shape=", X.shape)
clf = Stree(random_state=0)
print("Accuracy whole dataset", clf.fit(X, y).score(X, y))
clf = Stree(random_state=0)
print("Accuracy cfs", clf.fit(X[:, cfs_f], y).score(X[:, cfs_f], y))
clf = Stree(random_state=0)
print("Accuracy fcfb", clf.fit(X[:, fcbf_f], y).score(X[:, fcbf_f], y))
clf = Stree(random_state=0)
print("Accuracy iwss", clf.fit(X[:, iwss_f], y).score(X[:, iwss_f], y))
