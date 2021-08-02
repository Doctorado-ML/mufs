import warnings
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
cfs_f = mufsc.cfs(X, y).get_results()
print(cfs_f)
print("FCBF")
fcfb_f = mufsc.fcbf(X, y, 5e-2).get_results()
print(fcfb_f, len(fcfb_f))
print("X.shape=", X.shape)
clf = Stree(random_state=0)
print("Accuracy whole dataset", clf.fit(X, y).score(X, y))
clf = Stree(random_state=0)
print("Accuracy cfs", clf.fit(X[:, cfs_f], y).score(X[:, cfs_f], y))
clf = Stree(random_state=0)
subf = fcfb_f
print("Accuracy fcfb", clf.fit(X[:, subf], y).score(X[:, subf], y))
