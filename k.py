import warnings
from sklearn.datasets import load_wine
from mfs import MFS
from mfs.Metrics import Metrics
from stree import Stree

mfsc = MFS(discrete=False)
mfsd = MFS(discrete=True)
X, y = load_wine(return_X_y=True)
m, n = X.shape
print("* Differential entropy in X")
for i in range(n):
    print(i, Metrics.differential_entropy(X[:, i], k=10))
print("* Information Gain")
print("- Discrete features")
print(Metrics.information_gain(X, y))
for i in range(n):
    print(i, Metrics.information_gain(X[:, i], y))
print("- Continuous features")
print(Metrics.information_gain_cont(X, y))
for i in range(n):
    print(i, Metrics.information_gain_cont(X[:, i], y))
# Classification
warnings.filterwarnings("ignore")
print("CFS Discrete")
cfs_d = mfsd.cfs(X, y).get_results()
print(cfs_d)
print("CFS continuous")
cfs_f = mfsc.cfs(X, y).get_results()
print(cfs_f)
print("FCBF Discrete")
print(mfsd.fcbf(X, y, 5e-2).get_results())
print("FCBF continuous")
fcfb_f = mfsc.fcbf(X, y, 5e-2).get_results()
print(fcfb_f, len(fcfb_f), "X.shape=", X.shape)
clf = Stree(random_state=0)
print("completo", clf.fit(X, y).score(X, y))
clf = Stree(random_state=0)
print("cfs discreto", clf.fit(X[:, cfs_d], y).score(X[:, cfs_d], y))
print("cfs continuo", clf.fit(X[:, cfs_f], y).score(X[:, cfs_f], y))
clf = Stree(random_state=0)
subf = fcfb_f[:6]
print("fcfb", clf.fit(X[:, subf], y).score(X[:, subf], y))
