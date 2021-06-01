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
print("CFS Discrete")
print(mfsd.cfs(X, y).get_results())
print("CFS continuous")
cfs_f = mfsc.cfs(X, y).get_results()
print(cfs_f)
print("FCBF Discrete")
print(mfsd.fcbf(X, y, 1e-7).get_results())
print("FCBF continuous")
fcfb_f = mfsc.fcbf(X, y, 1e-7).get_results()
print(fcfb_f)
clf = Stree(random_state=0)
print("completo", clf.fit(X, y).score(X, y))
clf = Stree(random_state=0)
print("cfs", clf.fit(X[:, cfs_f], y).score(X[:, cfs_f], y))
clf = Stree(random_state=0)
print("fcfb", clf.fit(X[:, fcfb_f], y).score(X[:, fcfb_f], y))
