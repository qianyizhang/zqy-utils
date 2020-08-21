# zqy_utils
some utility functions created/used by zqy while he is here

- <b>This is no longer maintained by the onriginal owner</b>
- Last modified @ 2020/08/21

## install

```
# at current root
pip install -e .    # this is installed to python lib like numpy, in dev mode
```
or
```
# add current root to your PYTHONPATH (environment variable) in .bashrc
export PYTHONPATH="$PYTHONPATH:$CURRENT_ROOT"   # you have to modify CURRENT_ROOT yourself
```

## usage
There are a number of functions that can be useful, only a few are listed, rest are open to explore
```
# the load and save are handy, they automatically address the extentions
from zqy_utils import load, save

a = load("a.npy")
b = load("b.json")
c = load("c.pkl")
d = load("d.dcm")
...
save(a, "a.npy")
...

# 
```