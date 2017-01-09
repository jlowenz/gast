import time
import sys, os
import inspect
import numpy as np
from collections import defaultdict

def get_dict(d, k):
    if k not in d:
        d[k] = {}
    return d[k]

def get_list(d, k):
    if k not in d:
        d[k] = []
    return d[k]

class Stats(object):
    def __init__(self):
        self.counters_ = defaultdict(dict)
        self.metrics_ = defaultdict(dict)

    @property
    def raw_metrics(self):
        return self.metrics_

    @property
    def raw_counters(self):
        return self.counters_
        
    def _get_counter(self, root, path):
        if isinstance(path,list):
            if len(path) == 1:
                if path[0] in root:
                    return root[path[0]]
                else:
                    root[path[0]] = 0
                    return root[path[0]]
            return self._get_counter(get_dict(root,path[0]), path[1:])
        if path in root:
            return root[path]
        else:
            root[path] = 0
            return root[path]        

    def _set_counter(self, root, path, v):        
        if isinstance(path,list):
            if len(path) == 1:
                root[path[0]] = v
                return v
            else:
                return self._set_counter(root[path[0]], path[1:], v)

        root[path] = v
        return v

    def _get_metric(self, root, path):
        if isinstance(path,list):
            if len(path) == 1:
                if path[0] in root:
                    return root[path[0]]
                else:
                    root[path[0]] = []
                    return root[path[0]]
            return self._get_metric(get_dict(root,path[0]), path[1:])
        if path in root:
            return root[path]
        else:
            root[path] = []
            return root[path]

    def get_metric(self, path):
        return self._get_metric(self.metrics_, path)
        
    def get_counter(self, path):
        return self._get_counter(self.counters_, path)
    def set_counter(self, path, v):
        return self._set_counter(self.counters_, path, v)
    
    def incr(self, path, count=1):
        v = self.get_counter(path)
        self.set_counter(path, v + count)

    def metric(self, path, m):
        self.get_metric(path).append(m)

    def select_metric(self, path):
        v = self.get_metric(path)
        if isinstance(v,list):
            a = np.array(v, dtype=np.float32)
            return (np.amin(a), np.amax(a), np.mean(a), np.std(a))
        return v

def log_elapsed(elapsed, args=[], kw={}, logger=None, name_fn=None, name=None):
    #pdb.set_trace()
    n = ""
    if name_fn:
        n = name_fn(*args, **kw)
    elif name:        
        n = name
    else:
        (frm, fname, lineno, fn_name, lines, idx) = inspect.stack()[1]
        n = "{0} : {1} : {2}".format(fname, fn_name, lineno)
    if logger:
        logger(n, elapsed)
    print "%r %2.6f sec" % (n, elapsed)

class Timer(object):
    def __init__(self, logger=None, name_fn=None, name=None):
        self.logger = logger
        self.name_fn = name_fn
        self.name = name
    
    def __enter__(self):
        self.ts = time.time()

    def __exit__(self, typ, value, tb):
        te = time.time()
        elapsed = te-self.ts
        log_elapsed(elapsed, logger=self.logger,
                    name_fn=self.name_fn, name=self.name)

## timing utility
def timeit(method, logger=None, name_fn=None, name=None):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        elapsed = te-ts
        log_elapsed(elapsed, args=args, kw=kw, logger=logger, name_fn=name_fn, name=name)
        return result
    return timed
