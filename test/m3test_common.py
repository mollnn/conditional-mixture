import os
import time
import drjit as dr
import numpy as np

LTIME = lambda: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

LOGD = lambda: "\033[30m" + LTIME() + " mts3"
LOGI = lambda: "\033[34m" + LTIME() + " mts3"
LOGT = lambda: "\033[35m" + LTIME() + " mts3"
LOGK = lambda: "\033[32m" + LTIME() + " mts3"
LOGW = lambda: "\033[33m" + LTIME() + " mts3"
LOGE = lambda: "\033[31m" + LTIME() + " mts3"
LEND = "\033[30m"

def read_float_from_file(filename):
    fp = open(filename)
    ans = float(fp.readlines()[0].split()[0])
    fp.close()
    return ans

def write_float_to_file(filename, value):
    fp = open(filename, 'w')
    print(value, file=fp)
    fp.close()

def add_float_to_file(filename, value):
    original = read_float_from_file(filename)
    ans = original + value 
    write_float_to_file(filename, ans)
    return ans 

def try_remove(filename):
    if os.path.exists(filename):
        os.remove(filename)

def get_elapsed_execution_time():
    hist = dr.kernel_history()
    elapsed_time = 0
    for entry in hist:
        elapsed_time += entry['execution_time']
    return elapsed_time

def se(a, b):
    return dr.sqr(a - b) / 1.0

def mse(a, b):
    return dr.mean(se(a, b))

def relse(a, b):
    return dr.sqr(a - b) / (dr.sqr(b) + 0.1)

def relmse(a, b):
    return dr.mean(relse(a, b))

def normalize(normalmap):
    return 0.5 + (normalmap - 0.5) * 0.5 / np.linalg.norm(normalmap - 0.5, axis=-1, keepdims=True)

def decode(normalmap):
    t = normalmap * 2 - 1
    return t / np.linalg.norm(t, axis=-1, keepdims=True)
