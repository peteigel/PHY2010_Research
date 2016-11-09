from numpy import *
from scipy import signal
from scipy import fftpack
env_filter_b, env_filter_a = signal.butter(2, 20/22050, btype='lowpass')

def safe_compare(x1, x2, fn):
        size1 = size(x1)
        size2 = size(x2)
        if (size1 > size2):
            return fn(x1[0:size2], x2)
        else:
            return fn(x1, x2[0:size1])

def compare_samples(x1, x2, func=lambda x: x):
    return safe_compare(
        x1, x2,
        lambda x1, x2: sum((func(x1) - func(x2)) ** 2)
    )

def compare_envelope(x1, x2):
    return compare_samples(
        x1, x2,
        func = lambda x: signal.lfilter(env_filter_b, env_filter_a, x)
    )

def compare_spectra(x1, x2):
    return compare_samples(
        x1, x2,
        lambda x: fftpack.dct(x, n=2205)
    )
