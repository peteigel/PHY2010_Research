from numpy import *
from scipy.io import wavfile

def loadWav (filename):
    Fs, rawData = wavfile.read(filename)
    scale = -(iinfo(rawData.dtype).min)
    floatData = rawData.astype(float32) / scale
    return Fs, floatData


def saveWav (filename, Fs, floatData):
    scale = -(iinfo(int16).min)
    rawData = (floatData * scale).astype(int16)
    wavfile.write(filename, Fs, rawData)
