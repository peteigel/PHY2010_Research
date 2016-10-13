from lib.audio.ioutils import *
from numpy import *
from matplotlib import pyplot

def sineSweep (duration, minFreq, maxFreq):
    freq = linspace(minFreq * pi, maxFreq * pi, num=duration, dtype=float32)
    signal = sin(freq * arange(0, duration))
    return signal

def createSweep (filename):
    Fs = 44100
    sweep = sineSweep(Fs*2, 20/Fs, 20000/Fs)
    saveWav(filename, Fs, sweep)

def analyzeSweep (filename):
    Fs, sweep = loadWav(filename)
    pyplot.plot(sweep)
    pyplot.show()

filename = 'sweep.temp.wav'
createSweep(filename)
analyzeSweep(filename)
