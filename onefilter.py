from scipy import signal
from numpy import *
from lib import genetic
from lib.audio import compare
from lib.audio import ioutils
from matplotlib import pyplot

# Params - [b0, b1, b2, a0, a1, a2]
def onefilter (params, x):
    b = params[0:3]
    a = params[3:6]
    return signal.lfilter(b, a, x)

Fs, drySignal = ioutils.loadWav('audioSamples/whiteNoiseRef.wav')
_, targetSignal = ioutils.loadWav('audioSamples/whiteNoiseOneBandEQ.wav')

def fitnessFunc (params):
    return compare.compare_spectrum(targetSignal, onefilter(params, drySignal))

filterPopulation = genetic.Population(
    dimm=6,
    means=zeros(6),
    std_devs=ones(6) * 10,
    fitness_func=fitnessFunc,
    stable_pop = 20
)

print("Evolving:")
filterPopulation.evolve(100)

king = filterPopulation.selectMostFit(1)[0]

print("Estimate: ", king)

outSignal = onefilter(king, drySignal)

fin, pin = signal.welch(targetSignal, Fs, scaling='spectrum')
fout, pout = signal.welch(outSignal, Fs, scaling='spectrum')

pyplot.loglog(fin, pin, fout, pout)
pyplot.show()

ioutils.saveWav('filterestimate.temp.wav', Fs, outSignal)
