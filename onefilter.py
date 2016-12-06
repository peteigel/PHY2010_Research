from scipy import signal
from numpy import *
from lib import genetic
from lib.audio import compare
from lib.audio import ioutils
from lib.audio import plots

# Params - [b0, b1, b2, a0, a1]
def onefilter (params, x):
    b = params[0:3]
    a = params[3:5]
    return signal.lfilter(b, a, x)

Fs, drySignal = ioutils.loadWav('audioSamples/whiteNoiseRef.wav')
_, targetSignal = ioutils.loadWav('audioSamples/whiteNoiseOneBandEQ.wav')

def fitnessFunc (params):
    return compare.compare_spectrum(targetSignal, onefilter(params, drySignal))

filterPopulation = genetic.Population(
    dimm=5,
    means=zeros(5),
    std_devs=ones(5) * 20,
    mutation=.5,
    fitness_func=fitnessFunc,
    stable_pop = 40
)

print("Evolving:")
filterPopulation.evolve(300)

king = filterPopulation.selectMostFit(1)[0]
#a change

print("Estimate: ", king)

outSignal = onefilter(king, drySignal)

plots.plotSpectrum(targetSignal, outSignal, Fs)

ioutils.saveWav('filterestimate.temp.wav', Fs, outSignal)
ioutils.saveWav('filtertarget.temp.wav', Fs, targetSignal)
