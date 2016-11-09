from numpy import *
from matplotlib import pyplot
from scipy import signal
from lib import genetic
from lib.audio import ioutils
from lib.audio import compare

# Params: [delay time samples, delay gain (lin)]
def delay (params, x):
    delaySamples = abs(floor(params[0]))
    delaySpace = zeros(delaySamples)
    return concatenate((x, delaySpace)) + params[1] * concatenate((delaySpace, x))

trueParams = array([44.1 * 100, 0.66])
Fs, inputSignal = ioutils.loadWav('audioSamples/drums.wav');
inputSignal = inputSignal[0:44100 * 5]
targetSignal = delay(trueParams, inputSignal)

def fitnessFunc (params):
    if (params[0] <= 0.0):
        return float("inf")
    return compare.compare_envelope(targetSignal, delay(params, inputSignal))

population = genetic.Population(
    dimm=2,
    means=array([44.1 * 300, 1]),
    std_devs=array([44.1 * 300, .5]),
    fitness_func=fitnessFunc,
    stable_pop = 20
)

print("Evolving:")
population.evolve(25)

king = population.selectMostFit(1)[0]

print("Actual: ", trueParams)
print("Estimate: ", king)

ioutils.saveWav('fftarget.temp.wav', Fs, targetSignal)
ioutils.saveWav('ffestimate.temp.wav', Fs, delay(king, inputSignal))
