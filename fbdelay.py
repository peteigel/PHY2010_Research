from numpy import *
from matplotlib import pyplot
from scipy import signal
from lib import genetic
from lib.audio import ioutils
from lib.audio import compare


def delay (params, x):
    d = abs(floor(params[0]))
    g = params[1]
    y = zeros(inputSignal.size)
    for t in range(x.size):
        if (t-d) < 1:
            y[t] = x[t]
        else:
            y[t] = x[t] + g * y[t-d]
    return y

trueParams = array([44.1 * 100, 0.66])
Fs, inputSignal = ioutils.loadWav('audioSamples/drums.wav');
inputSignal = inputSignal[0:44100 * .5]
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
    stable_pop = 5
)

print("Evolving:")
population.evolve(5)

king = population.selectMostFit(1)[0]

print("Actual: ", trueParams)
print("Estimate: ", king)

ioutils.saveWav('fbtarget.temp.wav', Fs, targetSignal)
ioutils.saveWav('fbestimate.temp.wav', Fs, delay(king, inputSignal))