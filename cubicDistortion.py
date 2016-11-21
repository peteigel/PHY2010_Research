from numpy import *
from matplotlib import pyplot
from scipy import signal
from lib import genetic
from lib.audio import ioutils
from lib.audio import compare


def distortion (params, x):
    g = params[0]
    y = zeros(inputSignal.size)
    curr = 0.0

    for t in range(x.size):
       curr = x[t]
       y[t] = curr - ((1/3)*curr**3)
    return y*g + x*(1-g)

trueParams = array([0.66])
Fs, inputSignal = ioutils.loadWav('audioSamples/drums.wav');
#Fs = 44,100
#inputSignal = sin(2*pi*1000*(1/Fs))
inputSignal = inputSignal[0:44100 * 2]
targetSignal = distortion(trueParams, inputSignal)

def fitnessFunc (params):
    if (params[0] <= 0.0):
        return float("inf")
    return compare.compare_envelope(targetSignal, distortion(params, inputSignal))

population = genetic.Population(
    dimm=1,
    means=array([.5]),
    std_devs=array([.3]),
    fitness_func=fitnessFunc,
    stable_pop = 5
)

print("Evolving:")
population.evolve(5)

king = population.selectMostFit(1)[0]

print("Actual: ", trueParams)
print("Estimate: ", king)

outSignal = distortion(king, inputSignal)

fin, pin = signal.welch(targetSignal, Fs, scaling='spectrum')
fout, pout = signal.welch(outSignal, Fs, scaling='spectrum')

pyplot.loglog(fin, pin, fout, pout)
pyplot.show()

ioutils.saveWav('cdtarget.temp.wav', Fs, targetSignal)
ioutils.saveWav('cdestimate.temp.wav', Fs, distortion(king, inputSignal))
