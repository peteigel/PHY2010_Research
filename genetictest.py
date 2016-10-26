from numpy import *
from matplotlib import pyplot
from lib import genetic

def func (params, x):
    # y:= a * (x - b)^3 + c * sin(d * x - e)
    y1 = (x - params[0]) ** 3
    y2 = sin(params[1] * x - params[2])
    return params[3] * y1 + params[4] * y2

def randomParams ():
    paramRanges = array([1, 2, 50, 1, 100])
    return (random.rand(5) - 0.5) * 2 * paramRanges

#trueParams = randomParams()
# I like these ones
trueParams = array([  4.29383821e-02, 9.60946324e-01, 2.05602723e+01, -3.13699765e-01, -5.78706031e+01])

sampleX = linspace(-10, 10, 1000)
trueY = func(trueParams, sampleX)

myPopulation = genetic.Population(
    dimm = 5,
    std_devs = array([1, 1, 2, 1, 64]),
    fitness_func = lambda params: sum((func(params, sampleX) - trueY) ** 2)
)

myPopulation.evolve(100)

mostFit = myPopulation.selectMostFit(5)
mostFit = average(mostFit, 0)
estimatedY = func(mostFit, sampleX)
rSquared = 1 - sum((estimatedY - trueY) ** 2) / sum((trueY - average(trueY)) ** 2)

print("Actual: ", trueParams)
print("Estimate: ", mostFit)
print("Error: ", (trueParams - mostFit) / trueParams)
print("R^2 = {}".format(rSquared))


pyplot.plot(sampleX, trueY, sampleX, estimatedY)
pyplot.show()
