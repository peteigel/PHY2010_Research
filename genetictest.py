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

trueParams = randomParams()
sampleX = linspace(-10, 10, 1000)
trueY = func(trueParams, sampleX)

pop_size = 300
repro_rate = 10
std_dev = array([1, 1, 10, 1, 100])
num_generations = 50

myPopulation = genetic.Population(5, pop_size)
myPopulation.populate(zeros(5), std_dev)

eval_fitness = lambda params: sum((func(params, sampleX) - trueY) ** 2)

for i in range(num_generations):
    myPopulation.createOffspring(repro_rate, std_dev * 0.1)
    myPopulation.cullOffspring(eval_fitness)
    print ("{:.0f}%".format(i / num_generations * 100))

mostFit = myPopulation.selectMostFit(5, eval_fitness)
mostFit = average(mostFit, 0)
estimatedY = func(mostFit, sampleX)
rSquared = 1 - (eval_fitness(mostFit) / sum((trueY - average(trueY)) ** 2))

print("Actual: ", trueParams)
print("Estimate: ", mostFit)
print("Error: ", (trueParams - mostFit) / trueParams)
print("R^2 = {}".format(rSquared))
pyplot.plot(sampleX, trueY, sampleX, estimatedY)
pyplot.show()
