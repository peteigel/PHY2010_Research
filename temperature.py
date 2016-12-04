from numpy import *
from lib import genetic

set_printoptions(precision=2, suppress=True)

degC = array([-40, 0, 100])
degF = array([-40, 32, 212])

def evaluateFitness(params):
    degFEstimate = params[0] * degC + params[1]
    return sum((degFEstimate - degF) ** 2)

tempPopulation = genetic.Population(
    dimm = 2,
    stable_pop = 10,
    std_devs = array([3, 10]),
    repro_rate = 3,
    delta_mutation = 0.99,
    fitness_func = evaluateFitness
)

print("Initial Population")
print(tempPopulation.population)

print("First Offspring")
tempPopulation.evolve(1)
print(tempPopulation.nextPopulation)

print("Next Population")
print(tempPopulation.population)

tempPopulation.evolve(100)
print("Final Population")
print(tempPopulation.population)
