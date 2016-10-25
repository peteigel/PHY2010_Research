from numpy import *

class Population:
    def __init__ (self, search_dimm, stable_pop):
        self.search_dimm = search_dimm
        self.stable_pop = stable_pop

    def populate (self, mean, std_dev):
        self.population = random.randn(self.stable_pop, self.search_dimm)
        for col in range(self.search_dimm):
            self.population[:, col] = self.population[:, col] * std_dev[col] + mean[col]

    def createOffspring (self, repro_rate, std_dev):
        self.nextPopulation = empty((self.stable_pop * repro_rate, self.search_dimm))
        for i in range(self.stable_pop):
            offspring = random.randn(repro_rate, self.search_dimm)
            offspring = apply_along_axis(
                lambda row: row * std_dev + self.population[i],
                1,
                offspring
            )
            self.nextPopulation[(i * repro_rate):((i + 1) * repro_rate)] = offspring

    def cullOffspring (self, fitness_func):
        fitness = apply_along_axis(fitness_func, 1, self.nextPopulation)
        fit_indices = argpartition(fitness, self.stable_pop)[0:self.stable_pop]
        self.population = self.nextPopulation[fit_indices]

    def selectMostFit(self, n, fitness_func):
        fitness = apply_along_axis(fitness_func, 1, self.population)
        fit_indices = argpartition(fitness, n)[0:n]
        return self.population[fit_indices]
