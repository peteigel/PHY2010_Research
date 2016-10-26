from numpy import *

class  Population:
    def __init__ (
        self,
        dimm = None,
        stable_pop = 300,
        means = None,
        std_devs = None,
        repro_rate = 10,
        mutation = 0.1,
        fitness_func = None
    ):
        assert fitness_func is not None
        assert type(dimm) is int

        if means is None:
            means = zeros(dimm)

        if std_devs is None:
            std_devs = ones(dimm)

        self.dimm = dimm
        self.stable_pop = stable_pop
        self.means = means
        self.std_devs = std_devs
        self.repro_rate = repro_rate
        self.mutation = mutation
        self.fitness_func = fitness_func

        self.populate()
        self.nextPopulation = empty((self.stable_pop * repro_rate, self.dimm))

    def populate (self):
        self.population = random.randn(self.stable_pop, self.dimm)
        for col in range(self.dimm):
            self.population[:, col] = self.population[:, col] * self.std_devs[col] + self.means[col]

    def evolve (self, generations):
        for n in range(generations):
            self.procreate()
            self.select()

    def procreate (self):
        for i in range(self.stable_pop):
            offspring = random.randn(self.repro_rate, self.dimm)
            offspring = apply_along_axis(
                lambda row: row * (self.std_devs * self.mutation) + self.population[i],
                1,
                offspring
            )
            self.nextPopulation[(i * self.repro_rate):((i + 1) * self.repro_rate)] = offspring

    def select (self):
        fitness = apply_along_axis(self.fitness_func, 1, self.nextPopulation)
        fit_indices = argpartition(fitness, self.stable_pop)[0:self.stable_pop]
        self.population = self.nextPopulation[fit_indices]

    def selectMostFit (self, n):
        fitness = apply_along_axis(self.fitness_func, 1, self.population)
        fit_indices = argpartition(fitness, n)[0:n]
        return self.population[fit_indices]
