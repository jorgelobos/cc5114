import numpy as np
import matplotlib.pyplot as plt


class Genome:
    def __init__(self, pop_size, fitness_fx, generation_fx, mutation_rate, end_condition):
        self.pop_size = pop_size
        self.fitness_fx = fitness_fx
        self.generation_fx = generation_fx
        self.mutation_rate = mutation_rate
        self.end_condition = end_condition

    def start(self):
        # Solution to find
        expected_string = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        best_outputs = []
        mean_outputs = []
        worst_outputs = []
        # Creating the initial population.
        new_population = np.random.randint(2, size=self.pop_size)
        generation_max_fitness = -1
        last_best = -1
        for generation in range(self.end_condition):
            # Measuring the fitness of each chromosome in the population.
            fitness = self.fitness_fx(bit_string, new_population)
            this_best = np.max(fitness)
            best_outputs.append(this_best)
            if this_best > last_best:
                generation_max_fitness = generation
                last_best = this_best
            mean_outputs.append(np.mean(fitness))
            worst_outputs.append(np.min(fitness))

            # Selecting the best parents in the population for mating.
            num_parents_mating = 4
            parents = selection(new_population, fitness, num_parents_mating)

            # Generating next generation using reproduction.
            offspring_size = (sol_per_pop - parents.shape[0], num_weights)
            offspring_reproduction = reproduction(parents, offspring_size)

            # Adding some variations to the offspring using mutation.
            offspring_mutation = mutation(offspring_reproduction, self.generation_fx, self.mutation_rate)

            # Creating the new population based on the parents and offspring.
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation

        # Getting the best solution after iterating finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        # fitness = self.fitness_fx(bit_string, new_population)
        # Then return the index of that solution corresponding to the best fitness.
        # best_match_idx = np.where(fitness == np.max(fitness))

        # print("Expected solution : ", expected_string)
        # print("Best solution(s) : ", new_population[best_match_idx, :])
        # print("Best solution fitness : ", fitness[best_match_idx])
        return generation_max_fitness, best_outputs, mean_outputs, worst_outputs


MIN_INT = -2147483648


def evaluate_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calculates the sum of scores whether if each bit is equal or not.
    score = (pop == equation_inputs).astype(int)
    fitness = np.sum(score, axis=1)
    return fitness


def selection(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents
    # for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = MIN_INT
    return parents


def reproduction(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which reproduction takes place between two parents. Usually, it is a random point
    reproduction_point = np.random.randint(offspring_size[1])

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:reproduction_point] = parents[parent1_idx, 0:reproduction_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, reproduction_point:] = parents[parent2_idx, reproduction_point:]
    return offspring


def mutation(offspring_reproduction, generator, mutation_rate):
    # Mutation changes a number of genes as defined by the mutation_rate argument. The changes are random.
    for idx in range(offspring_reproduction.shape[0]):
        for gene_idx in range(offspring_reproduction.shape[1]):
            if np.random.uniform() < mutation_rate:
                offspring_reproduction[idx, gene_idx] = generator()
    return offspring_reproduction


def generator():
    return np.random.randint(2)


def plot(best_outputs, mean_outputs, worst_outputs):
    plt.plot(best_outputs, label='Best fitness')
    plt.plot(mean_outputs, label='Mean fitness')
    plt.plot(worst_outputs, label='Worst fitness')
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()


def plot_heatmap(a):
    fig, ax = plt.subplots()
    im = plt.imshow(a, cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Gen of max fitness", rotation=-90, va="bottom")
    plt.xlabel("Population")
    plt.ylabel("Mutation Rate")
    ax.set_xticks(np.arange(len(pop_amt)))
    ax.set_yticks(np.arange(len(mut_rate)))
    ax.set_xticklabels(pop_amt)
    ax.set_yticklabels(mut_rate)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(mut_rate)):
        for j in range(len(pop_amt)):
            text = ax.text(j, i, a[i, j], ha="center", va="center", color="w")
    ax.set_title("Heatmap of generations of max fitness achieved")
    fig.tight_layout()
    plt.show()


bit_string = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])

# Defining the population size.
# The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
pop_amt = [x for x in range(50, 1050, 50)]
num_weights = 14
mut_rate = [x / 10.0 for x in range(11)]
num_generations = 20

generation_reached_fitness = []
for mutation_rate in mut_rate:
    generation_reached_fitness_pop = []
    for sol_per_pop in pop_amt:
        pop_size = (sol_per_pop, num_weights)
        genome = Genome(pop_size, evaluate_fitness, generator, mutation_rate, num_generations)
        gen_best, best_outputs, mean_outputs, worst_outputs = genome.start()
        generation_reached_fitness_pop.append(gen_best)
    generation_reached_fitness.append(generation_reached_fitness_pop)
    print("Progress: " + str(mutation_rate*100) + "%")
print("Plotting fitness for mutation_rate=1 and population=1000")
plot(best_outputs, mean_outputs, worst_outputs)
print("Plotting heatmap")
plot_heatmap(np.array(generation_reached_fitness))
