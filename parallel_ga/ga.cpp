#include "ga.h"
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "CycleTimer.h"

#define DEBUG false

#define POPULATION_SIZE 10000
#define NUM_GENES 100
#define MUTATION_PROB 0.05
#define NUM_TRIALS 5
#define NUM_GENERATIONS 100

static population_t * initPopulation();
static void printPopulation(population_t * population);
static bool converged(population_t *population);
static int evaluate(population_t * population);
static int evaluateFitness(chromosome_t *chromo);
static void generateOffsprings(population_t * population, population_t * buffer);
static void crossover(population_t * population, population_t * buffer, int index, int p1, int p2);
static int *generateRoulette(population_t * population);
static int rouletteSelect(int * roulette, int n);
static void cleanupPopulation(population_t *population);


int main(int argc, const char *argv[]) {
    srand(time(NULL));
    double minTime = 1e30;
    for (int i = 0; i < NUM_TRIALS; i++) {
        population_t * population = initPopulation();
        printPopulation(population);
        population_t * buffer = initPopulation();
        evaluate(population);

        double startTime = CycleTimer::currentSeconds();
        int generation;
        for (generation = 0; generation < NUM_GENERATIONS; generation++) {
            printPopulation(population);
            generateOffsprings(population, buffer);
            if (converged(population)) {
                printPopulation(population);
                break;
            }
            // printf("generation #%d\n", index);
            printPopulation(population);
        }
        double endTime = CycleTimer::currentSeconds();
        double totalTime = endTime - startTime;
        int fitness = evaluate(population);
        if (generation < NUM_GENERATIONS) {
            printf("converge at generation #%d\n", generation);
        }
        printf("trial %d: %.4f seconds, fitness: %d\n", i, totalTime, fitness);
        minTime = std::min(minTime, totalTime);

        cleanupPopulation(population);
        cleanupPopulation(buffer);
    }

    printf("minimum time: %.4f seconds\n", minTime);
    return 0;
}


static int evaluate(population_t * population) {
    int sum = 0;
    for (int i = 0; i < population->size; i++) {
        sum += evaluateFitness(&(population->chromosomes[i]));
    }
    return sum;
}

static bool converged(population_t *population) {
    return evaluate(population) == POPULATION_SIZE * NUM_GENES;
}

static int evaluateFitness(chromosome_t *chromo) {
    int val = 0;
    for (int i = 0; i < chromo->numOfGenes; i++) {
        val += chromo->genes[i].val;
    }
    chromo->fitness = val;
    return val;
}

static void generateOffsprings(population_t * population, population_t * buffer) {
    int * roulette = generateRoulette(population);
    // printf("generated roulette\n");
    for (int i = 0; i < population->size; i += 2) {
        int parent1 = rouletteSelect(roulette, population->size + 1);
        int parent2 = rouletteSelect(roulette, population->size + 1);
        // printf("crossover %d & %d to generate %d & %d\n", parent1, parent2, i, i + 1);
        crossover(population, buffer, i, parent1, parent2);
    }

    free(roulette);

    population_t * tmp = population;
    population = buffer;
    buffer = tmp;
}

static void crossover(population_t * population, population_t * buffer, int index, int p1, int p2) {
    chromosome_t * parent1 = &(population->chromosomes[p1]);
    chromosome_t * parent2 = &(population->chromosomes[p2]);
    chromosome_t * child1 = &(population->chromosomes[index]);
    chromosome_t * child2 = &(population->chromosomes[index + 1]);

    int val = rand() / parent1->numOfGenes;
    for (int i = 0; i < parent1->numOfGenes; i++) {
        if (i < val) {
            child1->genes[i].val = parent1->genes[i].val;
            child2->genes[i].val = parent2->genes[i].val;
        } else {
            child1->genes[i].val = parent2->genes[i].val;
            child2->genes[i].val = parent1->genes[i].val;
        }

        double r = (double) rand() / ((double) RAND_MAX + 1.0);
        if (r < population->mutationProb) {
            child1->genes[i].val ^= 1;
        }

        r = (double) rand() / ((double) RAND_MAX + 1.0);
        if (r < population->mutationProb) {
            child2->genes[i].val ^= 1;
        }
    }
}

static int *generateRoulette(population_t * population) {
    int * roulette = (int *) malloc(sizeof(int) * (population->size + 1));

    roulette[0] = 0;
    for (int i = 1; i <= population->size; i++) {
        roulette[i] = roulette[i - 1] + population->chromosomes[i - 1].fitness;
        // printf("roulette: %d: %d, %d\n", i, roulette[i], (population->chromosomes)[i - 1].fitness);
    }
    return roulette;
}

static int rouletteSelect(int * roulette, int n) {
    int val = rand() % roulette[n - 1];

    for (int i = 0; i < n - 1; i++) {
        if (val < roulette[i + 1]) {
            return i;
        }
    }

    return -1;
}


static population_t * initPopulation() {
    population_t * population = (population_t *) malloc(sizeof(population_t));
    population->size = POPULATION_SIZE;
    population->mutationProb = MUTATION_PROB;
    population->chromosomes = (chromosome_t *) malloc(sizeof(chromosome_t) * POPULATION_SIZE);

    chromosome_t *chromos = population->chromosomes;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        chromos[i].numOfGenes = NUM_GENES;
        chromos[i].genes = (gene_t *) malloc(sizeof(gene_t) * NUM_GENES);
        for (int j = 0; j < NUM_GENES; j++) {
            chromos[i].genes[j].val = rand() % 2;
        }
        chromos[i].fitness = 0;
    }
    return population;
}


static void cleanupPopulation(population_t *population) {
    for (int i = 0; i < POPULATION_SIZE; i++) {
        free(population->chromosomes[i].genes);
    }
    free(population->chromosomes);
    free(population);
}


static void printPopulation(population_t * population) {
    if (DEBUG) {
        chromosome_t *chromos = population->chromosomes;
        for (int i = 0; i < POPULATION_SIZE; i++) {
            printf("chromosome %d: [", i);
            for (int j = 0; j < NUM_GENES; j++) {
                printf("%d, ", chromos[i].genes[j].val);
            }
            printf("] fitness: %d\n", chromos[i].fitness);
        }
    }
}
