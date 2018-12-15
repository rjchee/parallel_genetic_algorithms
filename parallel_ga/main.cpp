#include <stdio.h>
#include <stdlib.h>

#include "CycleTimer.h"
#include "ga.h"

#define NUM_TRIALS 5
#define POPULATION_SIZE 10000
#define NUM_GENES 100
#define MUTATION_PROB 0.05

static population_t* initPopulation();
static void cleanupPopulation(population_t *population);

void gaCuda(population_t *population, population_t *buffer);


int main(int argc, const char *argv[]) {
    srand(CycleTimer::currentTicks());
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        population_t *population = initPopulation();
        population_t *buffer = initPopulation();

        gaCuda(population, buffer);

        cleanupPopulation(population);
        cleanupPopulation(buffer);
    }
}


static population_t * initPopulation() {
    population_t * population = (population_t *) malloc(sizeof(population_t));
    population->numChromosomes = POPULATION_SIZE;
    population->genesPerChromosome = NUM_GENES;
    population->mutationProb = MUTATION_PROB;
    population->chromosomes = (chromosome_t *) malloc(sizeof(chromosome_t) * population->numChromosomes);
    population->genes = (gene_t *) malloc(sizeof(gene_t) * population->numChromosomes * population->genesPerChromosome);

    chromosome_t *chromos = population->chromosomes;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        chromos[i].geneIdx = i * NUM_GENES;
        gene_t *genes = &population->genes[chromos[i].geneIdx];
        for (int j = 0; j < NUM_GENES; j++) {
            genes[j].val = rand() % 2;
        }
        chromos[i].fitness = 0;
    }
    return population;
}


static void cleanupPopulation(population_t *population) {
    free(population->chromosomes);
    free(population->genes);
    free(population);
}
