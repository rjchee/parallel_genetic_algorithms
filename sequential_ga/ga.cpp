#include "ga.h"
#include <algorithm>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "CycleTimer.h"

static bool debug = false;
static size_t population_size = 10000;
static size_t num_genes = 100;
static float mutation_prob = 0.05;
static int num_trials = 5;
static int num_generations = 100;

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

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -p  --populationsize <INT>   Number of members of the population\n");
    printf("  -g  --numgenes <INT>         Number of genes each member of the population has\n");
    printf("  -m  --mutationprob <DOUBLE>  Probability for a gene to mutate\n");
    printf("  -t  --numtrials <INT>        Number of times to run the genetic algorithm\n");
    printf("  -n  --numgenerations <INT>   Max number of generations to grow\n");
    printf("  -?  --help                   This message\n");
}


int main(int argc, char *argv[]) {
    int opt;
    static struct option long_options[] = {
        {"populationsize", 1, NULL, 'p'},
        {"numgenes", 1, NULL, 'g'},
        {"mutationprob", 1, NULL, 'm'},
        {"numtrials", 1, NULL, 't'},
        {"numgenerations", 1, NULL, 'n'},
        {"debug", 0, NULL, 'd'},
        {"help", 0, NULL, '?'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?p:g:m:t:n:d", long_options, NULL)) != EOF) {
        switch (opt) {
            case 'p':
                population_size = atoi(optarg);
                if (population_size % 2 != 0) {
                    printf("population size must be even!\n");
                    return 1;
                }
                if (population_size <= 0) {
                    printf("population size must be positive!\n");
                }
                break;
            case 'g':
                num_genes = atoi(optarg);
                if (num_genes <= 0) {
                    printf("population size must be positive!\n");
                }
                break;
            case 'm':
                mutation_prob = atof(optarg);
                if (mutation_prob < 0 || mutation_prob > 1) {
                    printf("mutation probability must be a valid probability!\n");
                }
                break;
            case 't':
                num_trials = atoi(optarg);
                if (num_trials <= 0) {
                    printf("number of trials must be positive!\n");
                }
                break;
            case 'n':
                num_generations = atoi(optarg);
                break;
            case 'd':
                debug = true;
                break;
            case '?':
                usage(argv[0]);
                return 0;
        }
    }

    srand(time(NULL));
    double minTime = 1e30;
    for (int i = 0; i < num_trials; i++) {
        population_t * population = initPopulation();
        printPopulation(population);
        population_t * buffer = initPopulation();
        evaluate(population);

        double startTime = CycleTimer::currentSeconds();
        int generation;
        for (generation = 0; generation != num_generations; generation++) {
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
        if (generation != num_generations) {
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
    return evaluate(population) == (int)(population_size * num_genes);
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
    population->size = population_size;
    population->mutationProb = mutation_prob;
    population->chromosomes = (chromosome_t *) malloc(sizeof(chromosome_t) * population_size);

    chromosome_t *chromos = population->chromosomes;
    for (size_t i = 0; i < population_size; i++) {
        chromos[i].numOfGenes = num_genes;
        chromos[i].genes = (gene_t *) malloc(sizeof(gene_t) * num_genes);
        for (size_t j = 0; j < num_genes; j++) {
            chromos[i].genes[j].val = rand() % 2;
        }
        chromos[i].fitness = 0;
    }
    return population;
}


static void cleanupPopulation(population_t *population) {
    for (size_t i = 0; i < population_size; i++) {
        free(population->chromosomes[i].genes);
    }
    free(population->chromosomes);
    free(population);
}


static void printPopulation(population_t * population) {
    if (debug) {
        chromosome_t *chromos = population->chromosomes;
        for (size_t i = 0; i < population_size; i++) {
            printf("chromosome %lu: [", i);
            for (size_t j = 0; j < num_genes; j++) {
                printf("%d, ", chromos[i].genes[j].val);
            }
            printf("] fitness: %d\n", chromos[i].fitness);
        }
    }
}
