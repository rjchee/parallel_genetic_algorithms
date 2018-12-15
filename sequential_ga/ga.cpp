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
static int evaluateFitness(population_t *population, int chromoIdx);
static void generateOffsprings(population_t * population, population_t * buffer, int *roulette);
static void crossover(population_t * population, population_t * buffer, int index, int p1, int p2);
static void generateRoulette(population_t * population, int *roulette);
static int rouletteSelect(int * roulette, int size);
static void cleanupPopulation(population_t *population);

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -p  --populationsize <INT>   Number of members of the population\n");
    printf("  -g  --numgenes <INT>         Number of genes each member of the population has\n");
    printf("  -m  --mutationprob <DOUBLE>  Probability for a gene to mutate\n");
    printf("  -t  --numtrials <INT>        Number of times to run the genetic algorithm\n");
    printf("  -n  --numgenerations <INT>   Max number of generations to grow\n");
    printf("  -d                           Turn on debug prints\n");
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
        int *roulette = (int *)malloc((population->numChromosomes + 1) *sizeof(int));

        double startTime = CycleTimer::currentSeconds();
        int generation;
        for (generation = 0; generation != num_generations; generation++) {
            printPopulation(population);
            generateOffsprings(population, buffer, roulette);
            population_t *tmp = population;
            population = buffer;
            buffer = tmp;
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

        free(roulette);
        cleanupPopulation(population);
        cleanupPopulation(buffer);
    }

    printf("minimum time: %.4f seconds\n", minTime);
    return 0;
}


static int evaluate(population_t * population) {
    int sum = 0;
    for (int i = 0; i < population->numChromosomes; i++) {
        sum += evaluateFitness(population, i);
    }
    return sum;
}

static bool converged(population_t *population) {
    return evaluate(population) == (int)(population_size * num_genes);
}

static int evaluateFitness(population_t *population, int chromoIdx) {
    int val = 0;
    int startIdx = population->chromosomes[chromoIdx].geneIdx;
    int endIdx = startIdx + population->genesPerChromosome;
    for (int i = startIdx; i < endIdx; i++) {
        val += population->genes[i].val;
    }
    population->chromosomes[chromoIdx].fitness = val;
    return val;
}

static void generateOffsprings(population_t * population, population_t * buffer, int *roulette) {
    generateRoulette(population, roulette);
    // printf("generated roulette\n");
    for (int i = 0; i < population->numChromosomes; i += 2) {
        int parent1 = rouletteSelect(roulette, population->numChromosomes);
        int parent2 = rouletteSelect(roulette, population->numChromosomes);
        // printf("crossover %d & %d to generate %d & %d\n", parent1, parent2, i, i + 1);
        crossover(population, buffer, i, parent1, parent2);
    }

    free(roulette);
}

static void crossover(population_t * population, population_t * buffer, int index, int pc1, int pc2) {
    chromosome_t * parent1 = &population->chromosomes[pc1];
    chromosome_t * parent2 = &population->chromosomes[pc2];
    chromosome_t * child1 = &buffer->chromosomes[index];
    chromosome_t * child2 = &buffer->chromosomes[index + 1];

    int crossoverIdx = rand() % population->genesPerChromosome;
    int c1 = child1->geneIdx;
    int c2 = child2->geneIdx;
    int p1 = parent1->geneIdx;
    int p2 = parent2->geneIdx;
    for (int i = 0; i < population->genesPerChromosome; i++, c1++, c2++, p1++, p2++) {
        if (i < crossoverIdx) {
            buffer->genes[c1].val = population->genes[p1].val;
            buffer->genes[c2].val = population->genes[p2].val;
        } else {
            buffer->genes[c1].val = population->genes[p2].val;
            buffer->genes[c2].val = population->genes[p1].val;
        }

        double r = (double) rand() / ((double) RAND_MAX + 1.0);
        if (r < population->mutationProb) {
            buffer->genes[c1].val ^= 1;
        }

        r = (double) rand() / ((double) RAND_MAX + 1.0);
        if (r < population->mutationProb) {
            buffer->genes[c2].val ^= 1;
        }
    }
}

static void generateRoulette(population_t * population, int *roulette) {
    roulette[0] = 0;
    for (int i = 1; i <= population->numChromosomes; i++) {
        roulette[i] = roulette[i - 1] + population->chromosomes[i - 1].fitness;
        // printf("roulette: %d: %d, %d\n", i, roulette[i], (population->chromosomes)[i - 1].fitness);
    }
}

static int rouletteSelect(int * roulette, int size) {
    int val = rand() % roulette[size];

    for (int i = 0; i < size; i++) {
        if (val < roulette[i + 1]) {
            return i;
        }
    }

    return -1;
}


static population_t *initPopulation() {
    population_t *population = (population_t *) malloc(sizeof(population_t));
    population->numChromosomes = population_size;
    population->genesPerChromosome = num_genes;
    population->mutationProb = mutation_prob;
    population->chromosomes = (chromosome_t *) malloc(sizeof(chromosome_t) * population->numChromosomes);
    int totalNumGenes = population->numChromosomes * population->genesPerChromosome;
    population->genes = (gene_t *) malloc(sizeof(gene_t) * totalNumGenes);

    chromosome_t *chromos = population->chromosomes;
    for (int i = 0; i < population->numChromosomes; i++) {
        chromos[i].geneIdx = i * population->genesPerChromosome;
        gene_t *genes = &population->genes[chromos[i].geneIdx];
        for (int j = 0; j < population->genesPerChromosome; j++) {
            genes[j].val = rand() % 2;
        }
        chromos[i].fitness = 0;
    }
    return population;
}


static void cleanupPopulation(population_t *population) {
    free(population->genes);
    free(population->chromosomes);
    free(population);
}


static void printPopulation(population_t * population) {
    if (debug) {
        chromosome_t *chromos = population->chromosomes;
        gene_t *genes = population->genes;
        size_t numChromosomes = population->numChromosomes;
        size_t genesPerChromosome = population->genesPerChromosome;
        for (size_t i = 0; i < numChromosomes; i++) {
            size_t startGene = chromos[i].geneIdx;
            size_t endGene = startGene + genesPerChromosome;
            printf("chromosome %lu: fitness: %d [", i, chromos[i].fitness);
            for (size_t j = startGene; j < endGene; j++) {
                printf("%d, ", genes[j].val);
            }
            printf("]\n");
        }
    }
}
