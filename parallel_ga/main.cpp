#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "CycleTimer.h"
#include "ga.h"

static population_t* initPopulation();
static void cleanupPopulation(population_t *population);

void gaCuda(population_t *population, population_t *buffer, int num_generations, bool debug);

static bool debug = false;
static size_t population_size = 10000;
static size_t num_genes = 100;
static float mutation_prob = 0.05;
static int num_trials = 5;
static int num_generations = 100;

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

    srand(CycleTimer::currentTicks());
    for (int trial = 0; trial < num_trials; trial++) {
        printf("Starting trial %d\n", trial + 1);
        population_t *population = initPopulation();
        population_t *buffer = initPopulation();

        gaCuda(population, buffer, num_generations, debug);

        cleanupPopulation(population);
        cleanupPopulation(buffer);
    }
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
    free(population->chromosomes);
    free(population->genes);
    free(population);
}
