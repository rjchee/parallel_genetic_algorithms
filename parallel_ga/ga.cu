#include "ga.h"
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define NUM_GENERATIONS 100

__global__ void ga_eval_kernel();

static population_t *cudaInitPopulation(population_t *hostPopulation);
static bool converged(population_t *population);
static int evaluate(population_t * population);
static int evaluateFitness(chromosome_t *chromo);
static void generateOffsprings(population_t * population, population_t * buffer);
static void crossover(population_t * population, population_t * buffer, int index, int p1, int p2);
static int *generateRoulette(population_t * population);
static int rouletteSelect(int * roulette, int n);
static bool processGeneration(population_t *population, population_t *buffer);

static population_t *cudaInitPopulation(population_t *hostPopulation) {
    population_t *cudaPopulation;
    size_t populationBytes = sizeof(population_t);
    cudaMalloc(&cudaPopulation, populationBytes);
    chromosome_t *cudaChromosomes;
    size_t chromosomeBytes = hostPopulation->numChromosomes * sizeof(chromosome_t);
    cudaMalloc(&cudaChromosomes, chromosomeBytes);
    gene_t *cudaGenes;
    size_t geneBytes = hostPopulation->numGenes * sizeof(gene_t);
    cudaMalloc(&cudaGenes, geneBytes);
    population_t copy = *hostPopulation;
    copy.chromosomes = cudaChromosomes;
    copy.genes = cudaGenes;
    cudaMemcpy(cudaPopulation, &copy, populationBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaChromosomes, hostPopulation->chromosomes, chromosomeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaGenes, hostPopulation->genes, geneBytes, cudaMemcpyHostToDevice);

    return cudaPopulation;
}

static int evaluate(population_t * population) {
    int sum = 0;
    for (int i = 0; i < population->numChromosomes; i++) {
        sum += evaluateFitness(&population->chromosomes[i]);
    }
    return sum;
}


static bool converged(population_t *population) {
    int totalFitness = evaluate(population);
    return totalFitness == POPULATION_SIZE * NUM_GENES;
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


static bool processGeneration(population_t *population, population_t *buffer) {
    generateOffsprings(population, buffer);
    return converged(population);
}


void gaCuda(population_t *population, population_t *buffer) {
    const int threadsPerBlock = 256;
    const int blocks = (population->size + threadsPerBlock - 1) / threadsPerBlock;

    population_t *cudaPopulation;
    population_t *cudaBuffer;
    size_t populationBytes = sizeof(population_t);
    cudaMalloc(&cudaPopulation, populationBytes);
    cudaMalloc(&cudaBuffer, populationBytes);
    size_t chromosomeBytes = population->size * sizeof(chromosome_t);
    cudaMalloc(&cuda_population->chromosomes, chromosomeBytes);
    cudaMalloc(&cuda_buffer->chromosomes, chromosomeBytes);
    for (int i = 0; i < population->size; i++) {
        size_t geneBytes = population->chromosomes[i].numOfGenes * sizeof(gene_t);
        cudaMalloc(&cuda_population[i].genes, geneBytes);
        cudaMalloc(&cuda_buffer[i].genes, geneBytes);
    }
    size_t resultBytes = population->size * sizeof(int);

    double startTime = CycleTimer::currentSeconds();

    cudaMemcpy(cuda_population, population->chromosomes, chromosomeBytes, cudaMemcpyHostToDevice);
    for (int i = 0; i < population->size; i++) {
        size_t geneBytes = population->chromosomes[i].numOfGenes * sizeof(gene_t);
        cudaMemcpy(cuda_population[i].genes, population->chromosomes[i].genes, geneBytes, cudaMemcpyHostToDevice);
        // don't copy buffer unnecessarily
    }

    int generation;
    int totalFitness;

    int generation;
    for (generation = 0; generation < NUM_GENERATIONS; generation++) {
        bool exit = processGeneration(population, buffer);
        if (exit) {
            break;
        }
    }
    int totalFitness = evaluate(population);

    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double duration = endTime - startTime;
    printf("Overall time: %.4fs, fitness: %d\n", duration, totalFitness);

    for (int i = 0; i < population->size; i++) {
        cudaFree(cuda_population[i].genes);
        cudaFree(cuda_buffer[i].genes);
    }
    cudaFree(cuda_population);
    cudaFree(cuda_buffer);
}
