#include "ga.h"
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define NUM_GENERATIONS 100

#define THREADS_PER_BLOCK 256

// TODO: check if buffer's mutationProb, numChromosomes, and numGenes are initialized correctly
// TODO: store copies of heavily accessed shared memory to local memory

static void cudaInitPopulation(population_t *hostPopulation, population_t);
__device__ bool converged(population_t *population);
__device__ int evaluate(population_t *population);
__device__ int evaluateFitness(int threadID, population_t *population, int chromoIdx);
__device__ void generateOffsprings(int threadID, curandState_t *state, population_t * population, population_t * buffer, int *roulette);
__device__ void crossover(curandState_t *state, population_t *population, population_t *buffer, int index, int p1, int p2);
__device__ void generateRoulette(int threadID, population_t * population, int *roulette);
__device__ int rouletteSelect(curandState_t *state, int * roulette, int n);
__global__ void gaKernel(curandState_t *states, population_t population, population_t buffer, int *roulette);

static void cudaInitPopulation(population_t *hostPopulation, population_t *cudaPopulation) {
    size_t chromosomeBytes = hostPopulation->numChromosomes * sizeof(chromosome_t);
    cudaMalloc(&cudaPopulation->chromosomes, chromosomeBytes);
    size_t geneBytes = hostPopulation->numGenes * sizeof(gene_t);
    cudaMalloc(&cudaPopulation->genes, geneBytes);
    cudaMemcpy(cudaPopulation->chromosomes, hostPopulation->chromosomes, chromosomeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPopulation->genes, hostPopulation->genes, geneBytes, cudaMemcpyHostToDevice);
}

__device__ int evaluate(int threadID, population_t *population) {
    int chromosomesPerThread = (population->numChromosomes + blockDim.x * THREADS_PER_BLOCK - 1) / (blockDim.x * THREADS_PER_BLOCK);
    int startIdx = threadID  * chromosomesPerThread;
    int endIdx = startIdx + chromosomesPerThread;

    for (int i = startIdx; i < endIdx; i++) {
        evaluateFitness(threadID, population, i);
    }
    __syncthreads();
    int sum = 0;
    for (int i = 0; i < population->numChromosomes; i++) {
        sum += population->chromosomes[i].fitness;
    }
    return sum;
}


__device__ bool converged(int threadID, population_t *population) {
    int totalFitness = evaluate(threadID, population);
    population->totalFitness = totalFitness;
    return totalFitness == population->numChromosomes * population->numGenes;
}


__device__ int evaluateFitness(int threadID, population_t *population, int chromoIdx) {
    chromosome_t *chromosome = &population->chromosomes[chromoIdx];
    int startGene = chromosome->geneIdx;
    int endGene = startGene + chromosome->numOfGenes;
    int val = 0;
    for (int i = startGene; i < endGene; i++) {
        val += population->genes[i].val;
    }
    chromosome->fitness = val;
    return val;
}


__device__ void generateOffsprings(int threadID, curandState_t *state, population_t *population, population_t *buffer, int *roulette) {
    if (threadID == 0) {
        generateRoulette(threadID, population, roulette);
    }
    __syncthreads();
    // printf("generated roulette\n");
    int iterationsPerThread = (population->numChromosomes / 2 + blockDim.x* THREADS_PER_BLOCK - 1) / (blockDim.x * THREADS_PER_BLOCK);
    int startIdx = threadID * iterationsPerThread;
    int endIdx = startIdx + iterationsPerThread;
    for (int i = startIdx; i < endIdx; i++) {
        int parent1 = rouletteSelect(state, roulette, population->numChromosomes);
        int parent2 = rouletteSelect(state, roulette, population->numChromosomes);
        // printf("crossover %d & %d to generate %d & %d\n", parent1, parent2, i, i + 1);
        crossover(state, population, buffer, i * 2, parent1, parent2);
    }

    __syncthreads();
    if (threadID == 0) {
        population_t * tmp = population;
        population = buffer;
        buffer = tmp;
    }
}


__device__ void crossover(curandState_t *state, population_t *population, population_t *buffer, int index, int p1, int p2) {
    chromosome_t *parent1 = &(population->chromosomes[p1]);
    chromosome_t *parent2 = &(population->chromosomes[p2]);
    chromosome_t *child1 = &(buffer->chromosomes[index]);
    chromosome_t *child2 = &(buffer->chromosomes[index + 1]);

    int val = (int)(curand_uniform(state) * parent1->numOfGenes);
    for (int i = 0; i < parent1->numOfGenes; i++) {
        if (i < val) {
            child1->genes[i].val = parent1->genes[i].val;
            child2->genes[i].val = parent2->genes[i].val;
        } else {
            child1->genes[i].val = parent2->genes[i].val;
            child2->genes[i].val = parent1->genes[i].val;
        }

        double r = (double) curand_uniform(state);
        if (r < population->mutationProb) {
            child1->genes[i].val ^= 1;
        }

        r = (double) curand_uniform(state);
        if (r < population->mutationProb) {
            child2->genes[i].val ^= 1;
        }
    }
}


__device__ void generateRoulette(int threadID, population_t * population, int *roulette) {
    roulette[0] = 0;
    for (int i = 1; i <= population->numChromosomes; i++) {
        roulette[i] = roulette[i - 1] + population->chromosomes[i - 1].fitness;
        // printf("roulette: %d: %d, %d\n", i, roulette[i], (population->chromosomes)[i - 1].fitness);
    }
}


__device__ int rouletteSelect(curandState_t *state, int *roulette, int size) {
    int val = (int)(curand_uniform(state) * roulette[size]);

    for (int i = 0; i < size; i++) {
        if (val < roulette[i + 1]) {
            return i;
        }
    }

    return -1;
}

__global__ void gaKernel(curandState_t *states, population_t population, population_t buffer, int *roulette) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    curandState_t threadState = states[threadID];
    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        generateOffsprings(threadID, &threadState, &population, &buffer, roulette);
        if (converged(threadID, population)) {
            break;
        }
        __syncthreads();
    }
}


__global__ void setupCurand(curandState *state, unsigned long long seed_offset) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(id + seed_offset, 0, 0, &state[id]);
}


void gaCuda(population_t *population, population_t *buffer) {
    const int blocks = (population->numChromosomes + NUM_THREADS - 1) / NUM_THREADS;

    population_t cudaPopulation;
    population_t cudaBuffer;
    cudaInitPopulation(population, &cudaPopulation);
    cudaInitPopulation(buffer, &cudaBuffer);

    // array holding the integers used in the roulette function used in
    // generating new offspring
    int *cudaRoulette;
    int rouletteBytes = (population->numChromosomes + 1) * sizeof(int);
    cudaMalloc(&cudaRoulette, rouletteBytes);

    curandState_t *states;
    int curandStateBytes = blockDim.x * THREADS_PER_BLOCK * sizeof(curandState_t);
    cudaMalloc(&states, curandStateBytes);
    setupCurand<<blocks, THREADS_PER_BLOCK>>(states, CycleTimer::currentTicks());
    cudaThreadSynchronize();

    int *cudaResult;
    cudaMalloc(&cudaResult, sizeof(int));

    double startTime = CycleTimer::currentSeconds();

    gaKernel<<blocks, THREADS_PER_BLOCKS>>(states, cudaPopulation, cudaBuffer, cudaRoulette);
    cudaThreadSynchronize();
    int totalFitness;
    cudaMemcpy(cudaResult, &totalFitness, sizeof(int), cudaMemcpyDeviceToHost);

    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double duration = endTime - startTime;
    printf("Overall time: %.4fs, fitness: %d\n", duration, totalFitness);

    cudaFree(cudaResult);
    cudaFree(states);
    cudaFree(cudaPopulation.chromosomes);
    cudaFree(cudaPopulation.genes);
}
