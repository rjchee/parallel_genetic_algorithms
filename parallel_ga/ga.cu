#include "ga.h"
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 512

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

// TODO: check if buffer's mutationProb, numChromosomes, and numGenes are initialized correctly
// TODO: store copies of heavily accessed shared memory to local memory

__device__ void printPopulation(population_t *population);
static population_t *cudaInitPopulation(population_t *hostPopulation);
__device__ bool converged(int threadID, population_t *population, bool debug);
__device__ int evaluate(population_t *population);
__device__ int evaluateFitness(int threadID, population_t *population, int chromoIdx);
__device__ void generateOffsprings(int threadID, curandState_t *state, population_t * population, population_t * buffer, int *roulette);
__device__ void crossover(curandState_t *state, population_t *population, population_t *buffer, int index, int pc1, int pc2);
__device__ void generateRoulette(int threadID, population_t * population, int *roulette);
__device__ int rouletteSelect(curandState_t *state, int * roulette, int n);
__global__ void gaKernel(curandState_t *states, population_t *population, population_t *buffer, int *roulette, int *totalFitness, int num_generations, bool debug);
__global__ void setupCurand(curandState_t *state, unsigned long long seed_offset);


__device__ void printPopulation(population_t * population) {
    chromosome_t *chromos = population->chromosomes;
    gene_t *genes = population->genes;
    int numChromosomes = population->numChromosomes;
    int genesPerChromosome = population->genesPerChromosome;
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

static population_t *cudaInitPopulation(population_t *hostPopulation) {
    population_t tmpPopulation = *hostPopulation;
    size_t chromosomeBytes = hostPopulation->numChromosomes * sizeof(chromosome_t);
    cudaCheckError( cudaMalloc(&tmpPopulation.chromosomes, chromosomeBytes) );
    size_t geneBytes = hostPopulation->numChromosomes * hostPopulation->genesPerChromosome * sizeof(gene_t);
    cudaCheckError( cudaMalloc(&tmpPopulation.genes, geneBytes) );
    cudaCheckError( cudaMemcpy(tmpPopulation.chromosomes, hostPopulation->chromosomes, chromosomeBytes, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(tmpPopulation.genes, hostPopulation->genes, geneBytes, cudaMemcpyHostToDevice) );

    population_t *cudaPopulation;
    cudaCheckError( cudaMalloc(&cudaPopulation, sizeof(population_t)) );
    cudaCheckError( cudaMemcpy(cudaPopulation, &tmpPopulation, sizeof(population_t), cudaMemcpyHostToDevice) );
    return cudaPopulation;
}

static void cudaFreePopulation(population_t *cudaPopulation) {
    population_t hostPopulation;
    cudaCheckError( cudaMemcpy(&hostPopulation, cudaPopulation, sizeof(population_t), cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaFree(hostPopulation.chromosomes) );
    cudaCheckError( cudaFree(hostPopulation.genes) );
    cudaCheckError( cudaFree(cudaPopulation) );
}

__device__ int evaluate(int threadID, population_t *population) {
    int chromosomesPerThread = (population->numChromosomes + THREADS_PER_BLOCK - 1) / (THREADS_PER_BLOCK);
    int startIdx = threadID  * chromosomesPerThread;
    int endIdx = startIdx + chromosomesPerThread;
    if (startIdx > population->numChromosomes) {
        startIdx = population->numChromosomes;
    }
    if (endIdx > population->numChromosomes) {
        endIdx = population->numChromosomes;
    }

    int sum = 0;
    for (int i = startIdx; i < endIdx; i++) {
        sum += evaluateFitness(threadID, population, i);
    }
    __shared__ int sums[THREADS_PER_BLOCK];
    sums[threadID] = sum;
    __syncthreads();
    for (int i = 0; i < THREADS_PER_BLOCK; i++) {
        if (i != threadID) {
            sum += sums[i];
        }
    }
    return sum;
}


__device__ bool converged(int threadID, population_t *population, bool debug) {
    int totalFitness = evaluate(threadID, population);
    if (debug) {
        printf("thread %d: fitness = %d\n", threadID, totalFitness);
    }
    population->totalFitness = totalFitness;
    return totalFitness == population->numChromosomes * population->genesPerChromosome;
}


__device__ int evaluateFitness(int threadID, population_t *population, int chromoIdx) {
    chromosome_t *chromosome = &population->chromosomes[chromoIdx];
    int startGene = chromosome->geneIdx;
    int endGene = startGene + population->genesPerChromosome;
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
    int iterationsPerThread = (population->numChromosomes / 2 + THREADS_PER_BLOCK - 1) / (THREADS_PER_BLOCK);
    int startIdx = threadID * iterationsPerThread;
    int endIdx = startIdx + iterationsPerThread;
    if (endIdx > population->numChromosomes / 2) {
        endIdx = population->numChromosomes / 2;
    }
    for (int i = startIdx; i < endIdx; i++) {
        int parent1 = rouletteSelect(state, roulette, population->numChromosomes);
        int parent2 = rouletteSelect(state, roulette, population->numChromosomes);
        // printf("crossover %d & %d to generate %d & %d\n", parent1, parent2, i, i + 1);
        crossover(state, population, buffer, i * 2, parent1, parent2);
    }
}


__device__ void crossover(curandState_t *state, population_t *population, population_t *buffer, int index, int pc1, int pc2) {
    chromosome_t *parent1 = &population->chromosomes[pc1];
    chromosome_t *parent2 = &population->chromosomes[pc2];
    chromosome_t *child1 = &buffer->chromosomes[index];
    chromosome_t *child2 = &buffer->chromosomes[index + 1];

    int genesPerChromosome = population->genesPerChromosome;
    int crossoverIdx = (int)(curand_uniform(state) * genesPerChromosome);
    int c1 = child1->geneIdx;
    int c2 = child2->geneIdx;
    int p1 = parent1->geneIdx;
    int p2 = parent2->geneIdx;
    for (int i = 0; i < genesPerChromosome; i++, c1++, c2++, p1++, p2++) {
        if (i < crossoverIdx) {
            buffer->genes[c1].val = population->genes[p1].val;
            buffer->genes[c2].val = population->genes[p2].val;
        } else {
            buffer->genes[c1].val = population->genes[p2].val;
            buffer->genes[c2].val = population->genes[p1].val;
        }

        double r = (double) curand_uniform(state);
        if (r < population->mutationProb) {
            buffer->genes[c1].val ^= 1;
        }

        r = (double) curand_uniform(state);
        if (r < population->mutationProb) {
            buffer->genes[c2].val ^= 1;
        }
    }
}

__host__ __device__ int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__device__ void generateRoulette(int threadID, population_t * population, int *roulette) {
    int numChromosomes = population->numChromosomes;
    int N = nextPow2(numChromosomes + 1);
    int numsPerThread = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int startIdx = threadID * numsPerThread;
    if (startIdx > N) {
        startIdx = N;
    }
    int endIdx = startIdx + numsPerThread;
    if (endIdx > N) {
        endIdx = N;
    }
    chromosome_t *chromosomes = population->chromosomes;
    if (threadID == 0) {
        roulette[0] = 0;
        for (int i = 1; i < endIdx; i++) {
            roulette[i] = chromosomes[i - 1].fitness;
        }
    } else if (endIdx >= numChromosomes) {
        for (int i = startIdx; i < endIdx; i++) {
            roulette[i] = chromosomes[i - 1].fitness;
        }
    } else if (startIdx + 1 >= numChromosomes) {
        for (int i = startIdx; i < endIdx; i++) {
            roulette[i] = 0;
        }
    } else {
        for (int i = startIdx; i < endIdx; i++) {
            if (i - 1 < numChromosomes) {
                roulette[i] = chromosomes[i - 1].fitness;
            } else {
                roulette[i] = 0;
            }
        }
    }
    __syncthreads();
    for (int twod = 1; twod < N; twod *= 2) {
        int twod1 = twod * 2;
        int idx = twod1 * threadID;
        if ((idx + twod1 - 1) < N) {
            roulette[idx + twod1 - 1] += roulette[idx + twod - 1];
        }
        __syncthreads();
    }
    for (int twod = N / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        int idx1 = twod1 * threadID + twod1 - 1;
        int idx2 = twod1 * threadID + twod - 1;
        if (idx1 < N) {
            int tmp = roulette[idx2];
            roulette[idx2] = roulette[idx1];
            roulette[idx1] += tmp;
        }
        __syncthreads();
    }
    /*
    roulette[0] = 0;
    for (int i = 1; i <= population->numChromosomes; i++) {
        roulette[i] = roulette[i - 1] + population->chromosomes[i - 1].fitness;
        // printf("roulette: %d: %d, %d\n", i, roulette[i], (population->chromosomes)[i - 1].fitness);
    }
    */
}


__device__ int rouletteSelect(curandState_t *state, int *roulette, int size) {
    int val = (int)(curand_uniform(state) * roulette[size]);

    for (int i = 0; i < size; i++) {
        if (val < roulette[i + 1]) {
            return i;
        }
    }

    return size - 1;
}

__global__ void gaKernel(curandState_t *states, population_t *population, population_t *buffer, int *roulette, int *totalFitness, int num_generations, bool debug) {
    int threadID = threadIdx.x;
    curandState_t threadState = states[threadID];
    for (int i = 0; i < num_generations; i++) {
        generateOffsprings(threadID, &threadState, population, buffer, roulette);
        population_t *tmp = population;
        population = buffer;
        buffer = tmp;
        __syncthreads();
        bool hasConverged = converged(threadID, population, debug);
        if (threadID == 0) {
            *totalFitness = population->totalFitness;
        }
        if (hasConverged) {
            printf("converged\n");
            break;
        }
        if (debug && threadID == 0) {
            printPopulation(population);
        }
    }
}


__global__ void setupCurand(curandState_t *state, unsigned long long seed_offset) {
    int id = threadIdx.x;
    curand_init(id + seed_offset, 0, 0, &state[id]);
}


void gaCuda(population_t *population, population_t *buffer, int num_generations, bool debug) {
    const int numThreads = THREADS_PER_BLOCK;

    population_t *cudaPopulation = cudaInitPopulation(population);
    population_t *cudaBuffer = cudaInitPopulation(buffer);

    // array holding the integers used in the roulette function used in
    // generating new offspring
    int *cudaRoulette;
    int rouletteBytes = nextPow2(population->numChromosomes + 1) * sizeof(int);
    cudaCheckError( cudaMalloc(&cudaRoulette, rouletteBytes) );

    curandState_t *states;
    int curandStateBytes = THREADS_PER_BLOCK * sizeof(curandState_t);
    cudaCheckError( cudaMalloc(&states, curandStateBytes) );
    unsigned long long seed = CycleTimer::currentTicks();
    setupCurand<<<1, numThreads>>>(states, seed);
    cudaCheckError( cudaThreadSynchronize() );

    int *cudaResult;
    cudaCheckError( cudaMalloc(&cudaResult, sizeof(int)) );


    double startTime = CycleTimer::currentSeconds();

    gaKernel<<<1, THREADS_PER_BLOCK>>>(states, cudaPopulation, cudaBuffer, cudaRoulette, cudaResult, num_generations, debug);
    cudaCheckError( cudaThreadSynchronize() );
    int totalFitness;
    cudaCheckError( cudaMemcpy(&totalFitness, cudaResult, sizeof(int), cudaMemcpyDeviceToHost) );

    double endTime = CycleTimer::currentSeconds();

    double duration = endTime - startTime;
    printf("Overall time: %.4fs, fitness: %d\n", duration, totalFitness);

    cudaCheckError( cudaFree(cudaResult) );
    cudaCheckError( cudaFree(states) );
    cudaFreePopulation(cudaPopulation);
    cudaFreePopulation(cudaBuffer);
}
