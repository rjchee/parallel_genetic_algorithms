#ifndef __GA_H__
#define __GA_H__

typedef struct {
    char val;
} gene_t;

typedef struct {
    int fitness;
    int geneIdx;
} chromosome_t;

typedef struct {
    double mutationProb;
    int numChromosomes;
    int genesPerChromosome;
    chromosome_t *chromosomes;
    gene_t *genes;
} population_t;

#endif
