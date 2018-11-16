#ifndef __GA_H__
#define __GA_H__

typedef struct {
	char val;
} gene_t;

typedef struct {
	int numOfGenes;
	int fitness;
	gene_t * genes;
} chromosome_t;

typedef struct {
	double mutationProb;
	int size;
	chromosome_t * chromosomes;
} population_t;


#endif