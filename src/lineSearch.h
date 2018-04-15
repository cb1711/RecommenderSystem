#ifndef _lineSearch_h
#define _lineSearch_h

/*Function for linesearch */

void linesearch(float **items, float *user_sum, float**users, int k, float **gradient, int numItems, int *allotted, int totalItems, int* item_sparse_csr_r, int *user_sparse_csr_c,float lambda)
