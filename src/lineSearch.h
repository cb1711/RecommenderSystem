#ifndef _lineSearch_h
#define _lineSearch_h


#define SIGMA 1.0
#define BETA 1.0
#define LAMBDA 1.0
#define CLUSTERS 20
/*Returns the inner product of the given arrays A and B of size size*/
float innerProduct(float* A,float *B,int size);

/*Returns value of likelihood*/
void likelihood(float *Q,bool *selected,float *user_sum,float **items,float **users,int numItems,int* item_sparse_csr_r,int *user_sparse_csr_c,int *allotted,int totalItems);


/*Function for linesearch */
void linesearch(float **items, float *user_sum, float**users, float **gradient, int numItems, int *allotted, int totalItems, int* item_sparse_csr_r, int *user_sparse_csr_c);

#endif
