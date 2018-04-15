#ifndef _lineSearch_h
#define _lineSearch_h


#define sigma 1.0
#define beta 1.0
#define lambda 1.0
#define k 20
/*Returns the inner product of the given arrays A and B of size size*/
float innerProduct(float* A,float *B,int size);

/*Returns value of likelihood*/
void likelihood(float *Q,bool *selected,float *user_sum,float **items,float **users,int numItems,int* item_sparse_csr_r,int *user_sparse_csr_c,int *allotted,int totalItems);


/*Function for linesearch */
void linesearch(float **items, float *user_sum, float**users, float **gradient, int numItems, int *allotted, int totalItems, int* item_sparse_csr_r, int *user_sparse_csr_c,float lambda);

#endif
