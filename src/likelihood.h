#ifndef _likelihood_h
#define _likelihood_h

/*Returns the inner product of the given arrays A and B of size size*/
float innerProduct(float* A,float *B,int size);

/*Returns value of likelihood*/
void likelihood_item(float *Q,bool *selected,float *user_sum,float **items,float **users,int k,float lambda,int numItems,int* item_sparse_csr_r,int *user_sparse_csr_c,int *allotted,int totalItems);

#endif