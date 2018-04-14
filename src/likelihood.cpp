/*
 * Evaluates likelihood function for recommender system
 */
#include <iostream>
#include <omp.h>
#include <math.h>

/*
 *Computes inner product of two given vectors A,B of size 
*/
float innerProduct(float *A,float* B,int size){
	float val=0;
#pragma omp parallel for reduction(+:val){
	for(int i=0;i<size;i++)
		val=val+A[i]*B[i];
}
	return val;
}

void likelihood_item(float *Q,bool *selected,float *user_sum,float **items,float **users,int k,float lambda,int numItems,int* item_sparse_csr_r,int *user_sparse_csr_c,int *allotted,int totalItems){
	//allotted contains items allotted to the node
	//numItems has items allotted to the node
	//totalItems has total number of items in the dataset
	//k is number of co-clusters
	for(int i=0;i<numItems;i++){

		if(selected[i]==true){
			Q[i]=innerProduct(items[allotted[i]],user_sum,k)+lambda*innerProduct(items[allotted[i]],items[allotted[i]],k);
			int start=item_sparse_csr_r[allotted[i]];
			int end;
			if(allotted[i]!=totalItems)
				end=item_sparse_csr_r[allotted[i]+1];
			else
				end=totalItems;
			for(int j=start;j<end;j++){
				int uid=user_sparse_csr_c[j];
				float x=innerProduct(items[allotted[i]],users[uid],k);
				Q[i]=Q[i]-x-log(1-pow(M_E,x));//Replace with efficient implementation of e^x
			}
		}
	}
}
