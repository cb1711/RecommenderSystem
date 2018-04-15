#include <iostream>
#include <omp.h>
#include "lineSearch.h"
#define sigma 1.0
#define beta 1.0
#define lambda 1.0

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

void likelihood(float *Q,bool *selected,float *user_sum,float **items,float **users,int k,int numItems,int* item_sparse_csr_r,int *user_sparse_csr_c,int *allotted,int totalItems){
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


void linesearch(float **items, float *user_sum, float**users, int k, float **gradient, int numItems, int *allotted, int totalItems, int* item_sparse_csr_r, int *user_sparse_csr_c,float lambda){
	//numItems is number of items allotted to the node
	//totalItems is number of items in all
	//allotted contains items allotted to the node
	float **newItems,**tempItems;
	tempItems=new float[totalItems];
	newItems=new float[totalItems];
	int removed[omp_get_max_threads()];
	memset(removed,0,omp_get_max_threads());
	for(int i=0;i<totalItems;i++){
		newItems[i]=new float[k];
		tempItems[i]=new float[k];
	}

	bool *active=new bool[numItems];
	memset(active,true,numItems);
	float *Q=new float[numItems];
	float *Q2=new float[numItems];
	likelihood_item(Q,active,user_sum,items,users,lambda,numItems,item_sparse_csr_r,user_sparse_csr_r,allotted,totalItems);
	float alpha=1;
	bool flag=true;
	while(flag){
		#pragma omp parallel for
		for(int i=0;i<numItems;i++){
			if(active[i])
				for(int j=0;j<k;j++)
					newItems[allotted[i]][j]=items[allotted[i]][j]-alpha*gradient[allotted[i]][j];
		}
		likelihood_item(Q2,active,user_sum,newItems,users,lambda,numItems,item_sparse_csr_r,user_sparse_csr_r,allotted,totalItems);

		#pragma omp parallel
		{
			#pragma omp for
				for(int i=0;i<numItems;i++){
					for(int j=0;j<k;j++)
						tempItems[i][j]=newItems[i][j]-items[i][j];
				}
			#pragma omp for
				for(int i=0;i<numItems;i++){
					if (active[i]){
						if (Q1[i]-Q[i]<=sigma*innerProduct(gradient[i],tempItems[i],k)){
							active[i]=false;
							removed[omp_get_thread_num()]++;
						}
					}
				}
	        }
		alpha=alpha*beta;
		int sum=0;
		for(int i=0;i<omp_get_max_threads();i++)
			sum+=removed[i];
		if(sum==numItems)
			flag=true;
	}
	delete[] active;
	delete[] Q;
	delete[] Q2;
	for(int i=0;i<totalItems;i++){
        delete[] newItems[i];
        delete[]  tempItems[i];
    }
    delete[] newItems;
    delete[] tempItems;
}
