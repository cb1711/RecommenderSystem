#include <iostream>
#include <omp.h>
#include "likelihood.h"
#include "lineSearch.h"
#define sigma 1.0
#define beta 1.0


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
}
