#include <omp.h>
#include "lineSearch.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>
#include <iostream>
using namespace std;
/*
 *Computes inner product of two given vectors A,B of size
*/
float innerProduct(float *A,float* B,int size){
	float val=0;
//#pragma omp parallel for reduction(+:val)
	for(int i=0;i<size;i++){
		val=val+A[i]*B[i];

    }
	return val;
}
void likelihood(float *Q,bool *selected,float *user_sum,float **items,float **users,int numItems,int* item_sparse_csr_r,int *user_sparse_csr_c,int *allotted,int totalItems,bool &flag){
	//allotted contains items allotted to the node
	//numItems has items allotted to the node
	//totalItems has total number of items in the dataset
	//k is number of co-clusters
	for(int i=0;i<numItems;i++){
		if(selected[i]==true){
			Q[i]=innerProduct(items[allotted[i]],user_sum,CLUSTERS)+LAMBDA*innerProduct(items[allotted[i]],items[allotted[i]],CLUSTERS);
			int start=item_sparse_csr_r[allotted[i]];
			int end;
			if(allotted[i]!=totalItems)
				end=item_sparse_csr_r[allotted[i]+1];
			for(int j=start;j<end;j++){
				int uid=user_sparse_csr_c[j];
				float x=innerProduct(items[allotted[i]],users[uid],CLUSTERS);
				//if (x>0)
                    //flag=true;
                //else
                  //  flag=false;
                // if(x==0)
                //     Q[i]=0;
                // else
                Q[i]=Q[i]-x-log(1-pow(M_E,-x));//Replace with efficient implementation of e^x

			}
		}
	}
	/*for(int i=0;i<10;i++)
        std::cout<<Q[i]<<" ";
    std::cout<<std::endl;
    */
}


void linesearch(float **items, float *user_sum, float**users, float **gradient, int numItems, int *allotted, int totalItems, int* item_sparse_csr_r, int *user_sparse_csr_c){
	//numItems is number of items allotted to the node
	//totalItems is number of items in all
	//allotted contains items allotted to the node

    // std::cerr<<"In line search"<<std::endl;
	float **newItems,**tempItems;
	tempItems=new float*[totalItems];
	newItems=new float*[totalItems];
	int removed[omp_get_max_threads()];
	memset(removed,0,sizeof removed);
	for(int i=0;i<totalItems;i++){
		newItems[i]=new float[CLUSTERS];
		tempItems[i]=new float[CLUSTERS];
	}
	// for(int i=0;i<10;i++){
 //        std::cout<<i<<"\n";
 //        for(int j=0;j<CLUSTERS;j++)
 //            std::cout<<gradient[i][j]<<" ";
 //        std::cout<<"\n";
 //    }
	bool *active=new bool[numItems];
	memset(active,true,numItems*sizeof(bool));
	float *Q=new float[numItems];
	float *Q2=new float[numItems];
	bool check;
	likelihood(Q,active,user_sum,items,users,numItems,item_sparse_csr_r,user_sparse_csr_c,allotted,totalItems,check);
	/*for(int i=0;i<numItems;i++)
        for(int j=0;j<CLUSTERS;j++)
            cout<<Q[i]<<"q";
    cout<<endl;*/
	double alpha=1;
	bool flag=true;
	// cout<<"Going in"<<endl;
	while(flag){
        //std::cerr<<flag<<" "<<alpha<<std::endl;
		#pragma omp parallel for
		for(int i=0;i<numItems;i++){
			if(active[i])
				for(int j=0;j<CLUSTERS;j++){
					newItems[allotted[i]][j]=std::max((items[allotted[i]][j]-alpha*gradient[allotted[i]][j]),0.0);
					//if(newItems[allotted[i]][j]<=0.0)
                      //  newItems[allotted[i]][j]=.00000000000001;
                }
		}
		//cerr<<"frst"<<endl;
		likelihood(Q2,active,user_sum,newItems,users,numItems,item_sparse_csr_r,user_sparse_csr_c,allotted,totalItems,check);
		/*for(int i=0;i<10;i++)
            cout<<Q2[i]<<" ";
        cout<<endl;*/
        //if(check){
		//#pragma omp parallel
		{
			//#pragma omp for
				for(int i=0;i<numItems;i++){
					for(int j=0;j<CLUSTERS;j++)
						tempItems[allotted[i]][j]=newItems[allotted[i]][j]-items[allotted[i]][j];

				}
			//#pragma omp for
				for(int i=0;i<numItems;i++){
					if (active[i]){
                    	if (Q2[allotted[i]]-Q[allotted[i]]<=SIGMA*innerProduct(gradient[allotted[i]],tempItems[allotted[i]],CLUSTERS)){
							active[i]=false;
							//std::cerr<<"removed"<<std::endl;
							removed[omp_get_thread_num()]++;
						}
					}
				}
        }

		alpha=alpha*BETA;
		int sum=0;
    for(int i=0;i<omp_get_max_threads();i++)
			sum+=removed[i];
		//cout<<sum<<endl;
		if(sum==numItems)
			flag=false;
	}
	delete[] active;
	delete[] Q;
	delete[] Q2;
	for(int i=0;i<numItems;i++){
        for(int j=0;j<CLUSTERS;j++)
            items[allotted[i]][j]=newItems[allotted[i]][j];
    }
    for(int i=0;i<totalItems;i++){
        delete[] newItems[i];
        delete[] tempItems[i];
    }
    delete[] newItems;
    delete[] tempItems;
    // std::cerr<<"Out of line search"<<std::endl;
}
