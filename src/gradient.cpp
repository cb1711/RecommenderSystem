/*
 * Evaluate gradient for cyclic block coordinate descent
 */

#include "omp.h"
#include "lineSearch.h"
#include "gradient.h"
#include <cmath>
#include <iostream>
using namespace std;
void gradient(float** items,float** users,int* allotted,int numItems,int* csr_items,int* csr_users,float* user_sum,float** g){
    /*for(int i=0;i<CLUSTERS;i++)
        cout<<user_sum[i]<<"--\n";
    for(int i=0;i<10;i++){
        for(int j=0;j<CLUSTERS;j++)
            cout<<items[i][j]<<"-";
        cout<<"\n";
    }*/
	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0;i<numItems;i++){
			int item = allotted[i];
			for(int j=0;j<CLUSTERS;j++){

				g[item][j]=user_sum[j]+2*LAMBDA*items[item][j];
			}
		}
		#pragma omp for
		for(int i=0;i<numItems;i++){
			int item = allotted[i];
			for(int user_idx = csr_items[item]; user_idx < csr_items[item+1]; user_idx++){
				int user = csr_users[user_idx];
				float x = innerProduct(items[item],users[user],CLUSTERS);
				float factor = 1.0/(1-pow(M_E,-x));
				//else
				// if(x==0)
    //                 factor=0;

				for(int j=0; j<CLUSTERS; j++){
					g[i][j] -= users[user][j]*factor;
				}
			}
		}
	}/*
	for(int i=0;i<10;i++){
        cout<<"grad\n";
        for(int j=0;j<CLUSTERS;j++)
            cout<<g[i][j]<<"---";
        cout<<"\n";
    }*/
}
