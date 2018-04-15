/*
 * Evaluate gradient for cyclic block coordinate descent
 */

#include <omp.h>
#include "lineSearch.h"
#include "gradient.h"

void gradient(float** items,float** users,int* allotted,int numItems,int* csr_items,int* csr_users,float* user_sum,float** g){
	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0;i<numItems;i++){
			int item = allotted[i];
			for(int j=0;j<k;j++){
				g[item][j]=user_sum[j]+2*lambda*items[item][j];
			}
		}
		#pragma omp for
		for(int i=0;i<numItems;i++){
			int item = allotted[i];
			for(int user_idx = csr_items[item]; user_idx < csr_items[item+1]; user_idx++){
				int user = csr_users[user_idx];
				float x = inner_product(items[item],users[user],k);
				float factor = 1.0/(1-pow(M_E,-x))
				for(int j=0; j<k; j++){
					g[i][j] -= users[user][j]*factor;
				}
			}
		}
	}
}