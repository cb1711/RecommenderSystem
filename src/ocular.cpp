/*
 *	Training of OCuLaR model
 */

#include "ocular.h"
#include "lineSearch.h"
#include "gradient.h"
#include "mpi.h"
#include <ctime>
#include <cstdlib>
#define	num_it	100

void ocular(int numItem,int numUser,int* csr_item,int* users,int* csr_user,int* items,float** fi,float** fu,int* alloted_user,int* alloted_item,int count_user,int count_item){
	srand(time(NULL));
	fi = new float*[numItem];
	fu = new float*[numUser];
	for(int i = 0; i < numItem; i++){
		fi[i] = new float[CLUSTERS];
	}
	for(int i = 0; i < numUser; i++){
		fu[i] = new float[CLUSTERS];
	}
	float** curfi=new float*[count_item];
	for(int i = 0; i < count_item; i++){
		curfi[i] = new float[CLUSTERS];
		for(int j = 0; j < CLUSTERS; j++){
			curfi[i][j]=rand()*1.0/rand();
		}
	}
	float** curfu=new float*[count_user];
	for(int i = 0; i < count_user; i++){
		curfu[i] = new float[CLUSTERS];
		for(int j = 0; j < CLUSTERS; j++){
			curfu[i][j]=rand()*1.0/rand();
		}
	}
	// TODO: Change communicator
	MPI_Allgather(curfi,count_item*CLUSTERS,MPI_FLOAT,fi,numItem*CLUSTERS,MPI_FLOAT,MPI_COMM_WORLD);
	MPI_Allgather(curfu,count_user*CLUSTERS,MPI_FLOAT,fu,numUser*CLUSTERS,MPI_FLOAT,MPI_COMM_WORLD);
	float** gi = new float*[numItem];
	for(int item = 0; item < numItem; item++){
		gi[item] = new float[CLUSTERS];
	}
	float** gu = new float*[numUser];
	for(int user = 0; user < numUser; user++){
		gu[user] = new float[CLUSTERS];
	}
	float* sum_item = new float[CLUSTERS];
	float* sum_user = new float[CLUSTERS];
	for(int i = 0; i < CLUSTERS; i++){
		sum_item[i] = 0;
		sum_user[i] = 0;
	}
	for(int i = 0; i < numItem; i++){
		for(int j = 0; j < CLUSTERS; j++){
			sum_item[j] += fi[i][j];
		}
	}
	for(int i = 0; i < numUser; i++){
		for(int j = 0; j < CLUSTERS; j++){
			sum_user[j] += fu[i][j];
		}
	}
	
	for(int iter = 0; iter < num_it; iter++){
		gradient(fi,fu,alloted_item,numItem,csr_item,users,sum_user,gi);
		linesearch(fi,sum_user,fu,gi,count_item,alloted_item,numItem,csr_item,users);
		for(int item_idx = 0; item_idx < count_item; item_idx++){
			curfi[item_idx] = fi[alloted_item[item_idx]];
		}
		MPI_Allgather(curfi,count_item,MPI_FLOAT,fi,numItem,MPI_FLOAT,MPI_COMM_WORLD);
		for(int i = 0; i < CLUSTERS; i++){
			sum_item[i] = 0;
		}
		for(int i = 0; i < numItem; i++){
			for(int j = 0; j < CLUSTERS; j++){
				sum_item[j] += fi[i][j];
			}
		}
		gradient(fu,fi,alloted_user,numUser,csr_user,items,sum_item,gu);
		linesearch(fu,sum_item,fi,gu,count_user,alloted_user,numUser,csr_user,items);

		for(int user_idx = 0; user_idx < count_user; user_idx++){
			curfu[user_idx] = fu[alloted_user[user_idx]];
		}
		MPI_Allgather(curfu,count_user,MPI_FLOAT,fu,numUser,MPI_FLOAT,MPI_COMM_WORLD);
		for(int i = 0; i < CLUSTERS; i++){
			sum_user[i] = 0;
		}
		for(int i = 0; i < numUser; i++){
			for(int j = 0; j < CLUSTERS; j++){
				sum_user[j] += fu[i][j];
			}
		}
	}
	for(int i = 0; i < count_item; i++){
		delete[] curfi[i];
	}
	for(int i = 0; i < count_user; i++){
		delete[] curfu[i];
	}
	for(int i = 0; i < numItem; i++){
		delete[] gi[i];
	}
	for(int i = 0; i < numUser; i++){
		delete[] gu[i];
	}
	delete[] curfi;
	delete[] curfu;
	delete[] gi;
	delete[] gu;
	delete[] sum_item;
	delete[] sum_user;
}
