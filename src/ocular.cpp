/*
 *	Training of OCuLaR model
 */

#include "ocular.h"
#include "lineSearch.h"
#include "gradient.h"
#include "mpi.h"
#include <ctime>
#include <cstdlib>
#include <iostream>
#define	num_it	100

void ocular(int numItem,int numUser,int* csr_item,int* users,int* csr_user,int* items,float** fi,float** fu,int* alloted_item,int* alloted_user,int count_item,int count_user,int* proc_item,int* proc_user,int* displ_item,int* displ_user){
	srand(time(NULL));
	for(int i = 0; i < count_item; i++){
		for(int j = 0; j < CLUSTERS; j++){
			fi[alloted_item[i]][j]=(rand()+1)*1.0/rand();
		}
	}
	for(int i = 0; i < count_user; i++){
		for(int j = 0; j < CLUSTERS; j++){
			fu[alloted_user[i]][j]=(rand()+1)*1.0/rand();
		}
	}

	MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&(fi[0][0]),proc_item,displ_item,MPI_FLOAT,MPI_COMM_WORLD);
	MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&(fu[0][0]),proc_user,displ_user,MPI_FLOAT,MPI_COMM_WORLD);

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
		std::cerr << "Iteration: " << iter << std::endl;
		gradient(fi,fu,alloted_item,numItem,csr_item,users,sum_user,gi);

		linesearch(fi,sum_user,fu,gi,count_item,alloted_item,numItem,csr_item,users);

		MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&(fi[0][0]),proc_item,displ_item,MPI_FLOAT,MPI_COMM_WORLD);

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

		MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,&(fu[0][0]),proc_user,displ_user,MPI_FLOAT,MPI_COMM_WORLD);

		for(int i = 0; i < CLUSTERS; i++){
			sum_user[i] = 0;
		}
		for(int i = 0; i < numUser; i++){
			for(int j = 0; j < CLUSTERS; j++){
				sum_user[j] += fu[i][j];
			}
		}
	}
	for(int i = 0; i < numItem; i++){
		delete[] gi[i];
	}
	for(int i = 0; i < numUser; i++){
		delete[] gu[i];
	}
	delete[] gi;
	delete[] gu;
	delete[] sum_item;
	delete[] sum_user;
}
