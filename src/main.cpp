#include <iostream>
#include "gradient.h"
#include "lineSearch.h"
#include <mpi.h>
#include "ocular.h"
#define MASTER 0

using namespace std;
int main(int argc,char* argv){

	int numUsers,numItems,numRatings;
	cin >> numUsers>>numItems>>numRatings;
	//take input
	int *csr_users = new int[numRatings + 1];
	int *items = new int[numItems];
	/*Format conversion*/
	int prevRow = 0;
	csr_users[0] = 0;
	items[0]=0;
	for (int i = 0; i<numRatings; i++){
		cin >> csr_users[i];
		csr_users[i]--;
		int curr_row;
		cin >> curr_row;
		curr_row--;
		while (prevRow<curr_row){
			prevRow++;
			items[prevRow] = i;
		}
	}
	while (prevRow<numRows){
		prevRow++;
		items[prevRow]=numRatings;
	}

	//setup mpi
	int numProcs,rank;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numProcs);//Set up different communicators when work broadcast has to be removed
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);




	int *all_items,*all_users;
	int *alloted_items,*alloted_users;

	//Divide work
	if(rank==MASTER){ //Use better method for alloting
        *all_items=new int[numItems];
        *all_users=new int[numUsers];
        for(int i=0;i<numItems;i++){
            alloted_items[i]=i;
        }
        for(int i=0;i<numUsers;i++){
            alloted_items[i]=i;
        }
    }
    alloted_items=new int[numItems/numProcs + 1];
    alloted_users=new int[numUsers/numProcs + 1];
    MPI_Scatter(all_users,numUsers/numProcs+1,MPI_INT,alloted_users,numUsers/numProcs+1,MPI_INT,MASTER,MPI_COMM_WORLD);
    MPI_Scatter(all_items,numItems/numProcs+1,MPI_INT,alloted_items,numItems/numProcs+1,MPI_INT,MASTER,MPI_COMM_WORLD);
    float **fi,**fu;
    int count_user,count_item;
    count_user=(rank==numProcs-1)?(numUsers-(numProcs-1)*(numUsers/numProcs+1)):numUsers/numProcs+1;
    count_item=(rank==numProcs-1)?(numItems-(numProcs-1)*(numItems/numProcs+1)):numItems/numProcs+1;
    //Call ocular
    ocular(numItems,numUsers,csr_item,users,csr_user,items,fi,fu,alloted_users,alloted_items,count_user,count_item);

	MPI_Finalize();
	return 0;
}
