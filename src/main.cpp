#include <iostream>
#include <mpi.h>
#include "ocular.h"
#include "lineSearch.h"
#define MASTER 0

using namespace std;

int main(int argc,char* argv[]){

	int numUsers,numItems,numRatings;
	cin >> numUsers>>numItems>>numRatings;
	//take input
	int *csr_users = new int[numUsers+1];
	int *items = new int[numRatings];
	int *csr_items=new int[numItems+1];
	int *users=new int[numRatings];
    /*Format conversion*/
	int prevItem = 0;
	csr_users[0] = 0;
	int prevUser= 0;
	items[0]=0;
	for(int i=0;i<numRatings;i++){

		int curr_item;
		cin >> curr_item;
		curr_item--;
		cin >> users[i];
		users[i]--;
		while (prevItem<curr_item){
			prevItem++;
			csr_items[prevItem] = i;
		}
	}


	for (int i = 0; i<numRatings; i++){
		cin >> items[i];
		items[i]--;
		int curr_user;
		cin >> curr_user;
		curr_user--;
		while (prevUser<curr_user){
            prevUser++;
            csr_users[prevUser] = i;
		}
	}

	while (prevItem<=numItems){
		prevItem++;
		csr_items[prevItem]=numRatings;
	}
	while (prevUser<=numUsers){
		prevUser++;
		csr_users[prevUser]=numRatings;
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
        all_items=new int[numItems];
        all_users=new int[numUsers];
        for(int i=0;i<numItems;i++){
            all_items[i]=i;
        }
        for(int i=0;i<numUsers;i++){
            all_users[i]=i;
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
    fi = new float*[numItems];
	fu = new float*[numUsers];
	for(int i = 0; i < numItems; i++){
		fi[i] = new float[CLUSTERS];
	}
	for(int i = 0; i < numUsers; i++){
		fu[i] = new float[CLUSTERS];
	}
    ocular(numItems,numUsers,csr_items,users,csr_users,items,fi,fu,alloted_users,alloted_items,count_user,count_item);
    // std::cerr<<"Hello";
    // for(int j=0;j<10;j++){
    // for(int i=0;i<CLUSTERS;i++)
    //     std::cout<<fi[j][i]<<" ";

    // // std::cout<<"\n";
    // }
    for(int i = 0; i < numItems; i++){
    	for(int j = 0; j < CLUSTERS; j++){
    		std::cerr << fi[i][j] << " ";
    	}
    	std::cerr<<std::endl;
    }
    for(int i = 0; i < numUsers; i++){
    	for(int j = 0; j < CLUSTERS; j++){
    		std::cerr << fu[i][j] << " ";
    	}
    	std::cerr<<std::endl;
    }
	MPI_Finalize();
	return 0;
}
