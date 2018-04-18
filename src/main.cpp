#include <iostream>
#include <mpi.h>
#include "ocular.h"
#include "lineSearch.h"
#include <math.h>
#define MASTER 0

using namespace std;

int main(int argc,char* argv[]){
    int numProcs,rank;
    MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numProcs);//Set up different communicators when work broadcast has to be removed
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int numUsers,numItems,numRatings;
    int tempTransfer[]={0,0,0};
    if(rank==MASTER){
        cin >> numItems>>numUsers>>numRatings;
        tempTransfer[0]=numItems;
        tempTransfer[1]=numUsers;
        tempTransfer[2]=numRatings;
    }
    MPI_Bcast(tempTransfer,3,MPI_INT,MASTER,MPI_COMM_WORLD);
    numItems=tempTransfer[0];
    numUsers=tempTransfer[1];
    numRatings=tempTransfer[2];
    int *csr_users = new int[numUsers+1];
	int *items = new int[numRatings];
	int *csr_items=new int[numItems+1];
	int *users=new int[numRatings];
    if(rank==MASTER){

        //take input

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

	}
	//setup mpi

	MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(users,numRatings,MPI_INT,MASTER,MPI_COMM_WORLD);
    MPI_Bcast(items,numRatings,MPI_INT,MASTER,MPI_COMM_WORLD);
    MPI_Bcast(csr_users,(numUsers+1),MPI_INT,MASTER,MPI_COMM_WORLD);
    MPI_Bcast(csr_items,(numItems+1),MPI_INT,MASTER,MPI_COMM_WORLD);

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
    fi = new float*[((numItems/numProcs)+1)*numProcs];
	fu = new float*[((numUsers/numProcs)+1)*numProcs];
	for(int i = 0; i < ((numItems/numProcs)+1)*numProcs; i++){
		fi[i] = new float[CLUSTERS];
	}
	for(int i = 0; i < ((numUsers/numProcs)+1)*numProcs; i++){
		fu[i] = new float[CLUSTERS];
	}
    ocular(numItems,numUsers,csr_items,users,csr_users,items,fi,fu,alloted_users,alloted_items,count_user,count_item);

	MPI_Finalize();
	int user_id,item_id;
	cout<<"Enter the user and item"<<endl;
	//while(){
		//cin>>user_id>>item_id;
		user_id=0;
		item_id=2;

		float x=innerProduct(fi[item_id],fu[user_id],CLUSTERS);
		cout<<1-pow(M_E,-x)<<endl;
	//}
	return 0;
}
