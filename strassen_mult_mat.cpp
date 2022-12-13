/*
    Compile : g++ -o OpenMP  OpenMP-strassen_mult_mat.cpp -fopenmp
    Run : ./strassen.exe SIZE THRESHOLD number of threads
*/

#include "omp.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#include "mpi.h"

#define MAX_THREADS 65536

int k, k_dash;
int numOfThreads;

/*
here 'n' is the size of given matrix ; A is matrix as first input; B is matrix as second input; C is the output matrix
*/

/*Function for the general matrix multiplication*/
void generalMatMult(int n, int **A, int **B, int** C)
{
	for(int i=0; i<n; i++)  //loop for traversing rows
    {
		for(int j=0; j<n; j++) //loop for traversing columns
        {
			C[i][j] = 0;
			for(int k=0;k<n; k++)
            {
				C[i][j] += A[i][k] * B[k][j]; //logic for general matrix multiplication
			}
		}
	}
}

/*Function to allocate memory for a n*n matrix*/
int** MemAllocateForMatrix(int n)
{
	int **head = new int*[n]; // here **head is to collect head i.e., starting pointer for elements in each row

	// matrix storage
	int *nRows = new int[n*n];
	for(int i=0; i<n; i++)
    {
		*(head+i) = nRows;
		nRows+= n;
	}
	return head;
}

/*Function for 2D memory cleanup operation*/
void MatrixMemoryFlush(int **ptr)
{
	delete []*ptr; // Free up the pointers
	delete []ptr; // free up for rows
}

/*Function for 1D memory cleanup operation*/
void MatrixMemFree(int** ptr)
{
	delete[] ptr;
}

/*Function for addition*/
void matrixAddition(int **temp, int n, int **A, int **B)
{
	for (int i = 0; i < n ; i++)
		for (int j = 0; j < n; j++)
			temp[i][j] = A[i][j] + B[i][j];
	
}

/*Function for subtraction*/
void matrixSubtraction(int **temp, int n, int **A, int **B)
{
	for (int i = 0; i < n ; i++)
		for (int j = 0; j < n; j++)
			temp[i][j] = A[i][j] - B[i][j];
}

/*Function for computing quad/quater matrix*/
void subQuadMatrix(int **temp, int n, int **A, int x, int y)
{
	for(int i=0; i<n; i++)
    {
		temp[i] = &A[x+i][y];
	}
}

/*Main logic that utilizes Divide and conquer recursive method to calculates the product of two matrices*/
/*Function that computes the strassen multiplication for given matrices*/
void strassenMult(int n, int **A, int **B, int** C)
{
	/*monitoring for undesired conditions*/
	if(((float)n) <= pow(2, k)/pow(2, k_dash))
    {
		for(int i = 0; i<n; i++)
        {
			for(int j=0; j<n; j++)
            {
				C[i][j] =0;
				for(int k=0; k<n; k++)
					C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
    else
    {
		
		/*Here the matrix is divided into four equal parts*/
		int size = n/2;
		
		/*Assigning memory locations for M1 to M7*/
		int **M1= MemAllocateForMatrix(size);
		int **M2= MemAllocateForMatrix(size);
		int **M3= MemAllocateForMatrix(size);
		int **M4= MemAllocateForMatrix(size);
		int **M5= MemAllocateForMatrix(size);
		int **M6= MemAllocateForMatrix(size);
		int **M7= MemAllocateForMatrix(size);
		
		
		/*Matrices to hold the incomplete results
		 Naming convention: typeOfOperation_InputMatrix_Mx
		 a_A_M1: Here, addition is done on the submatrices of A that will be used in the M1 stage computation*/
		int **a_A_M1 = MemAllocateForMatrix(size);
		int **a_B_M1 = MemAllocateForMatrix(size);
		int **a_A_M2 = MemAllocateForMatrix(size);
		int **s_B_M3 = MemAllocateForMatrix(size);
		int **s_B_M4 = MemAllocateForMatrix(size);
		int **a_A_M5 = MemAllocateForMatrix(size);
		int **a_A_M6 = MemAllocateForMatrix(size);
		int **a_B_M6 = MemAllocateForMatrix(size);
		int **s_A_M7 = MemAllocateForMatrix(size);
		int **a_B_M7 = MemAllocateForMatrix(size);


		/*assigning memory locations for A, B, C*/
        int **C11 = new int*[size];
		int **C12 = new int*[size];
		int **C21 = new int*[size];
		int **C22 = new int*[size];

		int **A11 = new int*[size];
		int **A12 = new int*[size];
		int **A21 = new int*[size];
		int **A22 = new int*[size];

		int **B11 = new int*[size];
		int **B12 = new int*[size];
		int **B21 = new int*[size];
		int **B22 = new int*[size];		
		
		/*Creating quad-matrix structures*/
		subQuadMatrix(A11, size, A, 0, 0);
		subQuadMatrix(A12, size, A, 0, size);
		subQuadMatrix(A21, size, A, size, 0);
		subQuadMatrix(A22, size, A, size, size);
		
		subQuadMatrix(B11, size, B, 0, 0);
		subQuadMatrix(B12, size, B, 0, size);
		subQuadMatrix(B21, size, B, size, 0);
		subQuadMatrix(B22, size, B, size, size);
		
		subQuadMatrix(C11, size, C, 0, 0);
		subQuadMatrix(C12, size, C, 0, size);
		subQuadMatrix(C21, size, C, size, 0);
		subQuadMatrix(C22, size, C, size, size);
		
		/*Multithreading: Parallel computation for M1 to M7*/
		#pragma omp task
		{
			// 𝑀1 =(𝐴11 +𝐴22)*(𝐵11 +𝐵22)
			matrixAddition(a_A_M1, size, A11, A22);
			matrixAddition(a_B_M1, size, B11, B22);
			strassenMult(size, a_A_M1, a_B_M1, M1);
		}
		#pragma omp task
		{
			//𝑀2 =(𝐴21 +𝐴22)𝐵11
			matrixAddition(a_A_M2, size, A21, A22);
			strassenMult(size, a_A_M2, B11, M2);
		}
		#pragma omp task
		{
			//  𝑀3 =𝐴11(𝐵12 −𝐵22)
			matrixSubtraction(s_B_M3, size, B12, B22);
			strassenMult(size, A11, s_B_M3, M3);
		}
		#pragma omp task
		{
			//𝑀4 =𝐴22(𝐵21 −𝐵11)
			matrixSubtraction(s_B_M4, size, B21, B11);
			strassenMult(size, A22, s_B_M4, M4);
		}
		#pragma omp task
		{
			//𝑀5 =(𝐴11 +𝐴12)𝐵22
			matrixAddition(a_A_M5, size, A11, A12);
			strassenMult(size, a_A_M5, B22, M5);
		}
		#pragma omp task
		{
			//𝑀6 =(𝐴21 −𝐴11)(𝐵11 +𝐵12)
			matrixSubtraction(a_A_M6, size, A21, A11);
			matrixAddition(a_B_M6, size, B11, B12);
			strassenMult(size, a_A_M6, a_B_M6, M6);
		}
		#pragma omp task
		{
			//  𝑀7 =(𝐴12 −𝐴22)(𝐵21 +𝐵22)
			matrixSubtraction(s_A_M7, size, A12, A22);
			matrixAddition(a_B_M7, size, B21, B22);
			strassenMult(size, s_A_M7, a_B_M7, M7);
		}
		
		/*Hold till execution for all threads are done*/
		#pragma omp taskwait
	
		// Strassen Result calculation to be sent to the callers
        for(int i=0; i<size; i++)
        {
			for(int j=0; j<size; j++)
            {
				C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
				C12[i][j] = M3[i][j] + M5[i][j];
				C21[i][j] = M2[i][j] + M4[i][j];
				C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
			}
		}

		/*Here the garbage collection is handled*/
		MatrixMemoryFlush(a_A_M1);
		MatrixMemoryFlush(a_A_M2);
		MatrixMemoryFlush(a_A_M5);
		MatrixMemoryFlush(a_A_M6);
		MatrixMemoryFlush(s_A_M7);
		MatrixMemoryFlush(s_B_M3);
		MatrixMemoryFlush(s_B_M4);
		MatrixMemoryFlush(a_B_M1);
		MatrixMemoryFlush(a_B_M6);
		MatrixMemoryFlush(a_B_M7);
		
		/*clearing memory for M1 to M7*/
		MatrixMemoryFlush(M1);
		MatrixMemoryFlush(M2);
		MatrixMemoryFlush(M3);
		MatrixMemoryFlush(M4);
		MatrixMemoryFlush(M5);
		MatrixMemoryFlush(M6);
		MatrixMemoryFlush(M7);
		
		/*clearing memory for 1D-input matrix bits*/
		MatrixMemFree(A11);
		MatrixMemFree(A12);
		MatrixMemFree(A21);
		MatrixMemFree(A22);
		MatrixMemFree(B11);
		MatrixMemFree(B12);
		MatrixMemFree(B21);
		MatrixMemFree(B22);
		MatrixMemFree(C11);
		MatrixMemFree(C12);
		MatrixMemFree(C21);
		MatrixMemFree(C22);
	}
}

/*Main method definition for the entire code i.e., starting point for execution*/
int main(int argc, char* argv[])
{
	/*compute the k value for matrix of size n * n that is n = 2 pow k*/
	k = atoi(argv[1]);
	
	int n = pow(2, k);
	/*compute the k' value to determine the threshold*/
	k_dash = atoi(argv[2]);
	
	int q = atoi(argv[3]);
	
	if((q>=16) || (numOfThreads = (1 << q)) > MAX_THREADS)
    {
		printf("\nError: Maximum limit permitted for threads : %d.", MAX_THREADS);
		exit(0);
	}

	/*To calculate time and allocate memory for matrices*/
    int **C = MemAllocateForMatrix(n);
	int **A = MemAllocateForMatrix(n);
	int **B = MemAllocateForMatrix(n);
	int **resMat = MemAllocateForMatrix(n);
	
	/*Enforce the number of available and utilized threads passed with strict precision*/
	omp_set_dynamic(0);
	
	/*Add some random values to the slots in A and B*/
	for(int i=0; i<n; i++)
    {
		for(int j=0; j<n; j++)
        {
			A[i][j] = rand()%333;
		}
	}
	
	for(int i=0; i<n; i++)
    {
		for(int j=0; j<n; j++)
        {
			B[i][j] = rand()%333;
		}
	}
	
	/*Calculate standard multiplication matrix to cross-check the result*/
	generalMatMult(n, A, B, resMat);
	
	/*set number of parallel processing threads*/
	omp_set_num_threads(numOfThreads);
	
	/*Calculating Time*/
	struct timespec start, stop;
    double tot_time;

	/*Calculate the time taken*/
	clock_gettime(CLOCK_REALTIME, &start);
	
	/*Logic for Strassen Multiplication Matrix*/
	#pragma omp parallel
	{
		#pragma omp single
		{
			strassenMult(n, A, B, C);
		}
	}
	clock_gettime(CLOCK_REALTIME, &stop);
    tot_time = (stop.tv_sec-start.tv_sec)
	+0.000000001*(stop.tv_nsec-start.tv_nsec);
	
	/*Cross_Verification*/
	bool correct = true;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if(resMat[i][j] != C[i][j]) correct = false;
	if(correct)
    {
		printf("Correct: k=%d, the matrix_size = %d X %d, k'= %d, no_Of_threads = %d, strassen_time = %8.4f sec \n", k,n,n, k_dash,numOfThreads,tot_time);          
	}
	else
    {
		printf("Error: k=%d, the matrix_size = %d X %d, k' = %d, no_Of_threads = %d, strassen_time = %8.4f sec \n", k,n,n, k_dash,numOfThreads,tot_time); 
	}
	
	/*The final garbage collection*/
    MatrixMemoryFlush(resMat);
    MatrixMemoryFlush(C);
	MatrixMemoryFlush(A);
	MatrixMemoryFlush(B);
	
	return 0;
}

