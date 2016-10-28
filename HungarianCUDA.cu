// Fast CUDA implementation of the Hungarian algorithm.
// (maximum pay version)
//
// Satyendra Yadav and Paulo Lopes
// 
// Annex to the paper: Paulo Lopes, Satyendra Yadav et al., "Fast CUDA Implementation of the Hungarian Algorithm."
//
//
// Classical version of the Hungarian algorithm:
// (This algorithm was modified to result in an efficient GPU implementation, see paper)
//
// Initialize the slack matrix with the cost matrix, and then work with the slack matrix.
//
// STEP 1: Subtract the row minimum from each row. Subtract the column minimum from each column.
//
// STEP 2: Find a zero of the slack matrix. If there are no starred zeros in its column or row star the zero.
// Repeat for each zero.
//
// STEP 3: Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum.
//
// STEP 4: Find a non-covered zero and prime it. If there is no starred zero in the row containing this primed zero,
// Go to Step 5. Otherwise, cover this row and uncover the column containing the starred zero.
// Continue in this manner until there are no uncovered zeros left.
// Save the smallest uncovered value and Go to Step 6.
//
// STEP 5: Construct a series of alternating primed and starred zeros as follows:
// Let Z0 represent the uncovered primed zero found in Step 4.
// Let Z1 denote the starred zero in the column of Z0(if any).
// Let Z2 denote the primed zero in the row of Z1(there will always be one).
// Continue until the series terminates at a primed zero that has no starred zero in its column.
// Un-star each starred zero of the series, star each primed zero of the series, 
// erase all primes and uncover every row in the matrix. Return to Step 3.
//
// STEP 6: Add the minimum uncovered value to every element of each covered row, 
// and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered rows.

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <random>
#include <assert.h>

// Uncomment to use on SM20, (without dynamic parallelism).
// #define SM20

// Uncomment to use chars as the data type, otherwise use int
// #define CHAR_DATA_TYPE

// Uncomment to use a 4x4 predefined matrix for testing
// #define USE_TEST_MATRIX

#ifndef USE_TEST_MATRIX

// User inputs: These values should be changed by the user

const int n = 1024;						// size of the cost/pay matrix
const int range = n;					// defines the range of the random matrix.
// Bellow are the resulting total pays for matrixes of sizes for sizes of 128, 256, 512, 1024, 2048, 4096, 8192
// for range of 0 to 0.1n:	1408,	6144,	25600,		103423,		415744,		1671167,	6701055
// for range of 0 to n:		16130,	64959,	261071,		1046350,	4189849,	16768289,	67091170
// for range of 0 to 10n:	161909, 651152, 2613373,	10468839,	41907855,	167703929,	670949133

const int log2_n = 10;					// log2(n) needs to be entered manually
const int n_threads = 64;				// Number of threads used in small kernels grid size (typically grid size equal to n)
										// Used in steps 3ini, 3, 4ini, 4a, 4b, 5a and 5b
const int n_threads_reduction = 256;	// Number of threads used in the redution kernels in step 1 and 6
const int n_blocks_reduction = 256;		// Number of blocks used in the redution kernels in step 1 and 6
const int n_threads_full = 512;			// Number of threads used the largest grids sizes (typically grid size equal to n*n)
										// Used in steps 2 and 6

// End of user inputs

#else
const int n = 4;
const int log2_n = 2;
const int n_threads = 2;
const int n_threads_reduction = 2;
const int n_blocks_reduction = 2;
const int n_threads_full = 2;
#endif

const int n_blocks = n / n_threads;					// Number of blocks used in small kernels grid size (typically grid size equal to n)
const int n_blocks_full = n * n / n_threads_full;	// Number of blocks used the largest gris sizes (typically grid size equal to n*n)
const int row_mask = (1 << log2_n) - 1;				// Used to extract the row from tha matrix position index (matrices are column wise)
const int nrow = n, ncol = n;						// The matrix is square so the number of rows and columns is equal to n
const int max_threads_per_block = 1024;				// The maximum number of threads per block
const int seed = 45345;								// Initialization for the random number generator

// For the selection of the used data type
#ifndef CHAR_DATA_TYPE
typedef int data;
#define MAX_DATA INT_MAX
#define MIN_DATA INT_MIN
#else
typedef unsigned char data;
#define MAX_DATA 255
#define MIN_DATA 0
#endif

// Host Variables

// Some host variables start with h_ to distinguish them from the corresponding device variables
// Device variables have no prefix.

#ifndef USE_TEST_MATRIX
data pay[ncol][nrow];
#else
data pay[n][n] = { { 1, 2, 3, 4 }, { 2, 4, 6, 8 }, { 3, 6, 9, 12 }, { 4, 8, 12, 16 } };
#endif
int h_column_of_star_at_row[nrow];
int h_zeros_vector_size;
int h_n_matches;
bool h_found;
bool h_goto_5;

// Device Variables

__device__ data slack[nrow*ncol];						// The slack matrix
__device__ int zeros[nrow*ncol];						// A vector with the position of the zeros in the slack matrix
__device__ int zeros_vector_size;						// The size of the zeros vector
__device__ int row_of_star_at_column[ncol];				// A vector that given the column j gives the row of the star at that column (or -1, no star)
__device__ int column_of_star_at_row[nrow];				// A vector that given the row i gives the column of the star at that row (or -1, no star)
__device__ int cover_row[nrow];							// A vector that given the row i indicates if it is covered (1- covered, 0- uncovered)
__device__ int cover_column[ncol];						// A vector that given the column j indicates if it is covered (1- covered, 0- uncovered)
__device__ int column_of_prime_at_row[nrow];			// A vector that given the row i gives the column of the prime at that row (or -1, no prime)
__device__ int row_of_green_at_column[ncol];			// A vector that given the column j gives the row of the green at that column (or -1, no green)
__device__ int column_of_zero_at_row[nrow];				// The column of the zero at row i, found on step 4a

__device__ int n_matches;								// Used in step 3 to count the number of matches found
__device__ bool goto_5;									// After step 4b, goto step 5?
__device__ bool found = false;							// Found a zero in step 4a?

__device__ data max_in_mat_row[nrow];					// Used in step 1 to stores the maximum in rows
__device__ data min_in_mat_col[ncol];					// Used in step 1 to stores the minimums in columns
__device__ data d_min_in_mat_vect[n_blocks_reduction];	// Used in step 6 to stores the intermediate results from the first reduction kernel
__device__ data d_min_in_mat;							// Used in step 6 to store the minimum

__shared__ extern data sdata[];							// For access to shared memory

// -------------------------------------------------------------------------------------
// Device code
// -------------------------------------------------------------------------------------

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
#ifndef SM20
__device__
#endif
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

__global__ void Init()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	// initializations
	//for step 2
	if (i < nrow){
		cover_row[i] = 0;
		column_of_star_at_row[i] = -1;
	}
	if (i < ncol){
		cover_column[i] = 0;
		row_of_star_at_column[i] = -1;
	}
}

// STEP 1.
// a) Subtracting the maximum in each row by the row
const int n_rows_per_block = n / n_blocks_reduction;

__device__ void max_in_rows_warp_reduce(volatile data* sdata, int tid) {
	if (n_threads_reduction >= 64 && n_rows_per_block < 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
	if (n_threads_reduction >= 32 && n_rows_per_block < 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
	if (n_threads_reduction >= 16 && n_rows_per_block < 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
	if (n_threads_reduction >= 8 && n_rows_per_block < 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
	if (n_threads_reduction >= 4 && n_rows_per_block < 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
	if (n_threads_reduction >= 2 && n_rows_per_block < 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}

__global__ void max_in_rows()
{
	__shared__ data sdata[n_threads_reduction];		// One temporary result for each thread.

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	// One gets the line and column from the blockID and threadID.
	unsigned int l = bid * n_rows_per_block + tid % n_rows_per_block;
	unsigned int c = tid / n_rows_per_block;
	unsigned int i = c * nrow + l;
	const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
	data thread_min = MIN_DATA;

	while (i < n * n) {
		thread_min = max(thread_min, slack[i]);
		i += gridSize;  // go to the next piece of the matrix...
		// gridSize = 2^k * n, so that each thread always processes the same line or column
	}
	sdata[tid] = thread_min;

	__syncthreads();
	if (n_threads_reduction >= 1024 && n_rows_per_block < 1024) { if (tid < 512) { sdata[tid] = max(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (n_threads_reduction >= 512 && n_rows_per_block < 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (n_threads_reduction >= 256 && n_rows_per_block < 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (n_threads_reduction >= 128 && n_rows_per_block < 128) { if (tid <  64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) max_in_rows_warp_reduce(sdata, tid);
	if (tid < n_rows_per_block) max_in_mat_row[bid*n_rows_per_block + tid] = sdata[tid];
}

// b) subtracting the row by its minimum
const int n_cols_per_block = n / n_blocks_reduction;

__device__ void min_in_cols_warp_reduce(volatile data* sdata, int tid) {
	if (n_threads_reduction >= 64 && n_cols_per_block < 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
	if (n_threads_reduction >= 32 && n_cols_per_block < 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
	if (n_threads_reduction >= 16 && n_cols_per_block < 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
	if (n_threads_reduction >= 8 && n_cols_per_block < 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
	if (n_threads_reduction >= 4 && n_cols_per_block < 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
	if (n_threads_reduction >= 2 && n_cols_per_block < 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

__global__ void min_in_cols()
{
	__shared__ data sdata[n_threads_reduction];		// One temporary result for each thread

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	// One gets the line and column from the blockID and threadID.
	unsigned int c = bid * n_cols_per_block + tid % n_cols_per_block;
	unsigned int l = tid / n_cols_per_block;
	const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
	data thread_min = MAX_DATA;

	while (l < n) {
		unsigned int i = c * nrow + l;
		thread_min = min(thread_min, slack[i]);
		l += gridSize / n;  // go to the next piece of the matrix...
		// gridSize = 2^k * n, so that each thread always processes the same line or column
	}
	sdata[tid] = thread_min;

	__syncthreads();
	if (n_threads_reduction >= 1024 && n_cols_per_block < 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (n_threads_reduction >= 512 && n_cols_per_block < 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (n_threads_reduction >= 256 && n_cols_per_block < 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (n_threads_reduction >= 128 && n_cols_per_block < 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) min_in_cols_warp_reduce(sdata, tid);
	if (tid < n_cols_per_block) min_in_mat_col[bid*n_cols_per_block + tid] = sdata[tid];
}

__global__ void step_1_sub_row()
{

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int l = i & row_mask;

	slack[i] = max_in_mat_row[l] - slack[i];  // subtract the minimum in row from that row

}

__global__ void step_1_col_sub()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int c = i >> log2_n;
	slack[i] = slack[i] - min_in_mat_col[c]; // subtract the minimum in row from that row

	if (i == 0) zeros_vector_size = 0;
}

// Compress matrix
__global__ void compress_matrix(){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (slack[i] == 0) {
		int j = atomicAdd(&zeros_vector_size, 1);
		zeros[j] = i;
	}
}

// STEP 2
// Find a zero of slack. If there are no starred zeros in its
// column or row star the zero. Repeat for each zero.

__global__ void step_2()
{
	int i = threadIdx.x;
	bool repeat;

	do {
		repeat = false;
		__syncthreads();

		for (int j = i; j < zeros_vector_size; j += blockDim.x)
		{
			int z = zeros[j];
			int l = z & row_mask;
			int c = z >> log2_n;

			if (cover_row[l] == 0 && cover_column[c] == 0) {
				// thread trys to get the line
				if (!atomicExch(&(cover_row[l]), 1)){
					// only one thread gets the line
					if (!atomicExch(&(cover_column[c]), 1)){
						// only one thread gets the column
						row_of_star_at_column[c] = l;
						column_of_star_at_row[l] = c;
					}
					else {
						cover_row[l] = 0;
						repeat = true;
					}
				}
			}
		}
		__syncthreads();
	} while (repeat);
}

// STEP 3

// uncover all the rows and columns before going to step 3
__global__ void step_3ini()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
		cover_row[i] = 0;
		cover_column[i] = 0;
		if (i == 0) n_matches = 0;
}

// Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum
__global__ void step_3()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (row_of_star_at_column[i]>=0)
	{
		cover_column[i] = 1;
		atomicAdd((int*)&n_matches, 1);
	}
	
}

// STEP 4
// Find a noncovered zero and prime it. If there is no starred
// zero in the row containing this primed zero, go to Step 5.
// Otherwise, cover this row and uncover the column containing
// the starred zero. Continue in this manner until there are no
// uncovered zeros left. Save the smallest uncovered value and
// Go to Step 6.

__global__ void step_4_init()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	column_of_prime_at_row[i] =-1;
	row_of_green_at_column[i] = -1;
	column_of_zero_at_row[i] = -1;
}

// Maps the uncovered zeros into column_of_zero_at_row
__global__ void step_4a(){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < zeros_vector_size) {
		int z = zeros[i];

		int l = z & row_mask;
		int c = z >> log2_n;

		if (!cover_row[l] && !cover_column[c]){
			column_of_zero_at_row[l] = c;
			found = true; // This is set to false in 4b and at initialization.
		}
	}

	if (i == 0) {
		goto_5 = false;
	}
}

// The rest of step 4
__global__ void step_4b(){
	int l = blockDim.x * blockIdx.x + threadIdx.x;

	int c0 = column_of_zero_at_row[l];
	if (c0>=0)
	{
		column_of_prime_at_row[l] = c0;
		int c = column_of_star_at_row[l];
		if (c >= 0) {
			cover_row[l] = 1;
			cover_column[c] = 0;
			found = false;
		}
		else
			goto_5 = true;
	}

	column_of_zero_at_row[l] = -1;
}

/* STEP 5:
Construct a series of alternating primed and starred zeros as
follows:
Let Z0 represent the uncovered primed zero found in Step 4.
Let Z1 denote the starred zero in the column of Z0(if any).
Let Z2 denote the primed zero in the row of Z1(there will always
be one). Continue until the series terminates at a primed zero
that has no starred zero in its column. Unstar each starred
zero of the series, star each primed zero of the series, erase
all primes and uncover every line in the matrix. Return to Step 3.*/


// Eliminates joining paths
__global__ void step_5a()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int r_Z0, c_Z0;

	c_Z0 = column_of_prime_at_row[i];
	if (c_Z0 >= 0 && column_of_star_at_row[i] < 0){
		row_of_green_at_column[c_Z0] = i;

		while ((r_Z0 = row_of_star_at_column[c_Z0]) >= 0){
			c_Z0 = column_of_prime_at_row[r_Z0];
			row_of_green_at_column[c_Z0] = r_Z0;
		}
	}
}

// Applies the alternating paths
__global__ void step_5b()
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	int r_Z0, c_Z0, c_Z2;

	r_Z0 = row_of_green_at_column[j];

	if (r_Z0 >= 0 && row_of_star_at_column[j] < 0){

		c_Z2 = column_of_star_at_row[r_Z0];

		column_of_star_at_row[r_Z0] = j;
		row_of_star_at_column[j] = r_Z0;

		while (c_Z2 >= 0) {
			r_Z0 = row_of_green_at_column[c_Z2];	// row of Z2
			c_Z0 = c_Z2;							// col of Z2
			c_Z2 = column_of_star_at_row[r_Z0];		// col of Z4

			// star Z2
			column_of_star_at_row[r_Z0] = c_Z0;
			row_of_star_at_column[c_Z0] = r_Z0;
		}
	}
}

// STEP 6
// Add the minimum uncovered value to every element of each covered
// row, and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered lines.

template <unsigned int blockSize>
__device__ void min_warp_reduce(volatile data* sdata, int tid) {
	if (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
	if (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize>  // blockSize is the size of a block of threads
__device__ void min_reduce1(volatile data *g_idata, volatile data *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = MAX_DATA;

	while (i < n) {
		int i1 = i;
		int i2 = i + blockSize;
		int l1 = i1 & row_mask;
		int c1 = i1 >> log2_n; 
		int g1;
		if (cover_row[l1] == 1 || cover_column[c1] == 1) g1 = MAX_DATA;
		else g1 = g_idata[i1];
		int l2 = i2 & row_mask;
		int c2 = i2 >> log2_n;
		int g2;
		if (cover_row[l2] == 1 || cover_column[c2] == 1) g2 = MAX_DATA;
		else g2 = g_idata[i2];
		sdata[tid] = min(sdata[tid], min(g1, g2));
		i += gridSize;
	}

	__syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) min_warp_reduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ void min_reduce2(volatile data *g_idata, volatile data *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;

	sdata[tid] = min(g_idata[i], g_idata[i + blockSize]);

	__syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) min_warp_reduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void step_6_add_sub()
{
	// STEP 6:
	//	/*STEP 6: Add the minimum uncovered value to every element of each covered
	//	row, and subtract it from every element of each uncovered column.
	//	Return to Step 4 without altering any stars, primes, or covered lines. */
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int l = i & row_mask;
	int c = i >> log2_n;
	if (cover_row[l] == 1 && cover_column[c] == 1)
		slack[i] += d_min_in_mat;
	if (cover_row[l] == 0 && cover_column[c] == 0)
		slack[i] -= d_min_in_mat;

	if (i == 0) zeros_vector_size = 0;
}

__global__ void min_reduce_kernel1() {
	min_reduce1<n_threads_reduction>(slack, d_min_in_mat_vect, nrow*ncol);
}

__global__ void min_reduce_kernel2() {
	min_reduce2<n_threads_reduction / 2>(d_min_in_mat_vect, &d_min_in_mat, n_blocks_reduction);
}

// Hungarian_Algorithm
// This function is run on the device if device if dynamic parallelism is enabled,
// or in the host if it is disabled (SM20).

#ifdef SM20
void Hungarian_Algorithm()
#else
__global__ void Hungarian_Algorithm()
#endif
{
	// Initialization
	Init << < n_blocks, n_threads >> > ();
	checkCuda(cudaDeviceSynchronize());

	// Step 1 kernels
	max_in_rows << < n_blocks_reduction, n_threads_reduction >> >();
	checkCuda(cudaDeviceSynchronize());
	step_1_sub_row << <n_blocks_full, n_threads_full >> >();
	checkCuda(cudaDeviceSynchronize());

	min_in_cols << < n_blocks_reduction, n_threads_reduction >> >();
	checkCuda(cudaDeviceSynchronize());
	step_1_col_sub << <n_blocks_full, n_threads_full >> >();
	checkCuda(cudaDeviceSynchronize());

	// compress_matrix
	compress_matrix << < n_blocks_full, n_threads_full >> > ();
	checkCuda(cudaDeviceSynchronize());

	// Step 2 kernels
	step_2 << <1, min(max_threads_per_block, nrow) >> > ();
	checkCuda(cudaDeviceSynchronize());

	while (1) {  // repeat steps 3 to 6

		// Step 3 kernels
		step_3ini << <n_blocks, n_threads >> >();
		checkCuda(cudaDeviceSynchronize());
		step_3 << <n_blocks, n_threads >> > ();
		checkCuda(cudaDeviceSynchronize());

#ifdef SM20
		cudaMemcpyFromSymbol(&h_n_matches, n_matches, sizeof(int));
		if (h_n_matches >= ncol) return;		// It's done
#else
		if (n_matches >= ncol) return;			// It's done
#endif

		//step 4_kernels
		step_4_init << <n_blocks, n_threads >> > ();
		checkCuda(cudaDeviceSynchronize());

		while (1) // repeat step 4 and 6
		{
			do {  // step 4 loop
#ifdef SM20
				cudaMemcpyFromSymbol(&h_zeros_vector_size, zeros_vector_size, sizeof(int));
				if (h_zeros_vector_size > 100 * n)
					step_4a << < n_blocks_full, n_threads_full >> > ();
				else if (h_zeros_vector_size > 10 * n)
					step_4a << < 100 * n / n_threads, n_threads >> > ();
				else
					step_4a << < 10 * n / n_threads, n_threads >> > ();

#else
				if (zeros_vector_size>100 * n)
					step_4a << < n_blocks_full, n_threads_full >> > ();
				else if (zeros_vector_size>10 * n)
					step_4a << < 100 * n / n_threads, n_threads >> > ();
				else
					step_4a << < 10 * n / n_threads, n_threads >> > ();

#endif
				checkCuda(cudaDeviceSynchronize());
#ifdef SM20
				cudaMemcpyFromSymbol(&h_found, found, sizeof(bool));
				if (!h_found) break;
#else
				if (!found) break;
#endif

				step_4b << < n_blocks, n_threads >> >();
				checkCuda(cudaDeviceSynchronize());
#ifdef SM20
				cudaMemcpyFromSymbol(&h_goto_5, goto_5, sizeof(bool));

			} while (!h_goto_5);
#else
			} while (!goto_5);
#endif

#ifdef SM20
			cudaMemcpyFromSymbol(&h_goto_5, goto_5, sizeof(bool));
			if (h_goto_5) break; // Or if (!h_found)
#else
			if (goto_5) break;	// Or if (!found)
#endif

			//step 6_kernel
			min_reduce_kernel1 << <n_blocks_reduction, n_threads_reduction, n_threads_reduction*sizeof(int) >> >();
			checkCuda(cudaDeviceSynchronize());
			min_reduce_kernel2 << <1, n_blocks_reduction / 2, (n_blocks_reduction / 2) * sizeof(int) >> >();
			checkCuda(cudaDeviceSynchronize());
			step_6_add_sub << <n_blocks_full, n_threads_full >> >();
			checkCuda(cudaDeviceSynchronize());

			//compress_matrix
			compress_matrix << < n_blocks_full, n_threads_full >> > ();
			checkCuda(cudaDeviceSynchronize());

		} // repeat step 4 and 6

		step_5a << < n_blocks, n_threads >> > ();
		checkCuda(cudaDeviceSynchronize());
		step_5b << < n_blocks, n_threads >> > ();
		checkCuda(cudaDeviceSynchronize());

	}  // repeat steps 3 to 6
}

// -------------------------------------------------------------------------------------
// Host code
// -------------------------------------------------------------------------------------

// Used to make sure some constants are properly set
void check(bool val, char *str){
	if (!val) {
		printf("Check failed: %s!\n", str);
		getchar();
		exit(-1);
	}
}

int main()
{
	// Constant checks:
	check(n == (1 << log2_n), "Incorrect log2_n!");
	check(n_threads*n_blocks == n, "n_threads*n_blocks != n\n");
	// step 1
	check(n_blocks_reduction <= n, "Step 1: Should have several lines per block!");
	check(n % n_blocks_reduction == 0, "Step 1: Number of lines per block should be integer!");
	check((n_blocks_reduction*n_threads_reduction) % n == 0, "Step 1: The grid size must be a multiple of the line size!");
	check(n_threads_reduction*n_blocks_reduction <= n*n, "Step 1: The grid size is bigger than the matrix size!");
	// step 6
	check(n_threads_full*n_blocks_full <= n*n, "Step 6: The grid size is bigger than the matrix size!");

#ifndef USE_TEST_MATRIX
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<int> distribution(0, range-1);
	
	for (int c = 0; c < ncol; c++)
		for (int r = 0; r < nrow; r++) {
			pay[c][r] = distribution(generator);
		}
#endif
	
	// Copy vectors from host memory to device memory
	cudaMemcpyToSymbol(slack, pay, sizeof(data)*nrow*ncol); // symbol refers to the device memory hence "To" means from Host to Device
	
	// Invoke kernels
	
	time_t start_time = clock();

#ifdef SM20
	Hungarian_Algorithm();
#else
	Hungarian_Algorithm << < 1, 1 >> > ();
#endif
	cudaDeviceSynchronize();

	time_t stop_time = clock();

	// Copy assignments from Device to Host and calculate the total Cost.
	cudaMemcpyFromSymbol(h_column_of_star_at_row, column_of_star_at_row, nrow*sizeof(int));

	int total_pay = 0;
	for (int r = 0; r < nrow; r++) {
		int c = h_column_of_star_at_row[r];
		if (c >= 0) total_pay += pay[c][r];
	}

	printf("Total pay is: %d \n", total_pay);
	printf("Elapsed time: %f ms\n", 1000.0*(double)(stop_time - start_time) / CLOCKS_PER_SEC);
	printf("Note: This time measurment is portable but very inaccurate!\n");
}
