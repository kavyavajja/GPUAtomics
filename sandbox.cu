#include <iostream>
#include <math.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace std;

struct DATA
{
    float fData;
    unsigned int   nData;
    unsigned long long int ullData;

    DATA()
    {
        fData = 0.0f;
        nData = 0;
        ullData = 0;
    }
};

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

__global__ void delay(volatile int* flag,
    unsigned long long timeout_clocks = 10000000) {
    // Wait until the application notifies us that it has completed queuing up the
    // experiment, or timeout and exit, allowing the application to make progress
    long long int start_clock, sample_clock;
    start_clock = clock64();

    while (!*flag) {
        sample_clock = clock64();

        if (sample_clock - start_clock > timeout_clocks) {
            break;
        }
    }
}

__global__
void remoteAtomicsKernel(/*DATA* local_data,*/ DATA* peer1_data/*, DATA* peer2_data, DATA* peer3_data*/) // Params are commented our for final-benchmark-task
{
    //atomicAdd_system(&(local_data->ullData), 1); 
    atomicAdd_system(&(peer1_data->nData), 1);
    //atomicAdd_system(&(peer2_data->ullData), 1);
    //atomicAdd_system(&(peer3_data->ullData), 1);

    //local_data->ullData += 1;
//peer1_data->ullData += 1;
//peer2_data->ullData += 1;
//peer3_data->ullData += 1;
}

__global__
void localAtomicsKernel(DATA* local_data)
{
    atomicAdd_system(&(local_data->nData), 1);
}

__global__
void localAtomicsKernelEx(unsigned int* ptrData, unsigned int nTotalAtomics, int gridSize)
{
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    //size_t gridSize = blockDim.x * gridDim.x;

    for (size_t i = globalId; i < nTotalAtomics; i += gridSize) {
        atomicAdd_system(ptrData, 1);
    }
}

void localAtomics(int nGridSize, int nBlockSize)
{
    int nGPUCount = 0;
    cudaGetDeviceCount(&nGPUCount);

    std::cout << "Total GPUs: " << nGPUCount << std::endl;

    for (int i = 0; i < nGPUCount; i++)
    {
        cudaSetDevice(i);
        for (int j = 0; j < nGPUCount; j++)
        {
            if (i != j)
            {
                //cudaDeviceEnablePeerAccess(j, 0);
                cudaDeviceDisablePeerAccess(j);
            }
        }
    }

    size_t size = 1 * sizeof(DATA);

    // Initializing data for device-0...
    cudaSetDevice(0);

    DATA* device0_data;
    cudaMalloc(&device0_data, size);
    cudaMemset(&device0_data, 0, size);
    /*
        // Initializing data for device-1...
        cudaSetDevice(1);

        DATA* device1_data;
        cudaMalloc(&device1_data, size);
        cudaMemset(&device1_data, 0, size);

        // Initializing data for device-2...
        cudaSetDevice(2);

        DATA* device2_data;
        cudaMalloc(&device2_data, size);
        cudaMemset(&device2_data, 0, size);

        // Initializing data for device-3...
        cudaSetDevice(3);
    */
    DATA* device3_data;
    cudaMalloc(&device3_data, size);
    cudaMemset(&device3_data, 0, size);

    dim3 grid_dim(nGridSize, 1, 1);
    dim3 block_dim(nBlockSize, 1, 1);

    // Launching kenrel for device-0...
    cudaSetDevice(0);
    localAtomicsKernel << <grid_dim, block_dim >> > (device0_data);
    /*
        // Launching kenrel for device-0...
        cudaSetDevice(1);
        localAtomicsKernel << <grid_dim, block_dim >> > (device1_data);

        // Launching kenrel for device-0...
        cudaSetDevice(2);
        localAtomicsKernel << <grid_dim, block_dim >> > (device2_data);

        // Launching kenrel for device-0...
        cudaSetDevice(3);
        localAtomicsKernel << <grid_dim, block_dim >> > (device3_data);
    */
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    /*    cudaSetDevice(1);
        cudaDeviceSynchronize();
        cudaSetDevice(2);
        cudaDeviceSynchronize();
        cudaSetDevice(3);
        cudaDeviceSynchronize();
    */
    DATA* vthost = (DATA*)malloc(sizeof(DATA));

    cudaSetDevice(0);
    cudaMemcpy(vthost, device0_data, size, cudaMemcpyDeviceToHost);
    /*
        cudaSetDevice(1);
        cudaMemcpy(&vthost[1], device1_data, size, cudaMemcpyDeviceToHost);


        cudaSetDevice(2);
        cudaMemcpy(&vthost[2], device2_data, size, cudaMemcpyDeviceToHost);

        cudaSetDevice(3);
        cudaMemcpy(&vthost[3], device3_data, size, cudaMemcpyDeviceToHost);
    */
    //printf("%llu\n", vthost->ullData);
    printf("Total Atomics makde are %u\n", vthost->nData);
    cudaFree(device0_data);
    /*    cudaFree(device1_data);
        cudaFree(device2_data);
        cudaFree(device3_data);
    */
}

void remoteAtomics(int nGridSize, int nBlockSize)
{
    int nGPUCount = 0;
    cudaGetDeviceCount(&nGPUCount);

    std::cout << "Total GPUs: " << nGPUCount << std::endl;

    for (int i = 0; i < nGPUCount; i++)
    {
        cudaSetDevice(i);
        for (int j = 0; j < nGPUCount; j++)
        {
            if (i != j)
            {
                cudaDeviceEnablePeerAccess(j, 0);
                //cudaDeviceDisablePeerAccess(j);
            }
        }
    }

    size_t size = 1 * sizeof(DATA);

    /* For final-benchmarks-task I am commenting out the following code.. but it's in working condition
    // Initializing data for device-0...
    cudaSetDevice(0);

    DATA* device0_data;
    cudaMalloc(&device0_data, size);
    cudaMemset(&device0_data, 0, size);
    */

    // Initializing data for device-1...
    cudaSetDevice(1);

    DATA* device1_data;
    cudaMalloc(&device1_data, size);
    cudaMemset(&device1_data, 0, size);

    /* For final-benchmarks-task I am commenting out the following code.. but it's in working condition
    // Initializing data for device-2...
    cudaSetDevice(2);

    DATA* device2_data;
    cudaMalloc(&device2_data, size);
    cudaMemset(&device2_data, 0, size);

    // Initializing data for device-3...
    cudaSetDevice(3);

    DATA* device3_data;
    cudaMalloc(&device3_data, size);
    cudaMemset(&device3_data, 0, size);
    */

    dim3 grid_dim(nGridSize, 1, 1);
    dim3 block_dim(nBlockSize, 1, 1);

    // Launching kenrel for device-0...
    cudaSetDevice(0);
    remoteAtomicsKernel << <grid_dim, block_dim >> > (/*device0_data,*/ device1_data/*, device2_data, device3_data*/); // Params are commented our for final-benchmark-task

    /* For final-benchmarks-task I am commenting out the following code.. but it's in working condition
    // Launching kenrel for device-0...
    cudaSetDevice(1);
    remoteAtomicsKernel << <grid_dim, block_dim >> > (device1_data, device0_data, device2_data, device3_data);

    // Launching kenrel for device-0...
    cudaSetDevice(2);
    remoteAtomicsKernel << <grid_dim, block_dim >> > (device2_data, device0_data, device1_data, device3_data);

    // Launching kenrel for device-0...
    cudaSetDevice(3);
    remoteAtomicsKernel << <grid_dim, block_dim >> > (device3_data, device0_data, device1_data, device2_data);
    */

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    /* For final-benchmarks-task I am commenting out the following code.. but it's in working condition
    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaSetDevice(2);
    cudaDeviceSynchronize();
    cudaSetDevice(3);
    cudaDeviceSynchronize();
    */

    DATA* vthost = (DATA*)malloc(sizeof(DATA));

    /* For final-benchmarks-task I am commenting out the following code.. but it's in working condition
    cudaSetDevice(0);
    cudaMemcpy(&vthost[0], device0_data, size, cudaMemcpyDeviceToHost);
    */
    cudaSetDevice(1);
    cudaMemcpy(vthost, device1_data, size, cudaMemcpyDeviceToHost);

    /* For final-benchmarks-task I am commenting out the following code.. but it's in working condition
    cudaSetDevice(2);
    cudaMemcpy(&vthost[2], device2_data, size, cudaMemcpyDeviceToHost);

    cudaSetDevice(3);
    cudaMemcpy(&vthost[3], device3_data, size, cudaMemcpyDeviceToHost);
    */
    //printf("%llu\n", vthost->ullData);
    printf("Total Atomics makde are %u\n", vthost->nData);
    //cudaFree(device0_data);
    cudaFree(device1_data);
    //cudaFree(device2_data);
    //cudaFree(device3_data);
}

void localAtomicsEx(int nTotalAtomics, int nThreadCount, int nBlockSize)
{
    // Initializing data for device-0...
    cudaSetDevice(0);

    unsigned int* ptrData;
    cudaMalloc(&ptrData, sizeof(unsigned int));
    cudaMemset(&ptrData, 0, sizeof(unsigned int));

    int nGridSize = 1;

    if (nThreadCount > nBlockSize)
    {
        nGridSize = (int)(std::ceil((double)nThreadCount / nBlockSize));
    }
    else
    {
        nBlockSize = nThreadCount;
    }

    std::cout << "GridSize: " << nGridSize << ", BlockSize: " << nBlockSize << endl;

    // Launching kenrel for device-0...
    cudaSetDevice(0);
    localAtomicsKernelEx << <nGridSize, nBlockSize >> > (ptrData, nTotalAtomics, nGridSize * nBlockSize);

    cudaSetDevice(0);
    cudaDeviceSynchronize();

    unsigned int* ptrHostData = (unsigned int*)malloc(sizeof(unsigned int));

    cudaSetDevice(0);
    cudaMemcpy(ptrHostData, ptrData, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("%u\n", *ptrHostData);
    cudaFree(ptrData);

    delete ptrHostData;
}

typedef enum {
    P2P_WRITE = 0,
    P2P_READ = 1,
} P2PDataTransfer;

typedef enum {
    CE = 0,
    SM = 1,
} P2PEngine;

P2PEngine p2p_mechanism = CE;  // By default use Copy Engine

__global__ void copyp2p(int4* __restrict__ dest, int4 const* __restrict__ src,
    size_t num_elems) {
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;

#pragma unroll(5)
    for (size_t i = globalId; i < num_elems; i += gridSize) {
        dest[i] = src[i];
    }
}


void performP2PCopy(int* dest, int destDevice, int* src, int srcDevice,
    int num_elems, int repeat, bool p2paccess,
    cudaStream_t streamToRun) {
    int blockSize = 0;
    int numBlocks = 0;

    cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p);
    cudaCheckError();

    if (p2p_mechanism == SM && p2paccess) {
        for (int r = 0; r < repeat; r++) {
            // gridsise, blocksize, dynamically allocate memory, associated stream.
            copyp2p << <numBlocks, blockSize, 0, streamToRun >> > (
                (int4*)dest, (int4*)src, num_elems);
        }
    }
    else {
        for (int r = 0; r < repeat; r++) {
            cudaMemcpyPeerAsync(dest, destDevice, src, srcDevice,
                sizeof(int) * num_elems, streamToRun);
        }
    }
}

void copyData(int numElems, int numGPUs, bool p2p, P2PDataTransfer p2p_method, vector<double>& bandwidthMatrix)
{
    int repeat = 1;
    volatile int* flag = NULL;
    vector<int*> buffers(numGPUs);
    vector<int*> buffersD2D(numGPUs);  // buffer for D2D, that is, intra-GPU copy
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);
    vector<cudaStream_t> stream(numGPUs);

    cudaHostAlloc((void**)&flag, sizeof(*flag), cudaHostAllocPortable); //as a mutex
    cudaCheckError();

    for (int d = 0; d < numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
        cudaMalloc(&buffers[d], numElems * sizeof(int));
        cudaCheckError();
        cudaMemset(buffers[d], 0, numElems * sizeof(int));
        cudaCheckError();
        cudaMalloc(&buffersD2D[d], numElems * sizeof(int));
        cudaCheckError();
        cudaMemset(buffersD2D[d], 0, numElems * sizeof(int));
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    //        vector<double> bandwidthMatrix(numGPUs * numGPUs);

    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);

        for (int j = 0; j < numGPUs; j++) {
            int access = 0;
            if (p2p) {
                cudaDeviceCanAccessPeer(&access, i, j);
                if (access) {
                    cudaDeviceEnablePeerAccess(j, 0);
                    cudaCheckError();
                    cudaSetDevice(j);
                    cudaCheckError();
                    cudaDeviceEnablePeerAccess(i, 0);
                    cudaCheckError();
                    cudaSetDevice(i);
                    cudaCheckError();
                }
            }

            cudaStreamSynchronize(stream[i]);
            cudaCheckError();

            *flag = 0;
            delay << <1, 1, 0, stream[i] >> > (flag);
            cudaCheckError();
            cudaEventRecord(start[i], stream[i]);
            cudaCheckError();

            if (i == j) {
                // Perform intra-GPU, D2D copies
                performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat,
                    access, stream[i]);

            }
            else {
                if (p2p_method == P2P_WRITE) {
                    performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access,
                        stream[i]);
                }
                else {
                    performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access,
                        stream[i]);
                }
            }

            cudaEventRecord(stop[i], stream[i]);
            cudaCheckError();

            // Release the queued events
            *flag = 1;
            cudaStreamSynchronize(stream[i]);
            cudaCheckError();

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[i], stop[i]);
            //double time_s = time_ms / 1e3;

            //double gb = numElems * sizeof(int) * repeat / (double)1e9;
            //if (i == j) {
            //    gb *= 2;  // must count both the read and the write here
            //}
            bandwidthMatrix[i * numGPUs + j] += time_ms;// gb / time_s;
            if (p2p && access) {
                cudaDeviceDisablePeerAccess(j);
                cudaSetDevice(j);
                cudaDeviceDisablePeerAccess(i);
                cudaSetDevice(i);
                cudaCheckError();
            }
        }
    }

    /*printf("   D\\D");

    for (int j = 0; j < numGPUs; j++) {
        printf("%6d ", j);
    }

    printf("\n");

    for (int i = 0; i < numGPUs; i++) {
        printf("%6d ", i);

        for (int j = 0; j < numGPUs; j++) {
            printf("%6.02f ", bandwidthMatrix[i * numGPUs + j]);
        }

        printf("\n");
    }*/

    for (int d = 0; d < numGPUs; d++) {
        cudaSetDevice(d);
        cudaFree(buffers[d]);
        cudaFree(buffersD2D[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
        cudaStreamDestroy(stream[d]);
        cudaCheckError();
    }

    cudaFreeHost((void*)flag);
    cudaCheckError();
}


unsigned int* agg_input = NULL;
int current = 0;
size_t data_size = 1000;
int res_size = 1000;

void generate_uniform_random(size_t size, size_t ul) {

    std::random_device rd;
    std::mt19937 mte(rd());
    std::uniform_int_distribution<int> dist(0 + current, ul + current);
    std::generate_n(agg_input, data_size, [&]() {return dist(mte); });
}

size_t to_remove = 0;
void prepareData(short type = 0, size_t size = 1024, size_t inter = 8, double lambda = 0.05) {

    if (!agg_input)
        delete agg_input;
    current = 0;
    size_t to_remove = current;
    agg_input = (unsigned int*)calloc(size, sizeof(unsigned int));
    data_size = size;

    switch (type) {

    case 5: generate_uniform_random(data_size, inter);
        std::sort(agg_input, agg_input + data_size);
        break;

    default:
        std::cout << "Not distribution found" << std::endl;
        break;
    }

    res_size = (*std::max_element(agg_input, agg_input + data_size) + 1);
}

__global__
void combine_devices_results(unsigned int* result_d0, unsigned int* result_d1, unsigned int* result_d2, unsigned int* result_d3, unsigned int result_len)
{
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalId < result_len)
    {
        atomicAdd(&result_d0[globalId], result_d1[globalId]);
        atomicAdd(&result_d0[globalId], result_d2[globalId]);
        atomicAdd(&result_d0[globalId], result_d3[globalId]);
    }
}

__global__
void aggregate(unsigned int* data, unsigned int len, unsigned int* result)
{
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;

    if (globalId < len)
    {
        atomicAdd(&result[data[globalId]], 1);

    }
}

__global__
void aggregateEx(unsigned int* data, unsigned int* result_d0, unsigned int* result_d1, unsigned int* result_d2, unsigned int* result_d3, unsigned int data_len, unsigned int result_len, unsigned int groupSize)
{
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;

    int device_id = data[globalId] / groupSize;

    if (globalId < data_len)
    {
        switch (device_id)
        {
        case 0:
            atomicAdd(&result_d0[data[globalId]], 1);
            break;
        case 1:
            atomicAdd(&result_d1[data[globalId]], 1);
            break;
        case 2:
            atomicAdd(&result_d2[data[globalId]], 1);
            break;
        case 3:
            atomicAdd(&result_d3[data[globalId]], 1);
            break;
        }
    }
}

__global__
void aggregateExOpt(unsigned int* ptrData, unsigned int nDataLen, unsigned int* ptrLocalResult, unsigned int nResultLen, unsigned int* ptrPeer1Result, unsigned int nPeer1GroupId, unsigned int* ptrPeer2Result, unsigned int nPeer2GroupId, unsigned int* ptrPeer3Result, unsigned nPeer3GroupId, unsigned int nGroupSize, unsigned int nTotalThreads, unsigned int* nProcessedThreads)
{
    size_t nGlobalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;

    if (nGlobalId < nDataLen)
    {
        atomicAdd(&ptrLocalResult[ptrData[nGlobalId]], 1);
    }

    unsigned int nProcessedThreads_Old = atomicAdd(nProcessedThreads, 1);

    if (nProcessedThreads_Old + 1 == nTotalThreads)
    {
        for (int i = 0; i < nResultLen; i++)
        {
            int nGroupId = i / nGroupSize;

            if (nGroupId == nPeer1GroupId)
            {
                atomicAdd(&ptrPeer1Result[i], ptrLocalResult[i]);
            }
            else if (nGroupId == nPeer2GroupId)
            {
                atomicAdd(&ptrPeer2Result[i], ptrLocalResult[i]);
            }
            else if (nGroupId == nPeer3GroupId)
            {
                atomicAdd(&ptrPeer3Result[i], ptrLocalResult[i]);
            }
        }
    }
}

void initDeviceMemory(unsigned int id, unsigned int** ptrDeviceData, unsigned int data_len, unsigned int** ptrDeviceResult, unsigned int result_len)
{
    cudaSetDevice(id);

    cudaMalloc(&ptrDeviceData[id], data_len * sizeof(unsigned int));
    cudaMemcpy(ptrDeviceData[id], &agg_input[id * data_len], data_len * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc(&ptrDeviceResult[id], result_len * sizeof(unsigned int));
    cudaMemset(&ptrDeviceResult[id], 0, result_len * sizeof(unsigned int));
}

void initDeviceMemoryOpt(unsigned int id, unsigned int** ptrDeviceData, unsigned int data_len, unsigned int** ptrDeviceResult, unsigned int result_len, unsigned int** ptrDeviceProcessedThreads)
{
    cudaSetDevice(id);

    cudaMalloc(&ptrDeviceData[id], data_len * sizeof(unsigned int));
    cudaMemcpy(ptrDeviceData[id], &agg_input[id * data_len], data_len * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc(&ptrDeviceResult[id], result_len * sizeof(unsigned int));
    cudaMemset(&ptrDeviceResult[id], 0, result_len * sizeof(unsigned int));

    cudaMalloc(&ptrDeviceProcessedThreads[id], 4 * sizeof(unsigned int));
    cudaMemset(&ptrDeviceProcessedThreads[id], 0, 4 * sizeof(unsigned int));
}

void launchKernel(int nNumBlocks, int nBlockSize, unsigned int id, unsigned int** ptrDeviceData, unsigned int data_len, unsigned int** ptrDeviceResult)
{
    cudaSetDevice(id);
    aggregate << <nNumBlocks, nBlockSize >> > (ptrDeviceData[id], data_len, ptrDeviceResult[id]);
}

void launchKernelEx(int nNumBlocks, int nBlockSize, unsigned int id, unsigned int** ptrDeviceData, unsigned int data_len, unsigned int** ptrDeviceResult, unsigned int result_len, unsigned int groupSize)
{
    cudaSetDevice(id);
    aggregateEx << <nNumBlocks, nBlockSize >> > (ptrDeviceData[id], ptrDeviceResult[0], ptrDeviceResult[1], ptrDeviceResult[2], ptrDeviceResult[3], data_len, result_len, groupSize);
}

void launchKernelExOpt(int nNumBlocks, int nBlockSize, unsigned int** ptrData, unsigned int nDataLen, unsigned int** ptrResult, unsigned int nResultLen, unsigned int nGroupSize, unsigned int** ptrProcessedThreads)
{
    unsigned int nPeer1GroupId = 0;
    unsigned int nPeer2GroupId = 1;
    unsigned int nPeer3GroupId = 2;
    unsigned int nPeer4GroupId = 3;

    unsigned int nTotalThreads = nNumBlocks * nBlockSize;

    cudaSetDevice(0);
    //unsigned int* nTotalThreads_d0;
    //cudaMalloc(&nTotalThreads_d0, sizeof(unsigned int));
    //cudaMemcpy(&nTotalThreads_d0, &nTotalThreads, sizeof(unsigned int), cudaMemcpyHostToDevice);
    aggregateExOpt << <nNumBlocks, nBlockSize >> > (ptrData[0], nDataLen, ptrResult[0], nResultLen, ptrResult[1], 1, ptrResult[2], 2, ptrResult[3], 3, nGroupSize, nTotalThreads, ptrProcessedThreads[0]);

    cudaSetDevice(1);
    //unsigned int* nTotalThreads_d1;
    //cudaMalloc(&nTotalThreads_d1, sizeof(unsigned int));
    //cudaMemcpy(&nTotalThreads_d1, &nTotalThreads, sizeof(unsigned int), cudaMemcpyHostToDevice);
    aggregateExOpt << <nNumBlocks, nBlockSize >> > (ptrData[1], nDataLen, ptrResult[1], nResultLen, ptrResult[0], 0, ptrResult[2], 2, ptrResult[3], 3, nGroupSize, nTotalThreads, ptrProcessedThreads[1]);

    cudaSetDevice(2);
    //unsigned int* nTotalThreads_d2;
    //cudaMalloc(&nTotalThreads_d2, sizeof(unsigned int));
    //cudaMemcpy(&nTotalThreads_d2, &nTotalThreads, sizeof(unsigned int), cudaMemcpyHostToDevice);
    aggregateExOpt << <nNumBlocks, nBlockSize >> > (ptrData[2], nDataLen, ptrResult[2], nResultLen, ptrResult[0], 0, ptrResult[1], 1, ptrResult[3], 3, nGroupSize, nTotalThreads, ptrProcessedThreads[2]);

    cudaSetDevice(3);
    //unsigned int* nTotalThreads_d3;
    //cudaMalloc(&nTotalThreads_d3, sizeof(unsigned int));
    //cudaMemcpy(&nTotalThreads_d3, &nTotalThreads, sizeof(unsigned int), cudaMemcpyHostToDevice);
    aggregateExOpt << <nNumBlocks, nBlockSize >> > (ptrData[3], nDataLen, ptrResult[3], nResultLen, ptrResult[0], 0, ptrResult[1], 1, ptrResult[2], 2, nGroupSize, nTotalThreads, ptrProcessedThreads[3]);
}

void copyDataFromDevice(unsigned int id, unsigned int* data, unsigned int data_len, unsigned int** ptrDeviceData, unsigned int* result, unsigned int result_len, unsigned int** ptrDeviceResult)
{
    cudaSetDevice(id);
    cudaMemcpy(&result[id * result_len], ptrDeviceResult[id], result_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&data[id * data_len], ptrDeviceData[id], data_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

void copyDataFromDeviceEx(unsigned int id, unsigned int* data, unsigned int data_len, unsigned int** ptrDeviceData, unsigned int* result, unsigned int result_len, unsigned int** ptrDeviceResult)
{
    cudaSetDevice(id);
    cudaMemcpy(&result[id * result_len], ptrDeviceResult[id] + (id * result_len), result_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&data[id * data_len], ptrDeviceData[id], data_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

void testAggregateOperationSortedData(size_t nDataSizeMultiplier, unsigned int** ptrDeviceData, unsigned int nDataSizePerGPU, unsigned int** ptrDeviceResult, unsigned int nMaxRandomNumber)
{
    initDeviceMemory(0, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(1, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(2, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(3, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);

    int nNumBlocks = 0;
    int nBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&nNumBlocks, &nBlockSize, aggregateEx);

    nNumBlocks = nDataSizeMultiplier;

    launchKernel(nNumBlocks, nBlockSize, 0, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);
    launchKernel(nNumBlocks, nBlockSize, 1, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);
    launchKernel(nNumBlocks, nBlockSize, 2, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);
    launchKernel(nNumBlocks, nBlockSize, 3, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaSetDevice(2);
    cudaDeviceSynchronize();
    cudaSetDevice(3);
    cudaDeviceSynchronize();

    unsigned int* ptrAllDevicesData = (unsigned int*)malloc(nDataSizePerGPU * 4 * sizeof(unsigned int));
    memset(ptrAllDevicesData, 0, nDataSizePerGPU * 4 * sizeof(unsigned int));

    unsigned int* ptrAllDevicesResults = (unsigned int*)malloc(nMaxRandomNumber * 4 * sizeof(unsigned int));
    memset(ptrAllDevicesResults, 0, nMaxRandomNumber * 4 * sizeof(unsigned int));

    copyDataFromDevice(0, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber, ptrDeviceResult);
    copyDataFromDevice(1, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber, ptrDeviceResult);
    copyDataFromDevice(2, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber, ptrDeviceResult);
    copyDataFromDevice(3, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber, ptrDeviceResult);

    unsigned int nTotalUniqueRecords = 0;
    for (int j = 0; j < nMaxRandomNumber * 4; j++)
    {
        nTotalUniqueRecords += ptrAllDevicesResults[j];
        std::cout << j % nMaxRandomNumber << " : " << ptrAllDevicesResults[j] << std::endl;
    }

    for (int i = 0; i < nDataSizePerGPU * 4; i++)
    {
        if (ptrAllDevicesData[i] != agg_input[i])
            std::cout << "Error,";
    }

    std::cout << std::endl;

    std::cout << "Final: " << nTotalUniqueRecords << std::endl;

    cudaFree(ptrDeviceData[0]);
    cudaFree(ptrDeviceData[1]);
    cudaFree(ptrDeviceData[2]);
    cudaFree(ptrDeviceData[3]);

    cudaFree(ptrDeviceResult[0]);
    cudaFree(ptrDeviceResult[1]);
    cudaFree(ptrDeviceResult[2]);
    cudaFree(ptrDeviceResult[3]);

    delete[] ptrAllDevicesData;
    delete[] ptrAllDevicesResults;
}

std::chrono::duration<double> testAggregateOperationSortedDataOpt(int nDataSizeMultiplier, unsigned int** ptrDeviceData, unsigned int nDataSizePerGPU, unsigned int** ptrDeviceResult, unsigned int nMaxRandomNumber)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    initDeviceMemory(0, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(1, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(2, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(3, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);

    int nNumBlocks = 0;
    int nBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&nNumBlocks, &nBlockSize, aggregateEx);

    nNumBlocks = nDataSizeMultiplier;

    std::cout << "Local/Remote Atomics across individual GPUs. GridSize: " << nNumBlocks << ", BlockSize: " << nBlockSize << std::endl;

    launchKernelEx(nNumBlocks, nBlockSize, 0, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, nMaxRandomNumber / 4);
    launchKernelEx(nNumBlocks, nBlockSize, 1, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, nMaxRandomNumber / 4);
    launchKernelEx(nNumBlocks, nBlockSize, 2, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, nMaxRandomNumber / 4);
    launchKernelEx(nNumBlocks, nBlockSize, 3, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, nMaxRandomNumber / 4);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaSetDevice(2);
    cudaDeviceSynchronize();
    cudaSetDevice(3);
    cudaDeviceSynchronize();

    unsigned int* ptrAllDevicesData = (unsigned int*)malloc(nDataSizePerGPU * 4 * sizeof(unsigned int));
    memset(ptrAllDevicesData, 0, nDataSizePerGPU * 4 * sizeof(unsigned int));

    unsigned int* ptrAllDevicesResults = (unsigned int*)malloc(nMaxRandomNumber * sizeof(unsigned int));
    memset(ptrAllDevicesResults, 0, nMaxRandomNumber * sizeof(unsigned int));

    copyDataFromDeviceEx(0, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber / 4, ptrDeviceResult);
    copyDataFromDeviceEx(1, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber / 4, ptrDeviceResult);
    copyDataFromDeviceEx(2, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber / 4, ptrDeviceResult);
    copyDataFromDeviceEx(3, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber / 4, ptrDeviceResult);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    unsigned int nTotalUniqueRecords = 0;
    for (int j = 0; j < nMaxRandomNumber; j++)
    {
        nTotalUniqueRecords += ptrAllDevicesResults[j];
        //std::cout << j << " : " << ptrAllDevicesResults[j] << std::endl;
    }

    for (int i = 0; i < nDataSizePerGPU * 4; i++)
    {
        if (ptrAllDevicesData[i] != agg_input[i])
            std::cout << "Error,";
    }

    //std::cout << std::endl;

    //std::cout << "final: " << nTotalUniqueRecords << std::endl;
    if (nTotalUniqueRecords != nDataSizePerGPU * 4)
        std::cout << "**Error**" << std::endl;


    cudaFree(ptrDeviceData[0]);
    cudaFree(ptrDeviceData[1]);
    cudaFree(ptrDeviceData[2]);
    cudaFree(ptrDeviceData[3]);

    cudaFree(ptrDeviceResult[0]);
    cudaFree(ptrDeviceResult[1]);
    cudaFree(ptrDeviceResult[2]);
    cudaFree(ptrDeviceResult[3]);

    delete[] ptrAllDevicesData;
    delete[] ptrAllDevicesResults;

    return time_span;
}

void testAggregateOperationSortedDataOptEx(size_t nDataSizeMultiplier, unsigned int** ptrDeviceData, unsigned int nDataSizePerGPU, unsigned int** ptrDeviceResult, unsigned int nMaxRandomNumber, unsigned int** ptrDeviceProcessedThreads)
{
    initDeviceMemoryOpt(0, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, ptrDeviceProcessedThreads);
    initDeviceMemoryOpt(1, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, ptrDeviceProcessedThreads);
    initDeviceMemoryOpt(2, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, ptrDeviceProcessedThreads);
    initDeviceMemoryOpt(3, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, ptrDeviceProcessedThreads);

    int nNumBlocks = 0;
    int nBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&nNumBlocks, &nBlockSize, aggregateEx);

    nNumBlocks = nDataSizeMultiplier;

    launchKernelExOpt(nNumBlocks, nBlockSize, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, nMaxRandomNumber / 4, ptrDeviceProcessedThreads);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaSetDevice(2);
    cudaDeviceSynchronize();
    cudaSetDevice(3);
    cudaDeviceSynchronize();

    unsigned int* ptrAllDevicesData = (unsigned int*)malloc(nDataSizePerGPU * 4 * sizeof(unsigned int));
    memset(ptrAllDevicesData, 0, nDataSizePerGPU * 4 * sizeof(unsigned int));

    unsigned int* ptrAllDevicesResults = (unsigned int*)malloc(nMaxRandomNumber * sizeof(unsigned int));
    memset(ptrAllDevicesResults, 0, nMaxRandomNumber * sizeof(unsigned int));

    copyDataFromDeviceEx(0, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber / 4, ptrDeviceResult);
    copyDataFromDeviceEx(1, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber / 4, ptrDeviceResult);
    copyDataFromDeviceEx(2, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber / 4, ptrDeviceResult);
    copyDataFromDeviceEx(3, ptrAllDevicesData, nDataSizePerGPU, ptrDeviceData, ptrAllDevicesResults, nMaxRandomNumber / 4, ptrDeviceResult);

    unsigned int nTotalUniqueRecords = 0;
    for (int j = 0; j < nMaxRandomNumber; j++)
    {
        nTotalUniqueRecords += ptrAllDevicesResults[j];
        std::cout << j << " : " << ptrAllDevicesResults[j] << std::endl;
    }

    for (int i = 0; i < nDataSizePerGPU * 4; i++)
    {
        if (ptrAllDevicesData[i] != agg_input[i])
            std::cout << "Error,";
    }

    std::cout << std::endl;

    std::cout << "Final: " << nTotalUniqueRecords << std::endl;


    unsigned int* ptrProcessedthreads = (unsigned int*)malloc(4 * sizeof(unsigned int));
    memset(ptrProcessedthreads, 0, 4 * sizeof(unsigned int));

    cudaSetDevice(0);
    cudaMemcpy(&ptrProcessedthreads[0], ptrDeviceProcessedThreads[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "Total threads: " << nNumBlocks * nBlockSize << ", processed at GPU-0: " << ptrProcessedthreads[0] << std::endl;


    cudaFree(ptrDeviceData[0]);
    cudaFree(ptrDeviceData[1]);
    cudaFree(ptrDeviceData[2]);
    cudaFree(ptrDeviceData[3]);

    cudaFree(ptrDeviceResult[0]);
    cudaFree(ptrDeviceResult[1]);
    cudaFree(ptrDeviceResult[2]);
    cudaFree(ptrDeviceResult[3]);

    delete[] ptrAllDevicesData;
    delete[] ptrAllDevicesResults;
    delete ptrProcessedthreads;
}

std::chrono::duration<double>  testAggregateOperationBaseline(size_t nDataSizeMultiplier, unsigned int** ptrDeviceData, unsigned int nDataSizePerGPU, unsigned int** ptrDeviceResult, unsigned int nMaxRandomNumber)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    initDeviceMemory(0, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(1, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(2, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(3, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);

    int nNumBlocks = 0;
    int nBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&nNumBlocks, &nBlockSize, aggregateEx);

    nNumBlocks = nDataSizeMultiplier;

    launchKernel(nNumBlocks, nBlockSize, 0, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);
    launchKernel(nNumBlocks, nBlockSize, 1, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);
    launchKernel(nNumBlocks, nBlockSize, 2, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);
    launchKernel(nNumBlocks, nBlockSize, 3, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaSetDevice(2);
    cudaDeviceSynchronize();
    cudaSetDevice(3);
    cudaDeviceSynchronize();

    unsigned int* ptrAllDevicesData = (unsigned int*)malloc(nDataSizePerGPU * 4 * sizeof(unsigned int));
    memset(ptrAllDevicesData, 0, nDataSizePerGPU * 4 * sizeof(unsigned int));

    unsigned int* ptrAllDevicesResults = (unsigned int*)malloc(nMaxRandomNumber * sizeof(unsigned int));
    memset(ptrAllDevicesResults, 0, nMaxRandomNumber * sizeof(unsigned int));

    unsigned int* ptrDeviceResult_Temp = (unsigned int*)malloc(nMaxRandomNumber * sizeof(unsigned int));
    memset(ptrDeviceResult_Temp, 0, nMaxRandomNumber * sizeof(unsigned int));



    for (int i = 0; i < 4; i++)
    {
        cudaSetDevice(i);
        cudaMemcpy(ptrDeviceResult_Temp, ptrDeviceResult[i], nMaxRandomNumber * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ptrAllDevicesData[i * nDataSizePerGPU], ptrDeviceData[i], nDataSizePerGPU * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        for (int j = 0; j < nMaxRandomNumber; j++)
        {
            ptrAllDevicesResults[j] += ptrDeviceResult_Temp[j];
        }
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    unsigned int nTotalUniqueRecords = 0;
    for (int j = 0; j < nMaxRandomNumber; j++)
    {
        nTotalUniqueRecords += ptrAllDevicesResults[j];
        //std::cout << j << " : " << ptrAllDevicesResults[j] << std::endl;
    }

    for (int i = 0; i < nDataSizePerGPU * 4; i++)
    {
        if (ptrAllDevicesData[i] != agg_input[i])
            std::cout << "Error,";
    }




    std::cout << std::endl;

    std::cout << "Final: " << nTotalUniqueRecords << std::endl;

    cudaFree(ptrDeviceData[0]);
    cudaFree(ptrDeviceData[1]);
    cudaFree(ptrDeviceData[2]);
    cudaFree(ptrDeviceData[3]);

    cudaFree(ptrDeviceResult[0]);
    cudaFree(ptrDeviceResult[1]);
    cudaFree(ptrDeviceResult[2]);
    cudaFree(ptrDeviceResult[3]);

    delete[] ptrAllDevicesData;
    delete[] ptrAllDevicesResults;
    delete[] ptrDeviceResult_Temp;

    return time_span;
}

std::chrono::duration<double>  testAggregateOperationBaselineEx(size_t nDataSizeMultiplier, unsigned int** ptrDeviceData, unsigned int nDataSizePerGPU, unsigned int** ptrDeviceResult, unsigned int nMaxRandomNumber)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    initDeviceMemory(0, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(1, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(2, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
    initDeviceMemory(3, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);

    int nNumBlocks = 0;
    int nBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&nNumBlocks, &nBlockSize, aggregateEx);

    nNumBlocks = nDataSizeMultiplier;

    //std::cout << "Local Atomics across individual GPUs. GridSize: " << nNumBlocks << ", BlockSize: " << nBlockSize << std::endl;

    launchKernel(nNumBlocks, nBlockSize, 0, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);
    launchKernel(nNumBlocks, nBlockSize, 1, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);
    launchKernel(nNumBlocks, nBlockSize, 2, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);
    launchKernel(nNumBlocks, nBlockSize, 3, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaSetDevice(2);
    cudaDeviceSynchronize();
    cudaSetDevice(3);
    cudaDeviceSynchronize();


    if (nMaxRandomNumber > 32)
    {
        nBlockSize = 32;
        nNumBlocks = (int)(std::ceil((double)nMaxRandomNumber / 32));
        //	std::cout << "*** " <<  (int)(std::ceil((double)nMaxRandomNumber/(double)32))<< std::endl;
    }
    else
    {
        nBlockSize = 32;
        nNumBlocks = 1;
        //	std::cout << " ++++" <<std::endl;
    }

    //std::cout << "Local Atomics (merge results) across GPU-0. GridSize: " << nNumBlocks << ", BlockSize: " << nBlockSize << std::endl;

    cudaSetDevice(0);

    unsigned int* ptrResult_d1_copy, * ptrResult_d2_copy, * ptrResult_d3_copy;

    cudaMalloc(&ptrResult_d1_copy, nMaxRandomNumber * sizeof(unsigned int));
    cudaMemcpy(ptrResult_d1_copy, ptrDeviceResult[1], nMaxRandomNumber * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    cudaMalloc(&ptrResult_d2_copy, nMaxRandomNumber * sizeof(unsigned int));
    cudaMemcpy(ptrResult_d2_copy, ptrDeviceResult[2], nMaxRandomNumber * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    cudaMalloc(&ptrResult_d3_copy, nMaxRandomNumber * sizeof(unsigned int));
    cudaMemcpy(ptrResult_d3_copy, ptrDeviceResult[3], nMaxRandomNumber * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    std::cout << nMaxRandomNumber << "-Kernel: combine_devices_results; GridSize: " << nNumBlocks << ", BlockSize: " << nBlockSize << std::endl;

    combine_devices_results << <nNumBlocks, nBlockSize >> > (ptrDeviceResult[0], ptrResult_d1_copy, ptrResult_d2_copy, ptrResult_d3_copy, nMaxRandomNumber);

    cudaSetDevice(0);
    cudaDeviceSynchronize();


    unsigned int* ptrAllDevicesData = (unsigned int*)malloc(nDataSizePerGPU * 4 * sizeof(unsigned int));
    memset(ptrAllDevicesData, 0, nDataSizePerGPU * 4 * sizeof(unsigned int));

    unsigned int* ptrAllDevicesResults = (unsigned int*)malloc(nMaxRandomNumber * sizeof(unsigned int));
    memset(ptrAllDevicesResults, 0, nMaxRandomNumber * sizeof(unsigned int));

    cudaSetDevice(0);
    cudaMemcpy(ptrAllDevicesResults, ptrDeviceResult[0], nMaxRandomNumber * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++)
    {
        cudaSetDevice(i);
        //cudaMemcpy(ptrDeviceResult_Temp, ptrDeviceResult[i], nMaxRandomNumber * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ptrAllDevicesData[i * nDataSizePerGPU], ptrDeviceData[i], nDataSizePerGPU * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        //for (int j = 0; j < nMaxRandomNumber; j++)
        //{
        //    ptrAllDevicesResults[j] += ptrDeviceResult_Temp[j];
       // }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    unsigned int nTotalUniqueRecords = 0;
    for (int j = 0; j < nMaxRandomNumber; j++)
    {
        nTotalUniqueRecords += ptrAllDevicesResults[j];
        //std::cout << j << " : " << ptrAllDevicesResults[j] << std::endl;
    }

    for (int i = 0; i < nDataSizePerGPU * 4; i++)
    {
        if (ptrAllDevicesData[i] != agg_input[i])
            std::cout << "Error,";
    }




    //std::cout << std::endl;

    if (nTotalUniqueRecords != nDataSizePerGPU * 4)
        std::cout << "**Error**" << std::endl;

    cudaFree(ptrDeviceData[0]);
    cudaFree(ptrDeviceData[1]);
    cudaFree(ptrDeviceData[2]);
    cudaFree(ptrDeviceData[3]);

    cudaFree(ptrDeviceResult[0]);
    cudaFree(ptrDeviceResult[1]);
    cudaFree(ptrDeviceResult[2]);
    cudaFree(ptrDeviceResult[3]);

    cudaFree(ptrResult_d1_copy);
    cudaFree(ptrResult_d2_copy);
    cudaFree(ptrResult_d3_copy);

    delete[] ptrAllDevicesData;
    delete[] ptrAllDevicesResults;

    return time_span;
}

int main(int argc, char** argv)
{
    int nRepeat = 10;

    std::cout << "Program started.." << std::endl;

    if (std::string(argv[1]) == "s1"
        || std::string(argv[1]) == "s2"
        || std::string(argv[1]) == "s3"
        || std::string(argv[1]) == "b1"
        || std::string(argv[1]) == "b2")
    {
        size_t nDataSizeMultiplier = atoi(argv[2]);    // Default: 1
        int nRandomNumberMaxLimit = atoi(argv[3]);  // Default: 4

        int nGPUCount = 0;
        cudaGetDeviceCount(&nGPUCount);
        //std::cout << "nTotalUniqueRecords GPUs: " << nGPUCount << std::endl;

        size_t nMaxDataSize = 1024 * nDataSizeMultiplier;
        size_t nDataSizePerGPU = nMaxDataSize / nGPUCount;
        size_t nMaxRandomNumber = nGPUCount * nRandomNumberMaxLimit - 1;

        std::cout << "TotalDataSize: " << nMaxDataSize << ", DataSizePerGPU: " << nDataSizePerGPU << ", MaxRandomNumber: " << nMaxRandomNumber << std::endl;

        prepareData(5, nMaxDataSize, nMaxRandomNumber);

        nMaxRandomNumber++;     // Random number start from 0, so to make it divisible by 4 (GPU count) incrementing it by 1.

        unsigned int** ptrDeviceData = new unsigned int* [nGPUCount];
        unsigned int** ptrDeviceResult = new unsigned int* [nGPUCount];
        unsigned int** ptrDeviceProcessedThreads = new unsigned int* [nGPUCount];

        for (int i = 0; i < nGPUCount; i++)
        {
            cudaSetDevice(i);
            for (int j = 0; j < nGPUCount; j++)
            {
                if (i != j)
                {
                    cudaDeviceEnablePeerAccess(j, 0);
                    //cudaDeviceDisablePeerAccess(j);
                }
            }
        }

        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(now - now);
        for (int i = 0; i < nRepeat; i++)
        {
            //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

            if (std::string(argv[1]) == "s1")
            {
                testAggregateOperationSortedData(nDataSizeMultiplier, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
            }
            else if (std::string(argv[1]) == "s2")
            {
                time_span += testAggregateOperationSortedDataOpt(nDataSizeMultiplier, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
            }
            else if (std::string(argv[1]) == "s3")
            {
                testAggregateOperationSortedDataOptEx(nDataSizeMultiplier, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber, ptrDeviceProcessedThreads);
            }
            else if (std::string(argv[1]) == "b1")
            {
                time_span += testAggregateOperationBaseline(nDataSizeMultiplier, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
            }
            else if (std::string(argv[1]) == "b2")
            {
                time_span += testAggregateOperationBaselineEx(nDataSizeMultiplier, ptrDeviceData, nDataSizePerGPU, ptrDeviceResult, nMaxRandomNumber);
            }
            else
            {
                std::cout << "Please see the code file for options." << std::endl;
            }

            //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            //time_span += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        }
        std::cout << " CPU time: " << (time_span.count() / (double)nRepeat) << std::endl;

        if (argc >= 5)
        {
            // Create an output filestream object
            std::ofstream myFile(argv[4]);

            // Send data to the stream
            myFile << "CPU time" << "\n";
            myFile << time_span.count() << "\n";

            // Close the file
            myFile.close();
        }

        delete[] ptrDeviceData;
        delete[] ptrDeviceResult;
        delete[] ptrDeviceProcessedThreads;

        return 0;
    }


    if (std::string(argv[1]) == "la"
        || std::string(argv[1]) == "laex"
        || std::string(argv[1]) == "ra"
        || std::string(argv[1]) == "bt")
    {
        if (std::string(argv[1]) == "la")
        {
            int nGridSize = 0;
            int nBlockSize = 0;

            if (argc > 2)
            {
                nGridSize = atoi(argv[2]);
            }

            if (argc > 3)
            {
                nBlockSize = atoi(argv[3]);
            }

            for (int i = 0; i < nRepeat; i++)
            {
                localAtomics(nGridSize, nBlockSize);
            }
        }
        if (std::string(argv[1]) == "laex")
        {
            int nTotalAtomics = atoi(argv[2]);
            int nTotalThreads = atoi(argv[3]);
            int nBlockSize = atoi(argv[4]);

            for (int i = 0; i < nRepeat; i++)
            {
                localAtomicsEx(nTotalAtomics, nTotalThreads, nBlockSize);
            }
        }
        else if (std::string(argv[1]) == "ra")
        {
            int nGridSize = 0;
            int nBlockSize = 0;

            if (argc > 2)
            {
                nGridSize = atoi(argv[2]);
            }

            if (argc > 3)
            {
                nBlockSize = atoi(argv[3]);
            }

            for (int i = 0; i < nRepeat; i++)
            {
                remoteAtomics(nGridSize, nBlockSize);
            }
        }
        else if (std::string(argv[1]) == "bt")
        {
            int nGPUCount;
            cudaGetDeviceCount(&nGPUCount);

            int nTupleSize = 100000;
            if (argc > 2)
            {
                nTupleSize = atoi(argv[2]);
            }

            bool p2p = true;
            if (argc > 3)
            {
                if (std::string(argv[3]) == "false")
                {
                    p2p = false;
                }
            }

            vector<double> bandwidthMatrix(nGPUCount * nGPUCount);

            for (int i = 0; i < 100; i++)
            {
                copyData(nTupleSize, nGPUCount, p2p, P2P_WRITE, bandwidthMatrix);
            }


            printf("   D\\D");

            for (int j = 0; j < nGPUCount; j++) {
                printf("%6d ,", j);
            }

            printf("\n");

            for (int i = 0; i < nGPUCount; i++) {
                printf("%6d ,", i);

                for (int j = 0; j < nGPUCount; j++) {
                    printf("%6.02f ,", bandwidthMatrix[i * nGPUCount + j] / 100);
                }

                printf("\n");
            }

        }

        return 0;
    }

    return 0;
}
