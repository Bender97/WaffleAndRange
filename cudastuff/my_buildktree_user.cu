// ======================================================================== //
// Copyright 2018-2023 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "cukd/builder.h"
// fcp = "find closest point" query
#include "cukd/fcp.h"
#include <queue>
#include <iomanip>
#include <random>
#include "cukd/knn.h"
using namespace cukd;
using namespace cukd::common;
#include <cuda_runtime_api.h>
#include <mutex>

#define k 17

#define MAX_INPUT_PC_SIZE 200000
#define MAX_QUERY_SIZE 200000

struct Neig {
  int elem[17];
};

template<typename CandidateList>
__global__
void d_knn(Neig   *d_results,
           float3  *d_queries,
           int      numQueries,
           const cukd::box_t<float3> *d_bounds,
           float3  *d_nodes,
           int      numNodes,
           float    cutOffRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;
  
  CandidateList result(cutOffRadius);
  float sqrDist
    = cukd::cct::knn<CandidateList,float3,cukd::default_data_traits<float3>>
    (result,
     d_queries[tid],
     *d_bounds,
     d_nodes,numNodes);

  for (int i=0; i<k; i++) d_results[tid].elem[i] = result.get_pointID(i);

}


template<typename CandidateList>
__global__
void d_knn_1(int   *d_results,
           float3  *d_queries,
           int      numQueries,
           const cukd::box_t<float3> *d_bounds,
           float3  *d_nodes,
           int      numNodes,
           float    cutOffRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;
  
  CandidateList result(cutOffRadius);
  float sqrDist
    = cukd::cct::knn<CandidateList,float3,cukd::default_data_traits<float3>>
    (result,
     d_queries[tid],
     *d_bounds,
     d_nodes,numNodes);

  d_results[tid] = result.get_pointID(0);

}

class MyFastKDTree {

public:

  MyFastKDTree() {}

  void run(float *indatav,    size_t tree_size, 
           float *queries_k,  size_t numQueries_k,
           float *queries_1,  size_t numQueries_1,
           int*  d_results_k, int* d_results_1
  ) {
    

    float3 *d_points;
    float3 *d_queries_1;
    float3 *d_queries_k;
    cukd::box_t<float3> *d_bounds;
    float cutOffRadius;
    int bs,nb;
    Neig  *results_k;
    int  *results_1;
    //cudaEvent_t start, stop;


    tree_size /= 3;
    numQueries_k /= 3;
    numQueries_1 /= 3;
    bs = 128;
    cutOffRadius = std::numeric_limits<float>::infinity();

//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);

    cudaMallocManaged((char**)&d_points, tree_size*sizeof(*d_points));
    cudaMallocManaged((void**)&d_bounds, sizeof(cukd::box_t<float3>));

    cudaMallocManaged((char**)&d_queries_k, numQueries_k*sizeof(*d_queries_k));
    cudaMallocManaged((char**)&d_queries_1, numQueries_1*sizeof(*d_queries_1));

    cudaMallocManaged((void**)&results_k, k*numQueries_k*sizeof(results_k));
    cudaMallocManaged((void**)&results_1, numQueries_1*sizeof(results_1));
    

    //////////////    BUILD TREE    ////////////////

    //cudaEventRecord(start);
    for (int i=0, j=0;i<tree_size;i++) {
      d_points[i].x = indatav[j++];
      d_points[i].y = indatav[j++];
      d_points[i].z = indatav[j++];
    }
    cukd::buildTree(d_points, tree_size, d_bounds);
    cudaDeviceSynchronize();


    //////////////    QUERY K    ////////////////

    nb = divRoundUp((int)numQueries_k,bs);

    for (int i=0, j=0;i<numQueries_k;i++) {
      d_queries_k[i].x = queries_k[j++];
      d_queries_k[i].y = queries_k[j++];
      d_queries_k[i].z = queries_k[j++];
    }

    d_knn<FixedCandidateList<k>><<<nb,bs>>>
          (results_k,
           d_queries_k, numQueries_k,
           d_bounds,
           d_points, tree_size, cutOffRadius);
    
    cudaDeviceSynchronize();

    cudaMemcpy(d_results_k, results_k, k * numQueries_k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    //////////////    QUERY 1    ////////////////

    nb = divRoundUp((int)numQueries_1,bs);

    for (int i=0, j=0;i<numQueries_1;i++) {
      d_queries_1[i].x = queries_1[j++];
      d_queries_1[i].y = queries_1[j++];
      d_queries_1[i].z = queries_1[j++];
    }

    d_knn_1<FixedCandidateList<1>><<<nb,bs>>>
          (results_1,
           d_queries_1, numQueries_1,
           d_bounds,
           d_points, tree_size, cutOffRadius);
    
    cudaDeviceSynchronize();

    cudaMemcpy(d_results_1, results_1, numQueries_1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//
//    std::cout << "build time " << milliseconds << " ms" << std::endl;

    cudaFree(d_points);
    cudaFree(results_k);
    cudaFree(results_1);
    
  }

  
};
  

extern "C" {

  MyFastKDTree* MyFastKDTree_new(){ return new MyFastKDTree(); }
  void MyFastKDTree_run(MyFastKDTree* foo,
                        float *indatav,    size_t tree_size, 
                        float *queries_k,  size_t numQueries_k,
                        float *queries_1,  size_t numQueries_1,
                        int*  d_results_k, int* d_results_1
                        ){
                          foo->run(indatav, tree_size,
                                        queries_k, numQueries_k,
                                        queries_1, numQueries_1,
                                        d_results_k, d_results_1);
                        }
}