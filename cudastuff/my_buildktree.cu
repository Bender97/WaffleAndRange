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
  float3 *d_points;
  float3 *d_queries;
  size_t numQueries;
  cukd::box_t<float3> *d_bounds;
  float cutOffRadius;
  int bs,nb;
  size_t tree_size;
  Neig  *results_k;
  int  *results_1;
    cudaEvent_t start, stop;

  MyFastKDTree() {
    bs = 128;
    cutOffRadius = std::numeric_limits<float>::infinity();
    //cudaMallocManaged((char**)&d_points, MAX_INPUT_PC_SIZE*sizeof(*d_points));
    cudaMallocManaged((void**)&d_bounds, sizeof(cukd::box_t<float3>));
    cudaMallocManaged((char**)&d_queries, MAX_QUERY_SIZE*sizeof(*d_queries));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  void build(float *indatav, size_t size) {
    cudaEventRecord(start);

    cudaMallocManaged((char**)&d_points, size*sizeof(*d_points));

    tree_size = size / 3; 

    for (int i=0, j=0;i<tree_size;i++) {
      d_points[i].x = indatav[j++];
      d_points[i].y = indatav[j++];
      d_points[i].z = indatav[j++];
    }

    cukd::buildTree(d_points, tree_size, d_bounds);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "build time " << milliseconds << " ms" << std::endl;
    
  }

  void query(float *queries, size_t numQueries, int* d_results) {
    
    numQueries /= 3;
    nb = divRoundUp((int)numQueries,bs);
    cudaMallocManaged((void**)&results_k, k*numQueries*sizeof(results_k));

    for (int i=0, j=0;i<numQueries;i++) {
      d_queries[i].x = queries[j++];
      d_queries[i].y = queries[j++];
      d_queries[i].z = queries[j++];
    }
    //CUKD_CUDA_SYNC_CHECK();

    d_knn<FixedCandidateList<k>><<<nb,bs>>>
          (results_k,
           d_queries, numQueries,
           d_bounds,
           d_points, tree_size, cutOffRadius);
    

    cudaMemcpy(d_results, results_k, k * numQueries * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


  }

  void query_1(float *queries, size_t numQueries, int* d_results) {
    
    numQueries /= 3;
    nb = divRoundUp((int)numQueries,bs);
    cudaMallocManaged((void**)&results_1, numQueries*sizeof(results_1));
    
    for (int i=0, j=0;i<numQueries;i++) {
      d_queries[i].x = queries[j++];
      d_queries[i].y = queries[j++];
      d_queries[i].z = queries[j++];
    }

    d_knn_1<FixedCandidateList<1>><<<nb,bs>>>
          (results_1,
           d_queries, numQueries,
           d_bounds,
           d_points, tree_size, cutOffRadius);
    

    cudaDeviceSynchronize();

    cudaMemcpy(d_results, results_1, numQueries * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

  }

  void emptymemory() {
    cudaFree(d_points);
    cudaFree(results_k);
    cudaFree(results_1);
  }
  
};
  

extern "C" {

  MyFastKDTree* MyFastKDTree_new(){ return new MyFastKDTree(); }
  void MyFastKDTree_build  (MyFastKDTree* foo, float *indatav, size_t size){ foo->build(indatav, size); }
  void MyFastKDTree_query  (MyFastKDTree* foo, float *queries, size_t numQueries, int* d_results){ foo->query(queries, numQueries, d_results); }
  void MyFastKDTree_query_1(MyFastKDTree* foo, float *queries, size_t numQueries, int* d_results){ foo->query_1(queries, numQueries, d_results); }
  void MyFastKDTree_free(MyFastKDTree* foo){ foo->emptymemory(); }
}