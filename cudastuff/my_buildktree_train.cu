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

#define k 17


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


class MyFastKDTree {

public:

  MyFastKDTree() {}

  void run(float *indatav,    size_t tree_size, int*  d_results_k ) {
    

    float3 *d_points;
    float3 *d_queries_k;
    cukd::box_t<float3> *d_bounds;
    float numQueries_k;
    float cutOffRadius;
    int bs,nb;
    Neig  *results_k;

    tree_size    /= 3;
    numQueries_k  = tree_size;
    bs = 128;
    cutOffRadius = std::numeric_limits<float>::infinity();

    cudaMallocManaged((char**)&d_points, tree_size*sizeof(*d_points));
    cudaMallocManaged((void**)&d_bounds, sizeof(cukd::box_t<float3>));

    cudaMallocManaged((char**)&d_queries_k, numQueries_k*sizeof(*d_queries_k));
    cudaMallocManaged((void**)&results_k, k*numQueries_k*sizeof(results_k));
    

    //////////////    BUILD TREE    ////////////////

    for (int i=0, j=0;i<tree_size;i++) {
      d_points[i].x = indatav[j];
      d_queries_k[i].x = indatav[j++];
      d_points[i].y = indatav[j];
      d_queries_k[i].y = indatav[j++];
      d_points[i].z = indatav[j];
      d_queries_k[i].z = indatav[j++];
    }
    cukd::buildTree(d_points, tree_size, d_bounds);
    cudaDeviceSynchronize();

    //////////////    QUERY K    ////////////////

    nb = divRoundUp((int)numQueries_k,bs);

    d_knn<FixedCandidateList<k>><<<nb,bs>>>
          (results_k,
           d_queries_k, numQueries_k,
           d_bounds,
           d_points, tree_size, cutOffRadius);
    
    cudaDeviceSynchronize();

    cudaMemcpy(d_results_k, results_k, k * numQueries_k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_points);
    cudaFree(results_k);
    
  }

  
};
  

extern "C" {

  MyFastKDTree* MyFastKDTree_new(){ return new MyFastKDTree(); }
  void MyFastKDTree_run(MyFastKDTree* foo, float *indatav, size_t tree_size, int* d_results_k ){
                          foo->run(indatav, tree_size, d_results_k);
                        }
}