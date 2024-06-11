/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Kognia Sports Intelligence
 * Copyright 2021 Videonetics Technology Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PyNvCodec.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

static auto ThrowOnCudaError = [](CUresult res, int lineNum = -1) {
  if (CUDA_SUCCESS != res) {
    stringstream ss;

    if (lineNum > 0) {
      ss << __FILE__ << ":";
      ss << lineNum << endl;
    }

    const char* errName = nullptr;
    if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {
      ss << "CUDA error with code " << res << endl;
    } else {
      ss << "CUDA error: " << errName << endl;
    }

    const char* errDesc = nullptr;
    cuGetErrorString(res, &errDesc);

    if (!errDesc) {
      ss << "No error string available" << endl;
    } else {
      ss << errDesc << endl;
    }

    throw runtime_error(ss.str());
  }
};

CudaResMgr::CudaResMgr()
{
  lock_guard<mutex> lock_ctx(CudaResMgr::gInsMutex);

  ThrowOnCudaError(cuInit(0), __LINE__);

  int nGpu;
  ThrowOnCudaError(cuDeviceGetCount(&nGpu), __LINE__);

  for (int i = 0; i < nGpu; i++) {
    CUdevice cuDevice = 0;
    CUcontext cuContext = nullptr;
    g_Contexts.push_back(make_pair(cuDevice, cuContext));

    CUstream cuStream = nullptr;
    g_Streams.push_back(cuStream);
  }
  return;
}

CUcontext CudaResMgr::GetCtx(size_t idx)
{
  lock_guard<mutex> lock_ctx(CudaResMgr::gCtxMutex);

  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto& ctx = g_Contexts[idx];
  if (!ctx.second) {
    CUdevice cuDevice = 0;
    ThrowOnCudaError(cuDeviceGet(&cuDevice, idx), __LINE__);
    ThrowOnCudaError(cuDevicePrimaryCtxRetain(&ctx.second, cuDevice), __LINE__);
  }

  return g_Contexts[idx].second;
}

CUstream CudaResMgr::GetStream(size_t idx)
{
  lock_guard<mutex> lock_ctx(CudaResMgr::gStrMutex);

  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto& str = g_Streams[idx];
  if (!str) {
    auto ctx = GetCtx(idx);
    CudaCtxPush push(ctx);
    ThrowOnCudaError(cuStreamCreate(&str, CU_STREAM_NON_BLOCKING), __LINE__);
  }

  return g_Streams[idx];
}

CudaResMgr::~CudaResMgr()
{
  lock_guard<mutex> ins_lock(CudaResMgr::gInsMutex);
  lock_guard<mutex> ctx_lock(CudaResMgr::gCtxMutex);
  lock_guard<mutex> str_lock(CudaResMgr::gStrMutex);

  stringstream ss;
  try {
    {
      for (auto& cuStream : g_Streams) {
        if (cuStream) {
          cuStreamDestroy(cuStream); // Avoiding CUDA_ERROR_DEINITIALIZED while
                                     // destructing.
        }
      }
      g_Streams.clear();
    }

    {
      for (int i = 0; i < g_Contexts.size(); i++) {
        if (g_Contexts[i].second) {
          cuDevicePrimaryCtxRelease(
              g_Contexts[i].first); // Avoiding CUDA_ERROR_DEINITIALIZED while
                                    // destructing.
        }
      }
      g_Contexts.clear();
    }
  } catch (runtime_error& e) {
    cerr << e.what() << endl;
  }

#ifdef TRACK_TOKEN_ALLOCATIONS
  cout << "Checking token allocation counters: ";
  auto res = CheckAllocationCounters();
  cout << (res ? "No leaks dectected" : "Leaks detected") << endl;
#endif
}

CudaResMgr& CudaResMgr::Instance()
{
  static CudaResMgr instance;
  return instance;
}

size_t CudaResMgr::GetNumGpus() { return Instance().g_Contexts.size(); }

mutex CudaResMgr::gInsMutex;
mutex CudaResMgr::gCtxMutex;
mutex CudaResMgr::gStrMutex;

auto CopyBuffer_Ctx_Str = [](shared_ptr<CudaBuffer> dst,
                             shared_ptr<CudaBuffer> src, CUcontext cudaCtx,
                             CUstream cudaStream) {
  if (dst->GetRawMemSize() != src->GetRawMemSize()) {
    throw runtime_error("Can't copy: buffers have different size.");
  }

  CudaCtxPush ctxPush(cudaCtx);
  ThrowOnCudaError(cuMemcpyDtoDAsync(dst->GpuMem(), src->GpuMem(),
                                     src->GetRawMemSize(), cudaStream));
  ThrowOnCudaError(cuStreamSynchronize(cudaStream), __LINE__);
};

auto CopyBuffer = [](shared_ptr<CudaBuffer> dst, shared_ptr<CudaBuffer> src,
                     int gpuID) {
  auto ctx = CudaResMgr::Instance().GetCtx(gpuID);
  auto str = CudaResMgr::Instance().GetStream(gpuID);
  return CopyBuffer_Ctx_Str(dst, src, ctx, str);
};
