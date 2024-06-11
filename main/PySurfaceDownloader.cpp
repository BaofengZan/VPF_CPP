

#include "PyNvCodec.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PySurfaceDownloader::PySurfaceDownloader(uint32_t width, uint32_t height,
                                         Pixel_Format format, uint32_t gpu_ID)
{
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  upDownloader.reset(
      CudaDownloadSurface::Make(CudaResMgr::Instance().GetStream(gpu_ID),
                                CudaResMgr::Instance().GetCtx(gpu_ID),
                                surfaceWidth, surfaceHeight, surfaceFormat));
}

PySurfaceDownloader::PySurfaceDownloader(uint32_t width, uint32_t height,
                                         Pixel_Format format, CUcontext ctx,
                                         CUstream str)
{
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  upDownloader.reset(CudaDownloadSurface::Make(str, ctx, surfaceWidth,
                                               surfaceHeight, surfaceFormat));
}

Pixel_Format PySurfaceDownloader::GetFormat() { return surfaceFormat; }

bool PySurfaceDownloader::DownloadSingleSurface(shared_ptr<Surface> surface,
                                                std::vector<uint8_t>& frame)
{
  upDownloader->SetInput(surface.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  auto* pRawFrame = (Buffer*)upDownloader->GetOutput(0U);
  if (pRawFrame) {
    auto const downloadSize = pRawFrame->GetRawMemSize();
    if (downloadSize != frame.size()) {
      frame.resize({downloadSize}, false);
    }

    memcpy(frame.data(), pRawFrame->GetRawMemPtr(), downloadSize);
    return true;
  }

  return false;
}

bool PySurfaceDownloader::DownloadSingleSurface(shared_ptr<Surface> surface,
                                                std::vector<float>& frame)
{
  upDownloader->SetInput(surface.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  auto* pRawFrame = (Buffer*)upDownloader->GetOutput(0U);
  if (pRawFrame) {
    auto const downloadSize = pRawFrame->GetRawMemSize();
    if (downloadSize != frame.size() * sizeof(float)) {
      frame.resize({downloadSize}, false);
    }
    memcpy(frame.data(), pRawFrame->GetRawMemPtr(), downloadSize);
    return true;
  }

  return false;
}

bool PySurfaceDownloader::DownloadSingleSurface(shared_ptr<Surface> surface,
                                                std::vector<uint16_t>& frame)
{
  upDownloader->SetInput(surface.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  auto* pRawFrame = (Buffer*)upDownloader->GetOutput(0U);
  if (pRawFrame) {
    auto const downloadSize = pRawFrame->GetRawMemSize();
    if (downloadSize != frame.size() * sizeof(uint16_t)) {
      frame.resize({downloadSize / sizeof(uint16_t)}, false);
    }
    memcpy(frame.data(), pRawFrame->GetRawMemPtr(), downloadSize);
    return true;
  }

  return false;
}
