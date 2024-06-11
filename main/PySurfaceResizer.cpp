
#include "PyNvCodec.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PySurfaceResizer::PySurfaceResizer(uint32_t width, uint32_t height,
                                   Pixel_Format format, CUcontext ctx,
                                   CUstream str)
    : outputFormat(format)
{
  upResizer.reset(ResizeSurface::Make(width, height, format, ctx, str));
}

PySurfaceResizer::PySurfaceResizer(uint32_t width, uint32_t height,
                                   Pixel_Format format, uint32_t gpuID)
    : outputFormat(format)
{
  upResizer.reset(ResizeSurface::Make(width, height, format,
                                      CudaResMgr::Instance().GetCtx(gpuID),
                                      CudaResMgr::Instance().GetStream(gpuID)));
}

Pixel_Format PySurfaceResizer::GetFormat() { return outputFormat; }

shared_ptr<Surface> PySurfaceResizer::Execute(shared_ptr<Surface> surface)
{
  if (!surface) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  upResizer->SetInput(surface.get(), 0U);

  if (TASK_EXEC_SUCCESS != upResizer->Execute()) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  auto pSurface = (Surface*)upResizer->GetOutput(0U);
  return shared_ptr<Surface>(pSurface ? pSurface->Clone()
                                      : Surface::Make(outputFormat));
}
