#pragma once
/*
所有的头文件
*/

#include "FFmpegDemuxer.h"
#include "MemoryInterfaces.hpp"
#include "NvCodecCLIOptions.h"
#include "NvDecoder.h"
#include "TC_CORE.hpp"
#include "Tasks.hpp"

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include <libswscale/swscale.h>
}

using namespace VPF;

extern int nvcvImagePitch; // global variable to hold pitch value

struct MotionVector {
  int source;
  int w, h;
  int src_x, src_y;
  int dst_x, dst_y;
  int motion_x, motion_y;
  int motion_scale;
};

class HwResetException : public std::runtime_error
{
public:
  HwResetException(std::string& str) : std::runtime_error(str) {}
  HwResetException() : std::runtime_error("HW reset") {}
};

class CuvidParserException : public std::runtime_error
{
public:
  CuvidParserException(std::string& str) : std::runtime_error(str) {}
  CuvidParserException() : std::runtime_error("HW reset") {}
};

class CudaResMgr
{
private:
  CudaResMgr();

public:
  CUcontext GetCtx(size_t idx);
  CUstream GetStream(size_t idx);
  ~CudaResMgr();
  static CudaResMgr& Instance();
  static size_t GetNumGpus();

  std::vector<std::pair<CUdevice, CUcontext>> g_Contexts;
  std::vector<CUstream> g_Streams;

  static std::mutex gInsMutex;
  static std::mutex gCtxMutex;
  static std::mutex gStrMutex;
};

class PyFrameUploader
{
  std::unique_ptr<CudaUploadFrame> uploader;
  uint32_t surfaceWidth, surfaceHeight;
  Pixel_Format surfaceFormat;

public:
  PyFrameUploader(uint32_t width, uint32_t height, Pixel_Format format,
                  uint32_t gpu_ID);

  PyFrameUploader(uint32_t width, uint32_t height, Pixel_Format format,
                  CUcontext ctx, CUstream str);

  PyFrameUploader(uint32_t width, uint32_t height, Pixel_Format format,
                  size_t ctx, size_t str)
      : PyFrameUploader(width, height, format, (CUcontext)ctx, (CUstream)str)
  {
  }

  Pixel_Format GetFormat();

  std::shared_ptr<Surface> UploadSingleFrame(std::vector<uint8_t>& frame);

  std::shared_ptr<Surface> UploadSingleFrame(std::vector<float>& frame);
};

class PyBufferUploader
{
  std::unique_ptr<UploadBuffer> uploader;
  uint32_t elem_size, num_elems;

public:
  PyBufferUploader(uint32_t elemSize, uint32_t numElems, uint32_t gpu_ID);

  PyBufferUploader(uint32_t elemSize, uint32_t numElems, CUcontext ctx,
                   CUstream str);

  PyBufferUploader(uint32_t elemSize, uint32_t numElems, size_t ctx, size_t str)
      : PyBufferUploader(elemSize, numElems, (CUcontext)ctx, (CUstream)str)
  {
  }

  std::shared_ptr<CudaBuffer> UploadSingleBuffer(std::vector<uint8_t>& buffer);
};

class PySurfaceDownloader
{
  std::unique_ptr<CudaDownloadSurface> upDownloader;
  uint32_t surfaceWidth, surfaceHeight;
  Pixel_Format surfaceFormat;

public:
  PySurfaceDownloader(uint32_t width, uint32_t height, Pixel_Format format,
                      uint32_t gpu_ID);

  PySurfaceDownloader(uint32_t width, uint32_t height, Pixel_Format format,
                      CUcontext ctx, CUstream str);

  PySurfaceDownloader(uint32_t width, uint32_t height, Pixel_Format format,
                      size_t ctx, size_t str)
      : PySurfaceDownloader(width, height, format, (CUcontext)ctx,
                            (CUstream)str)
  {
  }

  Pixel_Format GetFormat();

  bool DownloadSingleSurface(std::shared_ptr<Surface> surface,
                             std::vector<uint8_t>& frame);
  bool DownloadSingleSurface(std::shared_ptr<Surface> surface,
                             std::vector<float>& frame);
  bool DownloadSingleSurface(std::shared_ptr<Surface> surface,
                             std::vector<uint16_t>& frame);
};

class PyCudaBufferDownloader
{
  std::unique_ptr<DownloadCudaBuffer> upDownloader;
  uint32_t elem_size, num_elems;

public:
  PyCudaBufferDownloader(uint32_t elemSize, uint32_t numElems, uint32_t gpu_ID);

  PyCudaBufferDownloader(uint32_t elemSize, uint32_t numElems, CUcontext ctx,
                         CUstream str);

  PyCudaBufferDownloader(uint32_t elemSize, uint32_t numElems, size_t ctx,
                         size_t str)
      : PyCudaBufferDownloader(elemSize, numElems, (CUcontext)ctx,
                               (CUstream)str)
  {
  }

  bool DownloadSingleCudaBuffer(std::shared_ptr<CudaBuffer> buffer,
                                std::vector<uint8_t>& np_array);
};

class PySurfaceConverter
{
  std::unique_ptr<ConvertSurface> upConverter;
  std::unique_ptr<Buffer> upCtxBuffer;
  Pixel_Format outputFormat;

public:
  PySurfaceConverter(uint32_t width, uint32_t height, Pixel_Format inFormat,
                     Pixel_Format outFormat, uint32_t gpuID);

  PySurfaceConverter(uint32_t width, uint32_t height, Pixel_Format inFormat,
                     Pixel_Format outFormat, CUcontext ctx, CUstream str);

  PySurfaceConverter(uint32_t width, uint32_t height, Pixel_Format inFormat,
                     Pixel_Format outFormat, size_t ctx, size_t str)
      : PySurfaceConverter(width, height, inFormat, outFormat, (CUcontext)ctx,
                           (CUstream)str)
  {
  }

  std::shared_ptr<Surface>
  Execute(std::shared_ptr<Surface> surface,
          std::shared_ptr<ColorspaceConversionContext> context);

  Pixel_Format GetFormat();
};

class PySurfaceResizer
{
  std::unique_ptr<ResizeSurface> upResizer;
  Pixel_Format outputFormat;

public:
  PySurfaceResizer(uint32_t width, uint32_t height, Pixel_Format format,
                   uint32_t gpuID);

  PySurfaceResizer(uint32_t width, uint32_t height, Pixel_Format format,
                   CUcontext ctx, CUstream str);

  PySurfaceResizer(uint32_t width, uint32_t height, Pixel_Format format,
                   size_t ctx, size_t str)
      : PySurfaceResizer(width, height, format, (CUcontext)ctx, (CUstream)str)
  {
  }

  Pixel_Format GetFormat();

  std::shared_ptr<Surface> Execute(std::shared_ptr<Surface> surface);
};

class PySurfaceRemaper
{
  std::unique_ptr<RemapSurface> upRemaper;
  Pixel_Format outputFormat;

public:
  PySurfaceRemaper(std::vector<float>& x_map, std::vector<float>& y_map,
                   Pixel_Format format, size_t ctx, size_t str);

  PySurfaceRemaper(std::vector<float>& x_map, std::vector<float>& y_map,
                   Pixel_Format format, uint32_t gpuID);

  Pixel_Format GetFormat();

  std::shared_ptr<Surface> Execute(std::shared_ptr<Surface> surface);
};

class PyFFmpegDemuxer
{
  // 解复用器
  std::unique_ptr<DemuxFrame> upDemuxer;

public:
  PyFFmpegDemuxer(const std::string& pathToFile);

  PyFFmpegDemuxer(const std::string& pathToFile,
                  const std::map<std::string, std::string>& ffmpeg_options);

  bool DemuxSinglePacket(std::vector<uint8_t>& packet,
                         std::vector<uint8_t>* sei);

  void GetLastPacketData(PacketData& pkt_data);

  bool Seek(SeekContext& ctx, std::vector<uint8_t>& packet);

  uint32_t Width() const;

  uint32_t Height() const;

  Pixel_Format Format() const;

  ColorSpace GetColorSpace() const;

  ColorRange GetColorRange() const;

  cudaVideoCodec Codec() const;

  double Framerate() const;

  double AvgFramerate() const;

  bool IsVFR() const;

  uint32_t Numframes() const;

  double Timebase() const;
};

class PyFfmpegDecoder
{
  std::unique_ptr<FfmpegDecodeFrame> upDecoder = nullptr;
  std::unique_ptr<PyFrameUploader> upUploader = nullptr;

  void* GetSideData(AVFrameSideDataType data_type, size_t& raw_size);

  uint32_t last_w;
  uint32_t last_h;
  uint32_t gpu_id;

  void UpdateState();
  bool IsResolutionChanged();

  void UploaderLazyInit();

public:
  PyFfmpegDecoder(const std::string& pathToFile,
                  const std::map<std::string, std::string>& ffmpeg_options,
                  uint32_t gpuID);

  bool DecodeSingleFrame(std::vector<uint8_t>& frame);
  std::shared_ptr<Surface> DecodeSingleSurface();

  std::vector<MotionVector> GetMotionVectors();

  uint32_t Width() const;
  uint32_t Height() const;
  double Framerate() const;
  ColorSpace Color_Space() const;
  ColorRange Color_Range() const;
  cudaVideoCodec Codec() const;
  Pixel_Format PixelFormat() const;
};

class DecodeContext
{
private:
  std::shared_ptr<Surface> pSurface;

  std::vector<uint8_t>* pSei;
  std::vector<uint8_t>* pPacket;

  PacketData* pInPktData;
  PacketData* pOutPktData;

  SeekContext* pSeekCtx;

  bool flush;

public:
  DecodeContext(std::vector<uint8_t>* sei, std::vector<uint8_t>* packet,
                PacketData* in_pkt_data, PacketData* out_pkt_data,
                SeekContext* seek_ctx, bool is_flush = false)
  {
    if (seek_ctx && packet) {
      throw std::runtime_error("Can't use seek in standalone mode.");
    }

    pSurface = nullptr;
    pSei = sei;
    pPacket = packet;
    pInPktData = in_pkt_data;
    pOutPktData = out_pkt_data;
    pSeekCtx = seek_ctx;
    flush = is_flush;
  }

  bool IsSeek() const { return (nullptr != pSeekCtx) && (nullptr == pPacket); }

  bool IsStandalone() const { return (nullptr != pPacket); }

  bool IsFlush() const { return flush; }

  bool HasSEI() const { return nullptr != pSei; }

  bool HasOutPktData() const { return nullptr != pOutPktData; }

  bool HasInPktData() const { return nullptr != pInPktData; }

  const std::vector<uint8_t>* GetPacket() const { return pPacket; }

  const PacketData* GetInPacketData() const { return pInPktData; }

  const SeekContext* GetSeekContext() const { return pSeekCtx; }

  SeekContext* GetSeekContextMutable() { return pSeekCtx; }

  std::shared_ptr<Surface> GetSurfaceMutable() { return pSurface; }

  void SetOutPacketData(PacketData* out_pkt_data)
  {
    if (!out_pkt_data || !pOutPktData) {
      throw std::runtime_error("Invalid data pointer");
    }

    memcpy(pOutPktData, out_pkt_data, sizeof(PacketData));
  }

  void SetSei(Buffer* sei)
  {
    if (!pSei) {
      throw std::runtime_error("Invalid data pointer");
    }

    if (!sei) {
      pSei->resize({0}, false);
      return;
    }

    pSei->resize({sei->GetRawMemSize()}, false);
    memcpy(pSei->data(), sei->GetRawMemPtr(), sei->GetRawMemSize());
  }

  void SetCloneSurface(Surface* p_surface)
  {
    if (!p_surface) {
      throw std::runtime_error("Invalid data pointer");
    }
    pSurface = std::shared_ptr<Surface>(p_surface->Clone());
  }
};

class PyNvDecoder
{
  std::unique_ptr<DemuxFrame> upDemuxer;
  std::unique_ptr<NvdecDecodeFrame> upDecoder;
  std::unique_ptr<PySurfaceDownloader> upDownloader;
  uint32_t gpuID;
  static uint32_t const poolFrameSize = 4U;
  Pixel_Format format;

  uint32_t last_w;
  uint32_t last_h;

  void UpdateState();
  bool IsResolutionChanged();

public:
  PyNvDecoder(uint32_t width, uint32_t height, Pixel_Format format,
              cudaVideoCodec codec, uint32_t gpuOrdinal);

  PyNvDecoder(const std::string& pathToFile, int gpuOrdinal);

  PyNvDecoder(const std::string& pathToFile, int gpuOrdinal,
              const std::map<std::string, std::string>& ffmpeg_options);

  PyNvDecoder(uint32_t width, uint32_t height, Pixel_Format format,
              cudaVideoCodec codec, CUcontext ctx, CUstream str);

  PyNvDecoder(uint32_t width, uint32_t height, Pixel_Format format,
              cudaVideoCodec codec, size_t ctx, size_t str)
      : PyNvDecoder(width, height, format, codec, (CUcontext)ctx, (CUstream)str)
  {
  }

  PyNvDecoder(const std::string& pathToFile, CUcontext ctx, CUstream str);

  PyNvDecoder(const std::string& pathToFile, size_t ctx, size_t str)
      : PyNvDecoder(pathToFile, (CUcontext)ctx, (CUstream)str)
  {
  }

  PyNvDecoder(const std::string& pathToFile, CUcontext ctx, CUstream str,
              const std::map<std::string, std::string>& ffmpeg_options);

  PyNvDecoder(const std::string& pathToFile, size_t ctx, size_t str,
              const std::map<std::string, std::string>& ffmpeg_options)
      : PyNvDecoder(pathToFile, (CUcontext)ctx, (CUstream)str, ffmpeg_options)
  {
  }

  static Buffer* getElementaryVideo(DemuxFrame* demuxer, SeekContext* seek_ctx,
                                    bool needSEI);

  static Surface* getDecodedSurface(NvdecDecodeFrame* decoder,
                                    DemuxFrame* demuxer, SeekContext* seek_ctx,
                                    bool needSEI);

  uint32_t Width() const;

  ColorSpace GetColorSpace() const;

  ColorRange GetColorRange() const;

  void LastPacketData(PacketData& packetData) const;

  uint32_t Height() const;

  double Framerate() const;

  double AvgFramerate() const;

  bool IsVFR() const;

  uint32_t Numframes() const;

  double Timebase() const;

  uint32_t Framesize() const;

  Pixel_Format GetPixelFormat() const;

  bool DecodeSurface(class DecodeContext& ctx);

  bool DecodeFrame(class DecodeContext& ctx, std::vector<uint8_t>& frame);

  Surface*
  getDecodedSurfaceFromPacket(const std::vector<uint8_t>* pPacket,
                              const PacketData* p_packet_data = nullptr,
                              bool no_eos = false);

  void DownloaderLazyInit();

  std::shared_ptr<Surface> make_empty_surface(Pixel_Format pixFmt)
  {
    auto pSurface = std::shared_ptr<Surface>(Surface::Make(pixFmt));
    return std::shared_ptr<Surface>(pSurface->Clone());
  }

  std::shared_ptr<Surface> DecodeSingleSurface()
  {
    DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, nullptr, false);
    if (DecodeSurface(ctx))
      return ctx.GetSurfaceMutable();
    else
      return make_empty_surface(GetPixelFormat());
  }

  std::map<NV_DEC_CAPS, int> Capabilities() const;
};

class PyNvEncoder
{
  std::unique_ptr<PyFrameUploader> uploader;
  std::unique_ptr<NvencEncodeFrame> upEncoder;
  uint32_t encWidth, encHeight;
  Pixel_Format eFormat;
  std::map<std::string, std::string> options;
  bool verbose_ctor;
  CUcontext cuda_ctx;
  CUstream cuda_str;

public:
  uint32_t Width() const;
  uint32_t Height() const;
  Pixel_Format GetPixelFormat() const;
  std::map<NV_ENC_CAPS, int> Capabilities();
  int GetFrameSizeInBytes() const;
  bool Reconfigure(const std::map<std::string, std::string>& encodeOptions,
                   bool force_idr = false, bool reset_enc = false,
                   bool verbose = false);

  PyNvEncoder(const std::map<std::string, std::string>& encodeOptions,
              int gpuOrdinal, Pixel_Format format = NV12, bool verbose = false);

  PyNvEncoder(const std::map<std::string, std::string>& encodeOptions,
              CUcontext ctx, CUstream str, Pixel_Format format = NV12,
              bool verbose = false);

  PyNvEncoder(const std::map<std::string, std::string>& encodeOptions,
              size_t ctx, size_t str, Pixel_Format format = NV12,
              bool verbose = false)
      : PyNvEncoder(encodeOptions, (CUcontext)ctx, (CUstream)str, format,
                    verbose)
  {
  }

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     std::vector<uint8_t>& packet,
                     const std::vector<uint8_t>& messageSEI, bool sync,
                     bool append);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     std::vector<uint8_t>& packet,
                     const std::vector<uint8_t>& messageSEI, bool sync);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     std::vector<uint8_t>& packet, bool sync);

  // bool EncodeFromNVCVImage(py::object nvcvImage, std::vector<uint8_t>&
  // packet,
  //                         bool bIsNVCVImage);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     std::vector<uint8_t>& packet,
                     const std::vector<uint8_t>& messageSEI);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     std::vector<uint8_t>& packet);

  bool EncodeFrame(std::vector<uint8_t>& inRawFrame,
                   std::vector<uint8_t>& packet);

  bool EncodeFrame(std::vector<uint8_t>& inRawFrame,
                   std::vector<uint8_t>& packet,
                   const std::vector<uint8_t>& messageSEI);

  bool EncodeFrame(std::vector<uint8_t>& inRawFrame,
                   std::vector<uint8_t>& packet, bool sync);

  bool EncodeFrame(std::vector<uint8_t>& inRawFrame,
                   std::vector<uint8_t>& packet,
                   const std::vector<uint8_t>& messageSEI, bool sync);

  bool EncodeFrame(std::vector<uint8_t>& inRawFrame,
                   std::vector<uint8_t>& packet,
                   const std::vector<uint8_t>& messageSEI, bool sync,
                   bool append);

  // Flush all the encoded frames (packets)
  bool Flush(std::vector<uint8_t>& packets);
  // Flush only one encoded frame (packet)
  bool FlushSinglePacket(std::vector<uint8_t>& packet);

  static void CheckValidCUDABuffer(const void* ptr)
  {
    if (ptr == nullptr) {
      throw std::runtime_error("NULL CUDA buffer not accepted");
    }

    cudaPointerAttributes attrs = {};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    cudaGetLastError(); // reset the cuda error (if any)
    if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered) {
      throw std::runtime_error("Buffer is not CUDA-accessible");
    }
  }

private:
  bool EncodeSingleSurface(struct EncodeContext& ctx);
};
