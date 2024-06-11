
#include "PyNvCodec.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PyNvDecoder::PyNvDecoder(const string& pathToFile, int gpuOrdinal)
    : PyNvDecoder(pathToFile, gpuOrdinal, map<string, string>())
{
}

PyNvDecoder::PyNvDecoder(const string& pathToFile, CUcontext ctx, CUstream str)
    : PyNvDecoder(pathToFile, ctx, str, map<string, string>())
{
}

PyNvDecoder::PyNvDecoder(const string& pathToFile, int gpuOrdinal,
                         const map<string, string>& ffmpeg_options)
{
  if (gpuOrdinal < 0 || gpuOrdinal >= CudaResMgr::Instance().GetNumGpus()) {
    gpuOrdinal = 0U;
  }
  gpuID = gpuOrdinal;

  vector<const char*> options;
  for (auto& pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));

  MuxingParams params;
  upDemuxer->GetParams(params);
  format = params.videoContext.format;

  upDecoder.reset(NvdecDecodeFrame::Make(
      CudaResMgr::Instance().GetStream(gpuID),
      CudaResMgr::Instance().GetCtx(gpuID), params.videoContext.codec,
      poolFrameSize, params.videoContext.width, params.videoContext.height,
      format));
}

PyNvDecoder::PyNvDecoder(const string& pathToFile, CUcontext ctx, CUstream str,
                         const map<string, string>& ffmpeg_options)
{
  vector<const char*> options;
  for (auto& pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));

  MuxingParams params;
  upDemuxer->GetParams(params);
  format = params.videoContext.format;

  upDecoder.reset(NvdecDecodeFrame::Make(
      str, ctx, params.videoContext.codec, poolFrameSize,
      params.videoContext.width, params.videoContext.height, format));
}

PyNvDecoder::PyNvDecoder(uint32_t width, uint32_t height,
                         Pixel_Format new_format, cudaVideoCodec codec,
                         uint32_t gpuOrdinal)
    : format(new_format)
{
  if (gpuOrdinal >= CudaResMgr::Instance().GetNumGpus()) {
    gpuOrdinal = 0U;
  }
  gpuID = gpuOrdinal;

  upDecoder.reset(
      NvdecDecodeFrame::Make(CudaResMgr::Instance().GetStream(gpuID),
                             CudaResMgr::Instance().GetCtx(gpuID), codec,
                             poolFrameSize, width, height, format));
}

PyNvDecoder::PyNvDecoder(uint32_t width, uint32_t height,
                         Pixel_Format new_format, cudaVideoCodec codec,
                         CUcontext ctx, CUstream str)
    : format(new_format)
{
  upDecoder.reset(NvdecDecodeFrame::Make(str, ctx, codec, poolFrameSize, width,
                                         height, format));
}

Buffer* PyNvDecoder::getElementaryVideo(DemuxFrame* demuxer,
                                        SeekContext* seek_ctx, bool needSEI)
{
  Buffer* elementaryVideo = nullptr;
  Buffer* pktData = nullptr;
  shared_ptr<Buffer> pSeekCtxBuf = nullptr;

  do {
    // Set 1st demuxer input to any non-zero value if we need SEI;
    if (needSEI) {
      demuxer->SetInput((Token*)0xdeadbeefull, 0U);
    }

    // Set 2nd demuxer input to seek context if we need to seek;
    if (seek_ctx && seek_ctx->use_seek) {
      pSeekCtxBuf =
          shared_ptr<Buffer>(Buffer::MakeOwnMem(sizeof(SeekContext), seek_ctx));
      demuxer->SetInput((Token*)pSeekCtxBuf.get(), 1U);
    }
    if (TASK_EXEC_FAIL == demuxer->Execute()) {
      return nullptr;
    }
    elementaryVideo = (Buffer*)demuxer->GetOutput(0U);

    /* Clear inputs and set down seek flag or we will seek
     * for one and the same frame multiple times. */
    if (seek_ctx) {
      seek_ctx->use_seek = false;
    }
    demuxer->ClearInputs();
  } while (!elementaryVideo);

  auto pktDataBuf = (Buffer*)demuxer->GetOutput(3U);
  if (pktDataBuf) {
    auto pPktData = pktDataBuf->GetDataAs<PacketData>();
    if (seek_ctx) {
      seek_ctx->out_frame_pts = pPktData->pts;
      seek_ctx->out_frame_duration = pPktData->duration;
    }
  }

  return elementaryVideo;
};

Surface* PyNvDecoder::getDecodedSurface(NvdecDecodeFrame* decoder,
                                        DemuxFrame* demuxer,
                                        SeekContext* seek_ctx, bool needSEI)
{
  decoder->ClearInputs();
  decoder->ClearOutputs();

  Surface* surface = nullptr;
  do {
    auto elementaryVideo = getElementaryVideo(demuxer, seek_ctx, needSEI);
    auto pktData = (Buffer*)demuxer->GetOutput(3U);

    decoder->SetInput(elementaryVideo, 0U);
    decoder->SetInput(pktData, 1U);
    if (TASK_EXEC_FAIL == decoder->Execute()) {
      break;
    }

    surface = (Surface*)decoder->GetOutput(0U);
  } while (!surface);

  return surface;
};

Surface*
PyNvDecoder::getDecodedSurfaceFromPacket(const std::vector<uint8_t>* pPacket,
                                         const PacketData* p_packet_data,
                                         bool no_eos)
{
  upDecoder->ClearInputs();
  upDecoder->ClearOutputs();

  Surface* surface = nullptr;
  unique_ptr<Buffer> packetData = nullptr;
  unique_ptr<Buffer> elementaryVideo = nullptr;

  if (pPacket && pPacket->size()) {
    elementaryVideo = unique_ptr<Buffer>(
        Buffer::MakeOwnMem(pPacket->size(), pPacket->data()));
  }

  if (no_eos) {
    upDecoder->SetInput((Token*)0xbaddf00dull, 2U);
  }

  if (p_packet_data) {
    packetData = unique_ptr<Buffer>(
        Buffer::MakeOwnMem(sizeof(PacketData), p_packet_data));
    upDecoder->SetInput(packetData.get(), 1U);
  }

  upDecoder->SetInput(elementaryVideo ? elementaryVideo.get() : nullptr, 0U);
  if (TASK_EXEC_FAIL == upDecoder->Execute()) {
    return nullptr;
  }

  return (Surface*)upDecoder->GetOutput(0U);
};

uint32_t PyNvDecoder::Width() const
{
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.width;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get width from demuxer instead");
  }
}

void PyNvDecoder::LastPacketData(PacketData& packetData) const
{
  if (upDemuxer) {
    auto mp_buffer = (Buffer*)upDemuxer->GetOutput(3U);
    if (mp_buffer) {
      auto mp = mp_buffer->GetDataAs<PacketData>();
      packetData = *mp;
    }
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get packet data from demuxer instead");
  }
}

ColorSpace PyNvDecoder::GetColorSpace() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.color_space;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get color space from demuxer instead");
  }
}

ColorRange PyNvDecoder::GetColorRange() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.color_range;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get color range from demuxer instead");
  }
}

uint32_t PyNvDecoder::Height() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.height;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get height from demuxer instead");
  }
}

double PyNvDecoder::Framerate() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.frameRate;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get framerate from demuxer instead");
  }
}

double PyNvDecoder::AvgFramerate() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.avgFrameRate;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get avg framerate from demuxer instead");
  }
}

bool PyNvDecoder::IsVFR() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.is_vfr;
  } else {
    throw runtime_error(
        "Decoder was created without built-in demuxer support. "
        "Please check variable framerate flag from demuxer instead");
  }
}

double PyNvDecoder::Timebase() const
{
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.timeBase;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get time base from demuxer instead");
  }
}

uint32_t PyNvDecoder::Framesize() const
{
  if (upDemuxer) {
    auto pSurface = Surface::Make(GetPixelFormat(), Width(), Height(),
                                  CudaResMgr::Instance().GetCtx(gpuID));
    if (!pSurface) {
      throw runtime_error("Failed to determine video frame size.");
    }
    uint32_t size = pSurface->HostMemSize();
    delete pSurface;
    return size;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get frame size from demuxer instead");
  }
}

uint32_t PyNvDecoder::Numframes() const
{
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.num_frames;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get num_frames from demuxer instead");
  }
}

Pixel_Format PyNvDecoder::GetPixelFormat() const { return format; }

void PyNvDecoder::UpdateState()
{
  last_h = Height();
  last_w = Width();
}

bool PyNvDecoder::IsResolutionChanged()
{
  try {
    if (last_h != Height()) {
      return true;
    }

    if (last_w != Width()) {
      return true;
    }
  } catch (exception& e) {
    return false;
  }

  return false;
}

bool PyNvDecoder::DecodeSurface(DecodeContext& ctx)
{

  if (!upDemuxer && !ctx.IsStandalone() && !ctx.IsFlush()) {
    throw std::runtime_error(
        "Tried to call DecodeSurface/DecodeFrame on a Decoder that has been "
        "initialized "
        "without a built-in demuxer. Please use "
        "DecodeSurfaceFromPacket/DecodeFrameFromPacket instead or "
        "intialize the decoder with a demuxer when decoding from a file");
  }
  try {
    UpdateState();
  } catch (exception& e) {
    // Prevent exception throw;
  }

  bool loop_end = false;
  // If we feed decoder with Annex.B from outside we can't seek;
  bool const use_seek = ctx.IsSeek();
  bool dec_error = false, dmx_error = false;

  Surface* pRawSurf = nullptr;

  // Check seek params & flush decoder if we need to seek;
  if (use_seek) {
    MuxingParams params;
    upDemuxer->GetParams(params);

    if (PREV_KEY_FRAME != ctx.GetSeekContext()->mode) {
      throw runtime_error(
          "Decoder can only seek to closest previous key frame");
    }

    // Flush decoder without setting eos flag;
    Surface* p_surf = nullptr;
    do {
      try {
        p_surf = getDecodedSurfaceFromPacket(nullptr, nullptr);
      } catch (decoder_error& dec_exc) {
        dec_error = true;
        cerr << dec_exc.what() << endl;
      } catch (cuvid_parser_error& cvd_exc) {
        dmx_error = true;
        cerr << cvd_exc.what() << endl;
      }
    } while (p_surf && !p_surf->Empty());
    upDecoder->ClearOutputs();

    // Set number of decoded frames to zero before the loop;
    ctx.GetSeekContextMutable()->num_frames_decoded = 0U;
  }

  /* Decode frames in loop if seek was done.
   * Otherwise will return after 1st iteration. */
  do {
    try {
      if (ctx.IsFlush()) {
        pRawSurf = getDecodedSurfaceFromPacket(nullptr, nullptr);
      } else if (ctx.IsStandalone()) {
        pRawSurf =
            getDecodedSurfaceFromPacket(ctx.GetPacket(), ctx.GetInPacketData());
      } else {
        pRawSurf = getDecodedSurface(upDecoder.get(), upDemuxer.get(),
                                     ctx.GetSeekContextMutable(), ctx.HasSEI());
      }

      if (!pRawSurf) {
        break;
      }
    } catch (decoder_error& dec_exc) {
      dec_error = true;
      cerr << dec_exc.what() << endl;
    } catch (cuvid_parser_error& cvd_exc) {
      dmx_error = true;
      cerr << cvd_exc.what() << endl;
    }

    // Increase the counter;
    if (use_seek)
      ctx.GetSeekContextMutable()->num_frames_decoded++;

    /* Get timestamp from decoder.
     * However, this doesn't contain anything beside pts. */
    auto pktDataBuf = (Buffer*)upDecoder->GetOutput(1U);
    if (pktDataBuf && ctx.HasOutPktData()) {
      ctx.SetOutPacketData(pktDataBuf->GetDataAs<PacketData>());
    }

    auto is_seek_done = [&](DecodeContext const& ctx, int64_t pts) {
      auto seek_ctx = ctx.GetSeekContext();
      if (!seek_ctx)
        throw runtime_error("No seek context.");

      int64_t seek_pts = 0;

      if (seek_ctx->IsByNumber()) {
        seek_pts = upDemuxer->TsFromFrameNumber(seek_ctx->seek_frame);
      } else if (seek_ctx->IsByTimestamp()) {
        seek_pts = upDemuxer->TsFromTime(seek_ctx->seek_tssec);
      } else {
        throw runtime_error("Invalid seek mode.");
      }

      return (pts >= seek_pts);
    };

    /* Check if seek is done. */
    if (!use_seek) {
      loop_end = true;
    } else if (pktDataBuf) {
      auto out_pkt_data = pktDataBuf->GetDataAs<PacketData>();
      if (AV_NOPTS_VALUE == out_pkt_data->pts) {
        throw runtime_error("Decoded frame doesn't have PTS, can't seek.");
      }
      loop_end = is_seek_done(ctx, out_pkt_data->pts);
    }

    if (dmx_error) {
      cerr << "Cuvid parser exception happened." << endl;
      throw CuvidParserException();
    }

    if (dec_error && upDemuxer) {
      time_point<system_clock> then = system_clock::now();

      MuxingParams params;
      upDemuxer->GetParams(params);

      upDecoder.reset(NvdecDecodeFrame::Make(
          CudaResMgr::Instance().GetStream(gpuID),
          CudaResMgr::Instance().GetCtx(gpuID), params.videoContext.codec,
          poolFrameSize, params.videoContext.width, params.videoContext.height,
          format));

      time_point<system_clock> now = system_clock::now();
      auto duration = duration_cast<milliseconds>(now - then).count();
      cerr << "HW decoder reset time: " << duration << " milliseconds" << endl;

      throw HwResetException();
    } else if (dec_error) {
      cerr << "HW exception happened. Please reset class instance" << endl;
      throw HwResetException();
    }

    if (ctx.HasSEI()) {
      auto seiBuffer = (Buffer*)upDemuxer->GetOutput(2U);
      ctx.SetSei(seiBuffer);
    }

  } while (use_seek && !loop_end);

  if (pRawSurf) {
    ctx.SetCloneSurface(pRawSurf);
    return true;
  } else {
    return false;
  }
}

auto make_empty_surface = [](Pixel_Format pixFmt) {
  auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
  return shared_ptr<Surface>(pSurface->Clone());
};

void PyNvDecoder::DownloaderLazyInit()
{
  if (IsResolutionChanged() && upDownloader) {
    upDownloader.reset();
    upDownloader = nullptr;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }
}

bool PyNvDecoder::DecodeFrame(class DecodeContext& ctx,
                              std::vector<uint8_t>& frame)
{
  if (!DecodeSurface(ctx))
    return false;

  DownloaderLazyInit();
  return upDownloader->DownloadSingleSurface(ctx.GetSurfaceMutable(), frame);
}

std::map<NV_DEC_CAPS, int> PyNvDecoder::Capabilities() const
{
  std::map<NV_DEC_CAPS, int> capabilities;
  capabilities.erase(capabilities.begin(), capabilities.end());

  for (int cap = BIT_DEPTH_MINUS_8; cap < NV_DEC_CAPS_NUM_ENTRIES; cap++) {
    capabilities[(NV_DEC_CAPS)cap] = upDecoder->GetCapability((NV_DEC_CAPS)cap);
  }

  return capabilities;
}
