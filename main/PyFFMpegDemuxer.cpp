

#include "PyNvCodec.hpp"
#include <streambuf>

using namespace std;
using namespace VPF;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PyFFmpegDemuxer::PyFFmpegDemuxer(const string& pathToFile)
    : PyFFmpegDemuxer(pathToFile, map<string, string>())
{
}

PyFFmpegDemuxer::PyFFmpegDemuxer(const string& pathToFile,
                                 const map<string, string>& ffmpeg_options)
{
  vector<const char*> options;
  for (auto& pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));
}

bool PyFFmpegDemuxer::DemuxSinglePacket(std::vector<uint8_t>& packet,
                                        std::vector<uint8_t>* sei)
{
  // SEI（补充增强信息）
  upDemuxer->ClearInputs();
  upDemuxer->ClearOutputs();

  Buffer* elementaryVideo = nullptr;
  do {
    if (nullptr != sei) {
      upDemuxer->SetInput((Token*)0xdeadbeefull, 0U);
    }

    // execute调用run 执行核心解复用
    //最后调用的ffmpeg函数进行解复用
    if (TASK_EXEC_FAIL == upDemuxer->Execute()) {
      upDemuxer->ClearInputs();
      return false;
    }
    elementaryVideo = (Buffer*)upDemuxer->GetOutput(0U);
  } while (!elementaryVideo);

  packet.resize({elementaryVideo->GetRawMemSize()}, false);
  memcpy(packet.data(), elementaryVideo->GetDataAs<void>(),
         elementaryVideo->GetRawMemSize());

  auto seiBuffer = (Buffer*)upDemuxer->GetOutput(2U);
  if (seiBuffer && sei) {
    sei->resize({seiBuffer->GetRawMemSize()}, false);
    memcpy(sei->data(), seiBuffer->GetDataAs<void>(),
           seiBuffer->GetRawMemSize());
  }

  upDemuxer->ClearInputs();
  return true;
}

void PyFFmpegDemuxer::GetLastPacketData(PacketData& pkt_data)
{
  auto pkt_data_buf = (Buffer*)upDemuxer->GetOutput(3U);
  if (pkt_data_buf) {
    auto pkt_data_ptr = pkt_data_buf->GetDataAs<PacketData>();
    pkt_data = *pkt_data_ptr;
  }
}

uint32_t PyFFmpegDemuxer::Width() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.width;
}

ColorSpace PyFFmpegDemuxer::GetColorSpace() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.color_space;
};

ColorRange PyFFmpegDemuxer::GetColorRange() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.color_range;
};

uint32_t PyFFmpegDemuxer::Height() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.height;
}

Pixel_Format PyFFmpegDemuxer::Format() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.format;
}

cudaVideoCodec PyFFmpegDemuxer::Codec() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.codec;
}

double PyFFmpegDemuxer::Framerate() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.frameRate;
}

double PyFFmpegDemuxer::AvgFramerate() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.avgFrameRate;
}

bool PyFFmpegDemuxer::IsVFR() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.is_vfr;
}

double PyFFmpegDemuxer::Timebase() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.timeBase;
}

uint32_t PyFFmpegDemuxer::Numframes() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.num_frames;
}

bool PyFFmpegDemuxer::Seek(SeekContext& ctx, std::vector<uint8_t>& packet)
{
  Buffer* elementaryVideo = nullptr;
  auto pSeekCtxBuf = shared_ptr<Buffer>(Buffer::MakeOwnMem(sizeof(ctx), &ctx));
  do {
    upDemuxer->SetInput((Token*)pSeekCtxBuf.get(), 1U);
    if (TASK_EXEC_FAIL == upDemuxer->Execute()) {
      upDemuxer->ClearInputs();
      return false;
    }
    elementaryVideo = (Buffer*)upDemuxer->GetOutput(0U);
  } while (!elementaryVideo);

  packet.resize({elementaryVideo->GetRawMemSize()}, false);
  memcpy(packet.data(), elementaryVideo->GetDataAs<void>(),
         elementaryVideo->GetRawMemSize());

  auto pktDataBuf = (Buffer*)upDemuxer->GetOutput(3U);
  if (pktDataBuf) {
    auto pPktData = pktDataBuf->GetDataAs<PacketData>();
    ctx.out_frame_pts = pPktData->pts;
    ctx.out_frame_duration = pPktData->duration;
  }

  upDemuxer->ClearInputs();
  return true;
}
