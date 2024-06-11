#pragma once

#include "FFmpegDemuxer.h"
#include "NvDecoder.h"
#include "PyNvCodec.hpp"
#include "Tasks.hpp"
#include "opencv.hpp"
#include "rtsp2flv.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <ostream>
#include <sstream>
#include <thread>
using namespace VPF;
// sampledecoder.py

// 默认dmx_mode=InitMode.STANDALONE

namespace SampleDecode
{
enum DecodeStatus { DEC_ERR, DEC_SUBM, DEC_READY };

class CNvDecoder
{
public:
  int gpu_id;
  std::unique_ptr<PyFFmpegDemuxer> nv_dmx;
  std::unique_ptr<PyNvDecoder> nv_dec;

  int sk_frm = -1;
  int num_frames_decoded = 0;
  std::vector<uint8_t> frame_nv12;
  std::vector<uint8_t> packet;
  PacketData packet_data;

  SeekMode seek_mode = SeekMode::PREV_KEY_FRAME;

  // self.out_file = open(dec_file, "wb")
  std::ofstream outfile; // 创建一个ofstream对象用于写文件

  CNvDecoder(int gpuid, std::string& enc_file, std::string& dec_file);
  ~CNvDecoder() { outfile.close(); }

  void decode(int frames_to_decode = -1, bool verbose = false);
  DecodeStatus decode_frame(bool verbose = false);
  DecodeStatus decode_frame_standalone(bool verbose = false);
  bool flush_frame(std::vector<uint8_t>& frame);
  void dump_frame()
  {
    //// cv::Mat image(464 * 1.5, 848, CV_8U, frame_nv12.data());
    cv::Mat image(480 * 1.5, 854, CV_8U, frame_nv12.data());
    cv::cvtColor(image, image, cv::COLOR_YUV2BGR_NV12);
    cv::imshow("s", image);
    cv::waitKey(1);
    for (auto& item : frame_nv12) {
      outfile << item; // 这里后面不能有空格，否则保存的视频是错误的！！
    }
    // outfile << std::endl;
  }
};

CNvDecoder::CNvDecoder(int gpuid, std::string& enc_file, std::string& dec_file)
{
  gpu_id = gpuid;
  //创建解复用器
  nv_dmx = std::make_unique<PyFFmpegDemuxer>(enc_file);
  //创建解码器
  nv_dec =
      std::make_unique<PyNvDecoder>(nv_dmx->Width(), nv_dmx->Height(),
                                    nv_dmx->Format(), nv_dmx->Codec(), gpuid);
  outfile.open(dec_file, std::ios::out | std::ios::binary);

  // 检查文件是否成功打开
  if (!outfile.is_open()) {
    std::cerr << "无法打开文件: " << dec_file << std::endl;
    return; // 如果文件无法打开，则退出程序
  }
}

void CNvDecoder::decode(int frames_to_decode, bool verbose)
{
  while (true) {
    frame_nv12.clear();
    packet.clear();
    auto status = decode_frame(verbose);
    if (status == DecodeStatus::DEC_ERR) {
      break;
    } else if (status == DecodeStatus::DEC_READY) {
      dump_frame();
    }
  }

  bool need_flush = true;
  while (need_flush) {
    if (!flush_frame(frame_nv12)) {
      break;
    } else {
      dump_frame();
    }
  }
}

DecodeStatus CNvDecoder::decode_frame(bool verbose /*= false*/)
{
  return decode_frame_standalone(verbose);
}

DecodeStatus CNvDecoder::decode_frame_standalone(bool verbose /*= false*/)
{
  auto status = DecodeStatus::DEC_ERR;
  try {
    if (sk_frm >= 0) {
      auto seek_ctx = SeekContext((int64_t)sk_frm, seek_mode);
      sk_frm = -1;
      if (!nv_dmx->Seek(seek_ctx, packet)) {
        return status;
      }
    } else if (!nv_dmx->DemuxSinglePacket(packet, nullptr)) {
      return status;
    }
    auto DecodeFrameFromPacket = [&](std::vector<uint8_t>& frame,
                                     std::vector<uint8_t>& packet) {
      DecodeContext ctx(nullptr, &packet, nullptr, nullptr, nullptr, false);
      return nv_dec->DecodeFrame(ctx, frame);
    };
    bool frame_ready = DecodeFrameFromPacket(frame_nv12, packet);
    // std::cout << "-----" << packet.size() << std::endl;
    if (frame_ready) {
      num_frames_decoded += 1;
      status = DecodeStatus::DEC_READY;
    } else {
      status = DecodeStatus::DEC_SUBM;
    }
    nv_dmx->GetLastPacketData(packet_data);
    if (verbose) {
      std::cout << "===============" << std::endl;
      std::cout << "frame pts (decode order) " << packet_data.pts << std::endl;
      std::cout << "frame dts (decode order)  " << packet_data.dts << std::endl;
      std::cout << "frame pos (decode order) " << packet_data.pos << std::endl;
      std::cout << "frame duration (decode order) " << packet_data.duration
                << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  return status;
}

bool CNvDecoder::flush_frame(std::vector<uint8_t>& frame)
{
  DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, nullptr, true);
  bool ret = nv_dec->DecodeFrame(ctx, frame);
  if (ret) {
    num_frames_decoded += 1;
  }
  return ret;
}

void SampleDecodeTest()
{
  std::string pathToFile = R"(D:\Datasets\video\fall_video.mp4)";
  std::string outFile = R"(D:\Datasets\video\fall_video.yuv)";
  std::shared_ptr<CNvDecoder> dec =
      std::make_shared<CNvDecoder>(0, pathToFile, outFile);
  dec->decode(-1, true);
}

}; // namespace SampleDecode

namespace SampleDecodeMultiThread
{
// 多线程解码
class Worker
{
public:
  Worker(int gpuid, std::string& encfile);
  ~Worker();
  void run();
  std::unique_ptr<PyNvDecoder> nvDec;
  std::unique_ptr<PySurfaceConverter> nvYuv;
  std::unique_ptr<PySurfaceConverter> nvCvt;
  std::unique_ptr<PySurfaceResizer> nvRes;
  std::unique_ptr<PySurfaceDownloader> nvDwn;

  CUcontext context;
  cudaStream_t stream;
  int gpuID{0};
  int width{0};
  int height{0};
  int hwidth{0};
  int hheight{0};
  int num_frame{0};
  ColorSpace cspace;
  ColorRange crange;
  std::shared_ptr<ColorspaceConversionContext> cc_ctx;
};

Worker::Worker(int gpuid, std::string& encfile)
{
  gpuID = gpuid;
  // int deviceCount;
  // cuDeviceGetCount(&deviceCount);
  // CUdevice device;
  // cuDeviceGet(&device, 0); // 获取第一个设备

  // cuCtxCreate(&context, 0, device);
  // cuDevicePrimaryCtxRetain(&context, gpuID); // 保留主上下文
  context = CudaResMgr::Instance().GetCtx(gpuid);
  cudaStreamCreate(&stream); // 每一个线程 创建一个stream

  nvDec = std::make_unique<PyNvDecoder>(encfile, context, stream);

  width = nvDec->Width();
  height = nvDec->Height();
  hwidth = width / 2;
  hheight = height / 2;

  /*
    确定颜色空间转换参数。
    有些视频流没有指定这些参数，所以默认值
    最常用的是bt601和mpeg。
  */
  cspace = nvDec->GetColorSpace();
  crange = nvDec->GetColorRange();

  if (cspace == ColorSpace::UNSPEC) {
    cspace = ColorSpace::BT_709;
  }
  if (crange == ColorRange::UDEF) {
    crange = ColorRange::MPEG;
  }
  cc_ctx = std::make_shared<ColorspaceConversionContext>(cspace, crange);

  if (nvDec->GetColorSpace() != ColorSpace::BT_709) {
    // 颜色空间不是709 先转为yuv420
    // 但是为什么要先转为YUV420呢
    // 如果强制 709也做这个转换：报错Rec.709 YUV -> RGB conversion isn't
    // supported yet.
    // 不支持709的yuv420到rgb 支持的是709下nv12到rgb
    nvYuv = std::make_unique<PySurfaceConverter>(
        width, height, nvDec->GetPixelFormat(), Pixel_Format::YUV420, context,
        stream);
  } else {
    nvYuv = nullptr;
  }

  if (nvYuv != nullptr) {
    // 然后再将YUV420转为rgb
    nvCvt = std::make_unique<PySurfaceConverter>(
        width, height, nvYuv->GetFormat(), Pixel_Format::RGB, context, stream);
  } else {
    // 如果颜色空间是709 就直接转为RGB了
    // NV12转RGB
    nvCvt = std::make_unique<PySurfaceConverter>(
        width, height, nvDec->GetPixelFormat(), Pixel_Format::RGB, context,
        stream);
  }
  nvRes = std::make_unique<PySurfaceResizer>(
      hwidth, hheight, nvCvt->GetFormat(), context, stream);

  nvDwn = std::make_unique<PySurfaceDownloader>(
      hwidth, hheight, nvRes->GetFormat(), context, stream);
}

Worker::~Worker()
{
  cudaStreamDestroy(stream);
  // cuDevicePrimaryCtxRelease(gpuID);
}

void Worker::run()
{
  try {
    std::shared_ptr<Surface> rawSurface;
    std::shared_ptr<Surface> cvtSurface;
    while (true) {
      try {
        rawSurface = nvDec->DecodeSingleSurface();
        if (rawSurface->Empty() || rawSurface == nullptr) { //
          break;
        }
      } catch (HwResetException* e) {
        std::cout << "HwResetException" << std::endl;
        continue;
      }

      if (nvYuv != nullptr) {
        auto yuvSurface = nvYuv->Execute(rawSurface, cc_ctx);
        cvtSurface = nvCvt->Execute(yuvSurface, cc_ctx);
      } else {
        cvtSurface = nvCvt->Execute(rawSurface, cc_ctx);
      }
      if (cvtSurface->Empty()) {
        std::cout << "Failed to do color conversion" << std::endl;
        break;
      }

      auto resSurface = nvRes->Execute(cvtSurface);
      if (resSurface->Empty()) {
        std::cout << "Failed to resize surface" << std::endl;
        break;
      }
      std::vector<uint8_t> rawFrame;
      rawFrame.resize(resSurface->HostMemSize());

      bool success = nvDwn->DownloadSingleSurface(resSurface, rawFrame);
      if (!success) {
        std::cout << "Failed to download surface" << std::endl;
        break;
      }

      //
      num_frame++;
      std::cout << "Thread  " << std::this_thread::get_id() << "  at frame "
                << num_frame << std::endl;
      cv::Mat image(hheight, hwidth, CV_8UC3, rawFrame.data());
      cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
      std::stringstream sin;
      sin << std::this_thread::get_id(); // 获取线程id
      cv::imshow(sin.str(), image);
      cv::waitKey(1000);
      // if (num_frame % int(nvDec->Framerate()) == 0) {
      //  std::cout << "Thread " << std::this_thread::get_id() << "at frame "
      //            << num_frame << std::endl;
      //}
    }

  } catch (...) {
    std::cout << "error..............." << std::endl;
  }
}

void create_threads(int gpu_id, std::string input_file, int num_threads)
{
  // CudaResMgr::Instance().GetStream(gpu_id),
  // CudaResMgr::Instance().GetCtx(gpu_id)

  std::vector<std::thread> thread_pool;
  Worker w(gpu_id, input_file);
  thread_pool.emplace_back(std::thread(&Worker::run, &w));
  Worker w1(gpu_id, input_file);
  thread_pool.emplace_back(std::thread(&Worker::run, &w1));

  // for (int i = 0; i < num_threads; ++i) {

  //  std::thread t(&Worker::run, &w);
  //  thread_pool.emplace_back(std::move(t));
  //}
  for (auto& item : thread_pool) {
    item.join();
  }
}

}; // namespace SampleDecodeMultiThread

// 测试编码
namespace SampleEncode
{

class PushRTSP
{
public:
  AVFormatContext* ofmt_ctx = nullptr;
  const AVCodec* out_codec = nullptr;
  AVStream* out_stream = nullptr;
  AVCodecContext* out_codec_ctx = nullptr;
  SwsContext* swsctx;
  AVFrame* frame;
  int w;
  int h;
  PushRTSP(const std::string url, int to_width, int to_height)
  {
    //
    avformat_network_init();
    w = to_width;
    h = to_height;
    initialize_avformat_context(ofmt_ctx, "flv");
    initialize_io_context(ofmt_ctx, url.c_str());

    out_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    out_stream = avformat_new_stream(ofmt_ctx, out_codec);
    out_codec_ctx = avcodec_alloc_context3(out_codec);

    set_codec_params(ofmt_ctx, out_codec_ctx, to_width, to_height, 30, 900000);
    initialize_codec_stream(
        out_stream, out_codec_ctx, out_codec,
        "high444"); ////(baseline | high | high10 | high422 |
                    /// high444 | main) (default: high444)"

    out_stream->codecpar->extradata = out_codec_ctx->extradata;
    out_stream->codecpar->extradata_size = out_codec_ctx->extradata_size;

    av_dump_format(ofmt_ctx, 0, url.c_str(), 1);

    swsctx = initialize_sample_scaler(out_codec_ctx, to_width, to_height);
    frame = allocate_frame_buffer(out_codec_ctx, to_width, to_height);

    int cur_size;
    uint8_t* cur_ptr;

    int ret = avformat_write_header(ofmt_ctx, nullptr);
    if (ret < 0) {
      return;
    }
  }

  ~PushRTSP()
  {
    //
    // av_frame_free(&frame);
    // avcodec_close(out_codec_ctx);
    // avio_close(ofmt_ctx->pb);
    // avformat_free_context(ofmt_ctx);
  }
  void push(std::vector<uint8_t>& encFrame, int id)
  {
    AVPacket pkt = {0};
    // av_new_packet(&pkt, 0);
    av_init_packet(&pkt);
    pkt.data = encFrame.data();
    pkt.size = encFrame.size();

    pkt.pts = id + 1;
    pkt.dts = id;
    pkt.stream_index = 0;
    av_interleaved_write_frame(ofmt_ctx, &pkt);
    av_packet_unref(&pkt);
    // cv::Mat img(h * 1.5, w, CV_8U, encFrame.data());
    // cv::cvtColor(img, img, cv::COLOR_YUV2BGR_NV12);
    //// memcpy(frame->data, img.data(), encFrame.size());
    // const int stride[] = {static_cast<int>(img.step[0])};
    // sws_scale(swsctx, &img.data, stride, 0, img.rows, frame->data,
    //          frame->linesize);
    // frame->pts +=
    //    av_rescale_q(1, out_codec_ctx->time_base, out_stream->time_base);
    // write_frame(out_codec_ctx, ofmt_ctx, frame);
  }
};

void encode()
{
  // int total_num_frames = 444;
  // gpuID, decFilePath, encFilePath, width, height, codec, format
  int gpuID = 0;
  std::string decFilePath = R"(D:\Datasets\video\fall_video.yuv)";
  std::string encFilePath = R"(D:\Datasets\video\fall_video.h264)";
  int width = 854;
  int height = 480;
  auto pixel_format = Pixel_Format::NV12;
  std::string profile = "high";
  std::string codec = "h264"; // 编码标准

  std::map<std::string, std::string> encoder_config;
  encoder_config["preset"] = "P5";
  encoder_config["tuning_info"] = "high_quality";
  encoder_config["codec"] = codec;
  encoder_config["profile"] = profile;
  encoder_config["s"] = std::to_string(width) + "x" + std::to_string(height);
  encoder_config["bitrate"] = "10M";

  auto nvEnc = PyNvEncoder(encoder_config, gpuID, pixel_format);

  //
  int frameSize =
      nvEnc.GetFrameSizeInBytes(); // 这里返回的实际是个数不是 字节数

  std::vector<uint8_t> encFrame;

  int framesSent = 0;     // 发送给encoder的帧数
  int framesReceived = 0; // 从编码器接收到的帧数

  int framesFlushed = 0; // flush期间收到的帧数

  // while (framesSent < total_num_frames) {
  //  bool success = nvEnc.EncodeFrame(rawFrame, encFrame, sync = False)
  //}
  std::string output_url = "rtmp://192.168.1.16:1935/live";
  PushRTSP rtsp(output_url, width, height);

  std::ifstream in_stream(decFilePath, std::ios::in | std::ios::binary);

  std::ofstream outfile; // 创建一个ofstream对象用于写文件

  outfile.open(encFilePath, std::ios::out | std::ios::binary);

  // 检查文件是否成功打开
  if (!outfile.is_open()) {
    std::cerr << "无法打开文件: " << encFilePath << std::endl;
    return; // 如果文件无法打开，则退出程序
  }

  std::vector<char> char_frame;
  in_stream.seekg(0, std::ios::end);
  std::streamsize size = in_stream.tellg();
  in_stream.seekg(0, std::ios::beg); // 定位回文件开始
  char_frame.resize(size);
  in_stream.read(char_frame.data(), size);
  std::vector<uint8_t> rawFrames(char_frame.begin(), char_frame.end());

  // char_rame.resize(frameSize);
  int total_num_frames = size / frameSize;
  int pp = 0;
  while (framesSent < total_num_frames) {

    /// 判断退出条件(此处可避免多读一次)。
    // if (in_stream.peek() == EOF) { // in.peek()!=EOF
    //  std::cout << "文件末尾" << std::endl;
    //  break;
    //}
    std::vector<uint8_t> rawFrame(rawFrames.begin() + framesSent * frameSize,
                                  rawFrames.begin() +
                                      (framesSent + 1) * frameSize);
    framesSent += 1;
    // if (framesSent >= total_num_frames) {
    //  framesSent = 0;
    //  framesReceived = 0;
    //  continue;
    //}
    encFrame.clear();
    bool success = nvEnc.EncodeFrame(rawFrame, encFrame, false);

    if (success) {
      framesReceived += 1;
      rtsp.push(encFrame, pp);
      pp++;
      for (auto& item : encFrame) {
        outfile << item;
      }
    }
  }

  while (true) {
    //
    encFrame.clear();
    bool success = nvEnc.FlushSinglePacket(encFrame);
    if (success) {
      framesReceived += 1;
      framesFlushed += 1;
      rtsp.push(encFrame, pp);
      pp++;
      for (auto& item : encFrame) {
        outfile << item;
      }
    } else {
      break;
    }
  }

  std::cout << framesFlushed << " frame(s) received during encoder flush."
            << std::endl;

  in_stream.close();
  outfile.close();
}
} // namespace SampleEncode

int main()
{
  // SampleDecode::SampleDecodeTest();
  std::string pathToFile =
      R"(D:\LearningCodes\GithubRepo\shouxieAI\VideoProcessingFramework\tests/test.mp4)";

  // SampleDecodeMultiThread::create_threads(0, pathToFile, 1);

  SampleEncode::encode();
  return 0;
}

/*
`AVPacket` 结构体是 FFmpeg
库中用于存储多媒体数据包（packet）的一个核心数据结构。它主要用来在解码器、编码器、过滤器以及输出媒体文件时传递数据。`AVPacket`
中存储的信息主要包括：

1. **数据缓冲区** (`data`):
一个指向数据的指针，这些数据可以是压缩的视频帧、音频样本或其他多媒体流的一部分。
2. **数据大小** (`size`): 指定 `data` 指向的数据的大小（字节数）。
3. **呈现时间戳** (`pts`): 表示数据包的呈现时间戳（Presentation
TimeStamp），用于在播放时确定数据包的显示时间。
4. **解码时间戳** (`dts`): 表示数据包的解码时间戳（Decoding
TimeStamp），用于确定数据包何时应该开始解码。
5. **序列号** (`stream_index`): 标识数据包所属的媒体流（如视频流、音频流等）。
6. **元数据** (`flags`):
包含关于数据包的额外信息，例如是否是关键帧（keyframe）。
7. **持续时间** (`duration`): 数据包的持续时间，以时间基（time_base）为单位。
8. **销毁回调** (`destruct`): 当 `AVPacket` 不再需要时，用于自动释放
`AVPacket`相关资源的回调函数。
9. **私有数据** (`opaque`): 一个私有数据指针，可用于存储私有数据或上下文信息。
10. **侧边数据** (`side_data`):
存储与数据包相关的额外数据，如字幕、视频帧的编码参数等。
`AVPacket` 结构体是 FFmpeg 处理多媒体数据流的基础，它允许开发者在不同的 FFmpeg
组件之间高效地传递数据。通过
`AVPacket`，可以访问和操作原始的多媒体数据，实现复杂的多媒体处理任务。

在 FFmpeg 中，`AVPacket` 通常与 `AVFrame` 结构体一起使用，其中 `AVFrame`
用于存储解码后的视频帧或音频样本。`AVPacket` 携带原始编码数据，而 `AVFrame`
则包含解码后的数据，两者共同协作完成多媒体数据的解码和处理。


-----------------
JPEG：使用全范围（Full Range）颜色表示，提供 0 到 255
的颜色取值范围，适用于静态图像，能提供更多的色彩细节。

MPEG：使用有限范围（Limited Range）颜色表示，亮度分量在 16 到 235
之间，色度分量在 16 到 240 之间，适用于视频，减少了信号处理中的失真

BT.601 是一种用于标准清晰度电视的颜色空间标准，定义了 YCbCr 颜色空间及其与
RGB颜色空间的转换关系。

特性          BT.601                    BT.709
应用领域    标准清晰度电视（SDTV）  高清晰度电视（HDTV）
色彩空间      YCbCr                     YCbCr
色域           较小                     较大
取样率      通常为 4:2:2            通常为 4:2:2 或 4:2:0

分辨率      720P                        1280x720 或 1920x1080

颜色空间与颜色范围结合的作用

1 图像质量：
全范围YCbCr：提供更大的色彩细节和动态范围，适用于高质量图像（如照片、静态图像）。
有限范围YCbCr：通过限制取值范围，避免信号处理中的过冲和下冲，减少视频传输和显示中的失真，适用于视频内容。

2 数据压缩：

YCbCr颜色空间：通过分离亮度和色度，可以更有效地进行数据压缩，因为人眼对亮度的变化更敏感，对色度的变化不太敏感。
有限范围：减少色度分量的动态范围，使得压缩算法能够更有效地压缩数据。

3 兼容性：

视频标准：许多视频标准（如
MPEG、H.264、H.265）规定使用有限范围的YCbCr（），以确保不同设备和平台之间的兼容性。

图像标准：JPEG 图像通常使用全范围的YCbCr，确保最大化图像质量。

*/