/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2023-07-21 09:54:48
 * @LastEditors: zwy
 * @LastEditTime: 2023-07-23 12:41:02
 */

#ifndef _RTSP2FLV_HPP
#define _RTSP2FLV_HPP

#include <iostream>
#include <vector>

//#include "clipp.h"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

cv::VideoCapture get_device(int camID, double width, double height);
cv::VideoCapture get_device(std::string camID, double width, double height);

void initialize_avformat_context(AVFormatContext*& fctx,
                                 const char* format_name);
void initialize_io_context(AVFormatContext*& fctx, const char* output);
void set_codec_params(AVFormatContext*& fctx, AVCodecContext*& codec_ctx,
                      double width, double height, int fps, int bitrate);
void initialize_codec_stream(AVStream*& stream, AVCodecContext*& codec_ctx,
                             const AVCodec*& codec, std::string codec_profile);
SwsContext* initialize_sample_scaler(AVCodecContext* codec_ctx, double width,
                                     double height);
AVFrame* allocate_frame_buffer(AVCodecContext* codec_ctx, double width,
                               double height);
void write_frame(AVCodecContext* codec_ctx, AVFormatContext* fmt_ctx,
                 AVFrame* frame);
void stream_video(double width, double height, int fps, int camID, int bitrate,
                  std::string codec_profile, std::string server);
void stream_video(double width, double height, int fps, std::string camID,
                  int bitrate, std::string codec_profile, std::string server);

// 自定义异常类
class FFmpegException : public std::exception
{
public:
  FFmpegException(const char* msg) : msg_(msg) {}

  const char* what() const noexcept override { return msg_; }

private:
  const char* msg_;
};

// 自定义错误处理回调函数
void ffmpeg_error_handler(void* ptr, int level, const char* fmt, va_list vl);

#endif