#ifndef TENSORFLOW_H
#define TENSORFLOW_H

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"



// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

class LFNET
{
public:
    LFNET(const string model_dir, int input_width, int input_height, int kpt_k, int dspt_k);
    //参数定义
    const string mmodel_dir;
    string mimage = "/home/zoe/data/rgbd_dataset_freiburg1_room/rgb/1305031910.765238.png";
    int32 minput_width;
    int32 minput_height;
    float minput_mean;
    float minput_std;
    int mkpt_k;
    int mdspt_k;
    string minput_layer;
    string mkpts_layer;
    string mkpts_ori_layer;
    string mkpts_scale_layer;
    string mdspts_layer;
    int mload_model;

    std::vector<cv::KeyPoint> mvkpts;
    std::vector<std::vector<float>> mvdspts;

    std::unique_ptr<tensorflow::Session> msession;

    //函数定义
    Status LoadGraph(std::unique_ptr<tensorflow::Session>* session);
    Status ReadEntireFile(tensorflow::Env* env, const string& filename, Tensor* output);
    Status ReadTensorFromImageFile(std::vector<Tensor>* out_tensors);
    void Predict(const cv::Mat &imgray);
};
#endif // TENSORFLOW_H
