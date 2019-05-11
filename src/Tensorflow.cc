#include "Tensorflow.h"


LFNET::LFNET(const string model_dir, int input_width, int input_height, int kpt_k, int dspt_k):
mmodel_dir(model_dir), minput_width(input_width),
minput_height(input_height), mkpt_k(kpt_k), mdspt_k(dspt_k)
{
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);// 设置第一个GPU可见
    //参数定义
    minput_mean = 0.0;
    minput_std = 255.0;
    minput_layer = "Placeholder";
    mkpts_layer = "MSDeepDet/add_5";
    mkpts_ori_layer = "Atan2_1";
    mkpts_scale_layer = "MSDeepDet/GatherNd";
    mdspts_layer = "DeepDesc/SimpleDesc/l2_normalize";
    mload_model = 0;
    Status load_graph_status = LoadGraph(&msession);
    if (!load_graph_status.ok()) 
    {
        LOG(ERROR) << load_graph_status;
    }
    else
    {
        mload_model = 1;
    }
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LFNET::LoadGraph( std::unique_ptr<tensorflow::Session>* session) 
{
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), mmodel_dir, &graph_def);
    if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        mmodel_dir, "'");
    }
    msession.reset(tensorflow::NewSession(tensorflow::SessionOptions()));

    Status session_create_status = msession->Create(graph_def);
    if (!session_create_status.ok()) {
    return session_create_status;
    }
    return Status::OK();
}

Status LFNET::ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) 
{
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status LFNET::ReadTensorFromImageFile(std::vector<Tensor>* out_tensors)
{
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), mimage, &input));

    // use a placeholder to read input data
    auto file_reader = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = 
    {
        {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 1;
    tensorflow::Output image_reader;
    if (tensorflow::str_util::EndsWith(mimage, ".png"))
    {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                DecodePng::Channels(wanted_channels));
    } 
    else if (tensorflow::str_util::EndsWith(mimage, ".gif")) 
    {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } 
    else if (tensorflow::str_util::EndsWith(mimage, ".bmp")) 
    {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
    } 
    else 
    {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                DecodeJpeg::Channels(wanted_channels));
    }

    // Now cast the image data to float so we can do normal math on it.
    auto float_caster = Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root, float_caster, 0);

    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(
        root, dims_expander,
        Const(root.WithOpName("size"), {minput_height, minput_width}));

    // Subtract the mean and divide by the scale.
    Div(root.WithOpName(output_name), Sub(root, resized, {minput_mean}),
        {minput_std});

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
    return Status::OK();
}

void LFNET::Predict(const cv::Mat &imgray)
{
    /*
    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main graph expects.
    std::vector<Tensor> resized_tensors;

    Status read_tensor_status = ReadTensorFromImageFile(&resized_tensors);
    if (!read_tensor_status.ok()) 
    {
        LOG(ERROR) << read_tensor_status;
    }
    Tensor resized_tensor = resized_tensors[0];
    */

    // allocate a Tensor
    cv::Mat imgray_float;
    imgray.convertTo(imgray_float, CV_32FC1, 1.0f/255.0f);
    float *imgray_float_data = (float*)imgray_float.data;
    tensorflow::TensorShape image_shape = tensorflow::TensorShape({1, minput_height, minput_width, imgray.channels()});
    Tensor image_input(tensorflow::DT_FLOAT, image_shape);
    std::copy_n((char*) imgray_float_data, image_shape.num_elements()*sizeof(float), const_cast<char*>(image_input.tensor_data().data()));

    Tensor phase_train(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_train.scalar<bool>()() = false;
    std::vector<std::pair<std::string, Tensor>> inputs = {
        {minput_layer, image_input},
    };  
    std::vector<Tensor> outputs;
    Status run_status = msession->Run({inputs},
                                    {mkpts_layer, mkpts_ori_layer, mkpts_scale_layer, mdspts_layer,"Const"}, {}, &outputs);
                                    //{feats_layer}, {}, &outputs);
    if (!run_status.ok()) 
    {
        LOG(ERROR) << "Running model failed: " << run_status;
    }

    // Fetch the first tensor
    Tensor kpts = outputs[0];
    Tensor kpts_ori = outputs[1];
    Tensor kpts_scale = outputs[2];
    Tensor dspts = outputs[3];
    Tensor train = outputs[4];
    /*
    std::cout << mkpts_layer << " shape:" << kpts.shape() << std::endl;
    std::cout << mkpts_ori_layer << " shape:" << kpts_ori.shape() << std::endl;
    std::cout << mkpts_scale_layer << " shape:" << kpts_scale.shape() << std::endl;
    std::cout << mdspts_layer << " shape:" << dspts.shape() << std::endl;
    */
    auto kptsmap = kpts.tensor<float, 2>(); 
    auto kptsorimap = kpts_ori.tensor<float, 1>(); 
    auto kptsscalemap = kpts_scale.tensor<float, 1>(); 
    auto dsptsmap = dspts.tensor<float, 2>();
    auto trainmap = train.tensor<bool, 1>();
    std::cout<<trainmap(0)<<std::endl;
    
    mvkpts.resize(mkpt_k);
    mvdspts.clear();
    /*
    std::ofstream outfile;
    outfile.open("/home/zoe/test.txt");
    if(!outfile.is_open())
    {
        std::cout<<" the file open fail"<<std::endl;
        exit(1);
    }
    */
    for (int i = 0; i < mkpt_k; i++)
	{
        mvkpts[i].pt.x = kptsmap(i,0);
        mvkpts[i].pt.y = kptsmap(i,1);
        //outfile << kptsmap(i,0) << " " << kptsmap(i,1);
        mvkpts[i].angle = (float)(kptsorimap(i) * 180.0f / CV_PI + 180.0f);
        mvkpts[i].octave = 0;//kptsscalemap(i);
        std::vector<float> dspt;
        dspt.resize(mdspt_k);
        for (int j = 0; j < mdspt_k; j++)
        {
            if(dsptsmap(i,j) < -0.3)
                dspt[j] = -0.3f;//0
            else if(dsptsmap(i,j) > 0.3)
                dspt[j] = 0.3f;//255
            else
                dspt[j] = dsptsmap(i,j);
            
            //outfile << dsptsmap(i,j) << " ";
        }
        std::cout << dsptsmap(i,1) << std::endl;
        mvdspts.push_back(dspt);
        //outfile<<"\r\n";
    }
    //outfile.close();
}

int main(int argc, char* argv[]) 
{
    const string dir = "/home/zoe/catkin_ws/src/ORB_SLAM2_LFNet/LFNet/lf_bn.pb";
    //string image = "data/rgbd_dataset_freiburg1_room/rgb/1305031910.765238.png";
    int input_width = 640;
    int input_height = 480;   
    int kpt_k = 1000;
    int dspt_k = 256;
    
    cv::Mat image_read = cv::imread("/home/zoe/data/rgbd_dataset_freiburg1_room/rgb/1305031910.797230.png",CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat image_gray;
    
    if(!image_read.data)
    {
        std::cerr << "image read1 failed" << std::endl;
    }
    cvtColor(image_read, image_gray, CV_RGB2GRAY);
    
    LFNET* mpLFNet = new LFNET(dir, input_width, input_height, kpt_k, dspt_k);
    
    // First we load and initialize the model.
    mpLFNet->Predict(image_gray);

    return 0;
}