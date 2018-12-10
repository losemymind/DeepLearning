#include "Matrix.hpp"
#include "BPNN.hpp"
#include <iostream>
#include <fstream>

static std::string TrainImagePath = "E:\\GitHub\\DeepLearning\\Debug\\data\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
static std::string TrainLabelPath = "E:\\GitHub\\DeepLearning\\Debug\\data\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";
static std::string TestImagePath = "E:\\GitHub\\DeepLearning\\Debug\\data\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte";
static std::string TestLabelPath = "E:\\GitHub\\DeepLearning\\Debug\\data\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte";

int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(std::string filename, std::vector<double>&labels)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of images = " << number_of_images << std::endl;
        labels.reserve(number_of_images);
        for (int i = 0; i < number_of_images; i++)
        {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back((double)label);
        }
    }
}

void read_Mnist_Images(std::string filename, std::vector<Matrix<double>>&images)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);

        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of images = " << number_of_images << std::endl;
        std::cout << "rows = " << n_rows << std::endl;
        std::cout << "cols = " << n_cols << std::endl;
        images.reserve(number_of_images);
        for (int i = 0; i < number_of_images; i++)
        {
            Matrix<double>tp(1,784);
            size_t index = 0;
            for (int r = 0; r < n_rows; r++)
            {
                for (int c = 0; c < n_cols; c++)
                {
                    unsigned char image = 0;
                    file.read((char*)&image, sizeof(image));
                    tp.set(0, index++, image);
                }
            }
            images.push_back(tp);
        }
    }
}

void  MNIST()
{
    BPNN nn;
    nn.initialize(Matrix<size_t>({ 784,100,10 }), 0.3);
    std::vector <Matrix<double>> DataSet;
    std::vector<double> LabelSet;
    read_Mnist_Images(TrainImagePath, DataSet);
    read_Mnist_Label(TrainLabelPath, LabelSet);
    for (size_t times = 0; times < DataSet.size(); ++times)
    {
        size_t randomnum = DL::random(0, 59999);
        printf("image index :%d ", randomnum);
        Matrix<double> XSet = DataSet[randomnum];
        Matrix<double> YSet(1, 10);
        YSet.set(0, LabelSet[randomnum], 1.0);
        nn.train(XSet, YSet, 255.0);
    }
    std::vector <Matrix<double>> TestDataSet;
    std::vector<double> TestLabelSet;
    read_Mnist_Images(TestImagePath, TestDataSet);
    read_Mnist_Label(TestLabelPath, TestLabelSet);
    size_t test_num = 0;
    for (size_t image_size = 0; image_size < TestDataSet.size(); ++image_size)
    {
        Matrix<double> XSet = TestDataSet[image_size];
        Matrix<double> YSet(1, 10);
        Matrix<double> ExpectSet(1, 10);
        ExpectSet.set(0, TestLabelSet[image_size], 1.0);
        nn.simulate(XSet, YSet, ExpectSet, 255.0);
        double max_val = 0.0;
        int    max_idx = 0;
        for (int i = 0; i < 10; ++i)
        {
            if (YSet.get(0, i) > max_val)
            {
                max_val = YSet.get(0, i);
                max_idx = i;
            }
        }
        if (max_idx == TestLabelSet[image_size])
        {
            ++test_num;
        }
    }
    double Accuracy = test_num * 1.0 / TestDataSet.size();
    printf("Accuracy = %f", Accuracy);
}

int main()
{
    MNIST();

    system("pause");
    return 0;
}