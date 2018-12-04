#include "Matrix.hpp"
#include "BPNN.hpp"
#include <iostream>

int main()
{

    BPNN nn;

    nn.initialize(Matrix<size_t>({ 1,3,1}), 1);

    for (int times = 0; times < 10000; ++times)
    {
        printf("times:%d \n", times);
        Matrix<float> DataSet = { 1 };
        Matrix<float> LabelSet = { 0.1 };
        nn.train(DataSet, LabelSet);

        DataSet = { 2 };
        LabelSet = { 0.9 };
        nn.train(DataSet, LabelSet);

        auto alldata = nn.to_string();

    }

    //std::vector<double> DataSet(4);
    //std::vector<double> LabelSet(2);
    //LabelSet[0] = 0.9; // ÆæÊý
    //LabelSet[1] = 0.1; // Å¼Êý
    //for (int times = 0; times < 10000; ++times)
    //{
    //    DataSet[0] = 10;
    //    DataSet[1] = 11;
    //    DataSet[2] = 12;
    //    DataSet[3] = 13;
    //    nn.train(DataSet, LabelSet);
    //}

    //std::cout << simulateInfo << "Aberration:" << (float)Aberration << std::endl;
    system("pause");
    return 0;
}