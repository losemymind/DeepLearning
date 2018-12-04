#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include <iostream>

int main()
{

    NeuralNetwork nn;

    nn.initialize(Matrix<size_t>({ 1,2,2,1}), 1);

    for (int times = 0; times < 10000; ++times)
    {
        printf("times:%d \n", times);
        Matrix<double> DataSet = { 1 };
        Matrix<double> LabelSet = { 0.1 };
        nn.train(DataSet, LabelSet);

        DataSet = { 2 };
        LabelSet = { 0.9 };
        nn.train(DataSet, LabelSet);
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