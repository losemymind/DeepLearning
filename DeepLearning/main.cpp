#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include <iostream>

int main()
{

    NeuralNetwork nn;

    nn.initialize(Matrix<size_t>({ 1,3,2,1}), 1);

    std::vector<double> DataSet(1);
    std::vector<double> LabelSet(1);

    for (int i = 0; i < 10; ++i)
    {
        DataSet[0] = i * 2.0;
        LabelSet[0] = 0; // 1是偶数，0是奇数
        nn.train(DataSet, LabelSet, 10);
    }
    std::cout << "=======================" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        DataSet[0] = i * 2.0 +1;
        LabelSet[0] = 0; // 1是偶数，0是奇数
        nn.train(DataSet, LabelSet, 10);
    }


    std::vector<double> Input(1);
    std::vector<double> Output(1);
    std::vector<double> expect(1);

    for (int i = 0; i < 10; ++i)
    {
        Input[0] = i * 1.0;
        expect[0] = (i % 2 == 0) ? 1 : 0; // 1是偶数，0是奇数
        auto Aberration = nn.simulate(Input, Output, expect);
        std::cout << "input:"<< i <<" Aberration:" << Aberration << std::endl;
    }

    system("pause");
    return 0;
}