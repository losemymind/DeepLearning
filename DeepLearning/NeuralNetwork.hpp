#pragma once
#include <vector>
#include <iostream>
#include "functional.hpp"
#include "Matrix.hpp"
class NeuralNetwork
{
public:
    NeuralNetwork()
    {

    }
    void initialize(Matrix<size_t> LayersInfo, double InLearnRate = 0.1, double InAttenuate = 1)
    {
        size_t LayerNum = LayersInfo.col();
        for (size_t i = 0; i < LayerNum; ++i)
        {
            Layers.push_back(Matrix<double>(1, LayersInfo.get(0, i)));
            Deltas.push_back(Matrix<double>(1, LayersInfo.get(0, i)));
        }

        for (size_t i = 0; i < LayerNum - 1; ++i)
        {
            Weights.push_back(Matrix<double>(Layers[i].col(), Layers[i + 1].col()));
        }

        for (size_t i = 0; i < LayerNum - 1; ++i)
        {
            Bias.push_back(Matrix<double>(1, Layers[i+1].col()));
        }

        for (auto& ws: Weights)
        {
            ws.random(0, 1);
        }

        for (auto& bs : Bias)
        {
            bs.random(0, 1);
        }
        LearnRate = InLearnRate;
        Attenuate = InAttenuate;
    }
    template<class LT, class W, class B>
    void forward(LT& LayerX, LT& LayerY, W& InWeights, B& InBias)
    {
        LayerY.multiply(LayerX, InWeights);
        //LayerY.foreach([&LayerY](auto& e) { return e / LayerY.col(); }); // 用于支持超大节点数
        LayerY += InBias;
        LayerY.foreach(DL::sigmoid);
    }

    template<class LX, class W, class B, class D>
    void backward(LX& LayerX, W& InWeights, B& InBias, D& DeltaX, D& DeltaY)
    {
        InWeights.update_weights(LayerX, DeltaY, LearnRate);
        InBias.update_bias(DeltaY, LearnRate);
        DeltaX.deltas(InWeights, DeltaY);
        DeltaX.hadamard(LayerX.foreach_n(DL::dsigmoid));
        // or
        //LayerX.foreach(DL::dsigmoid);
        //DeltaX.hadamard(LayerX);
    }

    bool train(const Matrix<double>& input, const Matrix<double>& output, double nor = 1)
    {
        Matrix<double>& InputLayer = Layers[0];
        InputLayer = input;
        InputLayer.normalize1(nor);
        // 正向传播
        for (size_t l = 0; l < Layers.size() - 1; ++l)
        {
            forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
        }

        // 判断误差
        LastAberration = 0.0;
        Matrix<double>& LayerN = Layers[Layers.size() - 1];
        Matrix<double> Aberration = Layers[Layers.size() - 1];
        Aberration.subtract(output, LayerN);
        Aberration.foreach_c([this](Matrix<double>::value_type e)
        {
            LastAberration += e * e / 2;
        });
        LastAberration = LastAberration / output.col();

        std::string trainInfo;
        trainInfo += "output ";
        trainInfo += ": ";
        trainInfo += LayerN.to_string();
        trainInfo += " ";
        printf("%s Lose:%0.32f \n", trainInfo.c_str(), LastAberration);

        // 反向修正
        Matrix<double>& DeltasN = Deltas[Deltas.size() - 1];
        DeltasN.multiply(Aberration.negate(), LayerN.foreach_n(DL::dsigmoid));
        size_t LayerCount = Layers.size();
        for (size_t l = 0; l < LayerCount - 1; ++l)
        {
            backward(Layers[LayerCount - l - 2],
                Weights[LayerCount - l - 2],
                Bias[LayerCount - l - 2],
                Deltas[LayerCount - l - 2],
                Deltas[LayerCount - l - 1]);
        }
        return false;
    }

    double simulate(const Matrix<double>& input, Matrix<double>& output, const Matrix<double>& expect, double nor = 1)
    {
        Matrix<double>& InputLayer = Layers[0];
        InputLayer = input;
        InputLayer.normalize1(nor);

        LastAberration = 0.0;
        // 正向传播
        for (size_t l = 0; l < Layers.size() - 1; ++l)
        {
            forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
        }

        // 判断误差
        Matrix<double> Aberration = Layers[Layers.size() - 1];

        output  = Layers[Layers.size() - 1];
        Aberration.subtract(expect, output);
        Aberration.foreach_c([this](Matrix<double>::value_type e)
        {
            LastAberration += e * e / 2;
        });
        LastAberration = LastAberration / output.col();
        return LastAberration;
    }

    std::string to_string()
    {
        std::string str = "Layers:{\n";
        for (auto& val : Layers)
        {
            str += val.to_string();
            str += "\n";
            str += "------------------------------------------------------------------------\n";
            str += "\n";
        }
        str += "}\nWeights:{\n";
        for (auto& val : Weights)
        {
            str += val.to_string();
            str += "\n";
            str += "------------------------------------------------------------------------\n";
            str += "\n";
        }
        str += "}\nBias:{\n";
        for (auto& val : Bias)
        {
            str += val.to_string();
            str += "\n";
            str += "------------------------------------------------------------------------\n";
            str += "\n";
        }
        str += "}\n";

        str += "LastAberration:";
        str += std::to_string(LastAberration);

        return str;
    }

private:
    std::vector<Matrix<double>> Layers;

    std::vector<Matrix<double>> Weights;

    std::vector<Matrix<double>> Bias;

    std::vector<Matrix<double>> Deltas;

    double                      LastAberration;
    double                      LearnRate;
    double                      Attenuate;
};
