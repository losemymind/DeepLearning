#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include "functional.hpp"
#include "Matrix.hpp"

//Back Propagation Neural Network
class BPNN
{
public:
    BPNN()
    {

    }
    void initialize(Matrix<size_t> LayersInfo, float InLearnRate = 0.1, float InAttenuate = 1)
    {
        size_t LayerNum = LayersInfo.col();
        for (size_t i = 0; i < LayerNum; ++i)
        {
            Layers.push_back(Matrix<float>(1, LayersInfo.get(0, i)));
            Deltas.push_back(Matrix<float>(1, LayersInfo.get(0, i)));
        }

        for (size_t i = 0; i < LayerNum - 1; ++i)
        {
            Weights.push_back(Matrix<float>(Layers[i].col(), Layers[i + 1].col()));
        }

        for (size_t i = 0; i < LayerNum - 1; ++i)
        {
            Bias.push_back(Matrix<float>(1, Layers[i+1].col()));
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

    bool train(const Matrix<float>& input, const Matrix<float>& output, float nor = 1)
    {
        Matrix<float>& InputLayer = Layers[0];
        InputLayer = input;
        InputLayer.normalize1(nor);
        // 正向传播
        for (size_t l = 0; l < Layers.size() - 1; ++l)
        {
            forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
        }

        // 判断误差
        total_cost = 0.0;
        Matrix<float>& LayerN = Layers[Layers.size() - 1];
        Matrix<float> Aberration = Layers[Layers.size() - 1];
        Aberration.subtract(output, LayerN);
        Aberration.foreach_c([this](Matrix<float>::value_type e)
        {
            total_cost += e * e / 2;
        });
        total_cost = total_cost / output.col();

        std::string trainInfo;
        trainInfo += "output ";
        trainInfo += ": ";
        trainInfo += LayerN.to_string();
        trainInfo += " ";
        printf("%s cost:%0.32f \n", trainInfo.c_str(), total_cost);

        // 反向修正
        Matrix<float>& DeltasN = Deltas[Deltas.size() - 1];
        DeltasN.hadamard(Aberration.negate(), LayerN.foreach_n(DL::dsigmoid));
        //DeltasN.hadamard(Aberration.negate(), LayerN.foreach(DL::dsigmoid));
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

    float simulate(const Matrix<float>& input, Matrix<float>& output, const Matrix<float>& expect, float nor = 1)
    {
        Matrix<float>& InputLayer = Layers[0];
        InputLayer = input;
        InputLayer.normalize1(nor);

        total_cost = 0.0;
        // 正向传播
        for (size_t l = 0; l < Layers.size() - 1; ++l)
        {
            forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
        }

        // 判断误差
        Matrix<float> Aberration = Layers[Layers.size() - 1];

        output  = Layers[Layers.size() - 1];
        Aberration.subtract(expect, output);
        Aberration.foreach_c([this](Matrix<float>::value_type e)
        {
            total_cost += e * e / 2;
        });
        total_cost = total_cost / output.col();
        return total_cost;
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

        str += "cost:";
        str += std::to_string(total_cost);

        return str;
    }

    float get_cost()
    {
        return total_cost;
    }

    // 权重衰减项
    float weights_attenuate()
    {
        float allWeights = 0.0;
        for (auto& w : Weights)
        {
            w.foreach_c([&allWeights](float e) 
            {
                allWeights += e;
            });
        }

        return Attenuate / 2 * allWeights;
    }

private:
    std::vector<Matrix<float>> Layers;

    std::vector<Matrix<float>> Weights;

    std::vector<Matrix<float>> Bias;

    std::vector<Matrix<float>> Deltas;

    float                      total_cost;
    float                      LearnRate;
    float                      Attenuate;
};
