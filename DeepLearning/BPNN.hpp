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
    void initialize(Matrix<size_t> LayersInfo, double InLearnRate = 0.1, double InAttenuate = 1)
    {
        size_t LayerNum = LayersInfo.col();
        for (size_t i = 0; i < LayerNum; ++i)
        {
            Layers.push_back(Matrix<double>(1, LayersInfo.get(0, i)));
            Deltas.push_back(Matrix<double>(1, LayersInfo.get(0, i)));
        }
        Aberration = Layers[Layers.size() - 1];

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
    void forward(LT& LayerX, LT& LayerY, W& InWeights, B& InBias, bool last = false)
    {
        LayerY.multiply(LayerX, InWeights);
        LayerY.foreach([&LayerY](auto& e) { return e / LayerY.col(); }); // 用于支持超大节点数
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
    }

    void train(const Matrix<double>& input, const Matrix<double>& output, double nor = 1)
    {
        Matrix<double>& InputLayer = Layers[0];
        InputLayer = input;
        InputLayer.normalize1(nor);
        total_cost = 0.0;
        // 正向传播
        for (size_t l = 0; l < Layers.size() - 1; ++l)
        {
            forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
        }

        // 判断误差
        Matrix<double>& LayerN = Layers[Layers.size() - 1];
        Aberration.subtract(output, LayerN);
        Aberration.foreach_c([this](Matrix<double>::value_type e)
        {
            total_cost += e * e / 2;
        });
        total_cost = total_cost / LayerN.col();

        printf("cost:%0.32f output:%s \n", total_cost, LayerN.to_string().c_str());

        // 反向修正
        Matrix<double>& DeltasN = Deltas[Deltas.size() - 1];
        DeltasN.hadamard(Aberration.negate(), LayerN.foreach_n(DL::dsigmoid));

        size_t LayerCount = Layers.size();
        for (size_t l = 0; l < LayerCount - 1; ++l)
        {
            backward(Layers[LayerCount - l - 2],
                Weights[LayerCount - l - 2],
                Bias[LayerCount - l - 2],
                Deltas[LayerCount - l - 2],
                Deltas[LayerCount - l - 1]);
        }
    }

    double simulate(const Matrix<double>& input, Matrix<double>& output, const Matrix<double>& expect, double nor = 1)
    {
        Matrix<double>& InputLayer = Layers[0];
        InputLayer = input;
        InputLayer.normalize1(nor);
        total_cost = 0.0;
        // 正向传播
        for (size_t l = 0; l < Layers.size() - 1; ++l)
        {
            forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
        }

        // 判断误差
        output = Layers[Layers.size() - 1];
        Aberration.subtract(expect, output);
        Aberration.foreach_c([this](Matrix<double>::value_type e)
        {
            total_cost += e * e / 2;
        });
        total_cost = total_cost / output.col();
        output = Layers[Layers.size() - 1];
        printf("cost:%0.32f output:%s \n", total_cost, output.to_string().c_str());
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

    double get_cost()
    {
        return total_cost;
    }

    // 权重衰减项
    double weights_attenuate()
    {
        double allWeights = 0.0;
        for (auto& w : Weights)
        {
            w.foreach_c([&allWeights](double e) 
            {
                allWeights += e;
            });
        }

        return Attenuate / 2 * allWeights;
    }

private:
    std::vector<Matrix<double>> Layers;
    std::vector<Matrix<double>> Weights;
    std::vector<Matrix<double>> Bias;
    std::vector<Matrix<double>> Deltas;
    Matrix<double>              Aberration;
    double                      total_cost;
    double                      LearnRate;
    double                      Attenuate;

};

