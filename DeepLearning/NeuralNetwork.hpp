#pragma once
#include <vector>
#include <iostream>
#include "Matrix.hpp"
#include "functional.hpp"

class NeuralNetwork
{
public:
    NeuralNetwork()
    {

    }

    void initialize(Matrix<size_t> LayersInfo, double InAttenuate)
    {
        size_t LayerNum = LayersInfo.col();
        for (size_t i = 0; i < LayerNum; ++i)
        {
            Layers.push_back(std::vector<double>(LayersInfo.get(0, i)));
            Deltas.push_back(std::vector<double>(LayersInfo.get(0, i)));
        }

        for (size_t i = 0; i < LayerNum - 1; ++i)
        {
            std::vector<std::vector<double>> ovec;
            for (size_t j = 0; j < Layers[i+1].size(); ++j)
            {
                std::vector<double> wvec;
                for (size_t k = 0; k < Layers[i].size(); ++k)
                {
                    wvec.push_back(RandomN(0, 1));
                }
                ovec.push_back(wvec);
            }
            Weights.push_back(ovec);
        }

        for (size_t i = 1; i < LayerNum; ++i)
        {
            auto BiasMatrix = std::vector<double>(Layers[i].size());
            for (auto& val : BiasMatrix)
            {
                val = RandomN(0, 1);
            }
            Bias.push_back(BiasMatrix);
        }

        Aberration = Layers[Layers.size() - 1];
        LearnRate = 0.001;
        Attenuate = InAttenuate;
    }
    template<class LT, class W, class B>
    void forward(LT& LayerX, LT& LayerY, W& InWeights, B& InBias)
    {
        for (auto i = 0; i < LayerY.size(); ++i)
        {
            auto ws = InWeights[i];
            for (auto j = 0; j < ws.size(); ++j)
            {
                LayerY[i] += ws[j] * LayerX[j];
            }
            LayerY[i] += InBias[i];
            LayerY[i] = Sigmoid(LayerY[i]);
        }
    }

    template<class LX, class W, class B, class D>
    void backward(LX& layerX, W& InWeights, B& InBias, D& deltaX, D& deltaY)
    {
        for (int i = 0; i < InWeights.size(); ++i)
        {
            for (int j = 0; j < InWeights[i].size(); ++j)
            {
                InWeights[i][j] = InWeights[i][j] - LearnRate * layerX[j] * deltaY[i];

                deltaX[j] += InWeights[i][j] * deltaY[i];
            }
        }

        for (auto i = 0; i < InBias.size(); ++i)
        {
            InBias[i] = InBias[i] - LearnRate * deltaY[i];
        }

        for (int i = 0; i < deltaX.size(); ++i)
        {
            deltaX[i] = deltaX[i] * SigmoidDerivative(layerX[i]);
        }
    }

    bool train(const std::vector<double>& input, const std::vector<double>& output, int times = 1, double nor = 1)
    {
        std::vector<double>& InputLayer = Layers[0];
        InputLayer = input;
        for (auto& val : InputLayer)
        {
            val = Normalize1(val, nor);
        }

        std::vector<double>& LayerN = Layers[Layers.size() - 1];
        std::vector<double>& DeltasN = Deltas[Deltas.size() - 1];
        const size_t LayerCount = Layers.size();
        for (int t = 0; t < times; ++t)
        {
            LastAberration = 0.0;
            // 正向传播
            for (size_t l = 0; l < Layers.size() - 1; ++l)
            {
                forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
            }

            for (auto i = 0; i < Aberration.size(); ++i)
            {
                Aberration[i] = output[i]- LayerN[i];
                LastAberration += Aberration[i] * Aberration[i]/2;
            }

            // 判断误差
            LastAberration = LastAberration /output.size();
            std::cout << "Lose:" << LastAberration << std::endl;

            // 反向修正
            for (auto i = 0; i < DeltasN.size(); ++i)
            {
                DeltasN[i] = -Aberration[i] * SigmoidDerivative(LayerN[i]);
            }

            for (size_t l = 0; l < Layers.size()-1; ++l)
            {
                backward(Layers[LayerCount - l - 2],
                    Weights[LayerCount - l - 2],
                    Bias[LayerCount - l - 2],
                    Deltas[LayerCount - l - 2],
                    Deltas[LayerCount - l - 1]);
            }
        }
        return true;
    }

    double simulate(const std::vector<double>& input, std::vector<double>& output, const std::vector<double>& expect, double nor = 1)
    {
        LastAberration = 0.0;
        std::vector<double>& InputLayer = Layers[0];
        InputLayer = input;
        for (auto& val : InputLayer)
        {
            val = Normalize1(val, nor);
        }
        // 正向传播
        for (size_t l = 0; l < Layers.size() - 1; ++l)
        {
            forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
        }

        output = Layers[Layers.size() - 1];
        for (auto i = 0; i < Aberration.size(); ++i)
        {
            Aberration[i] = expect[i] - output[i];
            LastAberration += Aberration[i] * Aberration[i]/2;
        }

        // 判断误差
        LastAberration = LastAberration / output.size();
        return LastAberration;
    }

    std::string to_string()
    {
        std::string str = "Layers:{\n";
        //for (auto& val : Layers)
        //{
        //    str += val.to_string();
        //    str += "\n";
        //    str += "------------------------------------------------------------------------\n";
        //    str += "\n";
        //}
        //str += "}\nWeights:{\n";
        //for (auto& val : Weights)
        //{
        //    str += val.to_string();
        //    str += "\n";
        //    str += "------------------------------------------------------------------------\n";
        //    str += "\n";
        //}
        //str += "}\nBias:{\n";
        //for (auto& val : Bias)
        //{
        //    str += val.to_string();
        //    str += "\n";
        //    str += "------------------------------------------------------------------------\n";
        //    str += "\n";
        //}
        //str += "}\n";

        //str += "LastAberration:";
        //str += std::to_string(LastAberration);

        return str;
    }

private:
    std::vector<std::vector<double>> Layers;

    std::vector<std::vector<std::vector<double>>>   Weights;

    std::vector<std::vector<double>> Bias;

    std::vector<std::vector<double>> Deltas;

    std::vector<double>         Aberration;
    double                      AberrationThreshold;
    double                      LastAberration;
    double                      LearnRate;
    double                      Attenuate;
};

//class NeuralNetwork
//{
//public:
//    NeuralNetwork()
//    {
//
//    }
//
//    void initialize(Matrix<size_t> LayersInfo, double InAttenuate)
//    {
//        size_t LayerNum = LayersInfo.col();
//        for (size_t i = 0; i < LayerNum; ++i)
//        {
//            Layers.push_back(Matrix<double>(1, LayersInfo.get(0, i)));
//            Deltas.push_back(Matrix<double>(1, LayersInfo.get(0, i)));
//        }
//
//        for (size_t i = 0; i < LayerNum-1; ++i)
//        {
//            auto WeightsMatrix = Matrix<double>(Layers[i].col(), Layers[i + 1].col());
//            WeightsMatrix.random(0, 1);
//            Weights.push_back(WeightsMatrix);
//        }
//
//        for (size_t i = 1; i < LayerNum; ++i)
//        {
//            auto BiasMatrix = Matrix<double>(1, Layers[i].col());
//            BiasMatrix.random(0,1);
//            Bias.push_back(BiasMatrix);
//        }
//
//        Aberration = Layers[Layers.size() - 1];
//        AberrationThreshold = 0.001;
//        LearnRate = 0.1;
//        Attenuate = InAttenuate;
//    }
//    template<class LT, class W, class B>
//    void forward(LT& LayerX, LT& LayerY, W& InWeights, B& InBias)
//    {
//        LayerY.multiply(LayerX,InWeights); // y = w.x;
//        LayerY += InBias;
//        LayerY.foreach(Sigmoid);
//    }
//
//    template<class LX, class W, class B, class D>
//    void backward(LX& layerX, W& InWeights, B& InBias, D& deltaX, D& deltaY)
//    {
//        InWeights.update_w(layerX, deltaY, LearnRate);
//        InBias.update_b(deltaY, LearnRate);
//        deltaX.delta_mult(InWeights, deltaY);
//        deltaX.hadamard(layerX.foreach_n(SigmoidDerivative));
//    }
//
//    bool train(const Matrix<double>& input, const Matrix<double>& output, int times = 1, double nor = 1)
//    {
//        Matrix<double>& InputLayer = Layers[0];
//        InputLayer = input;
//        InputLayer.normalize1(nor);
//        Matrix<double>& LayerN = Layers[Layers.size() - 1];
//        Matrix<double>& DeltasN = Deltas[Deltas.size() - 1];
//        const size_t LayerCount = Layers.size();
//        for (int t = 0; t < times; ++t)
//        {
//            // 正向传播
//            for (size_t l = 0; l < Layers.size()-1; ++l)
//            {
//                forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
//            }
//
//
//
//
//            // 判断误差
//            LastAberration = Aberration.subtract(output, LayerN).squariance() / 2;
//
//            if (LastAberration < AberrationThreshold) break;
//
//            // 反向修正
//            DeltasN.hadamard(-Aberration, LayerN.foreach_n(SigmoidDerivative));
//
//            for (size_t l = 0; l < Layers.size() - 1; ++l)
//            {
//                backward(Layers[LayerCount - l - 2], 
//                    Weights[LayerCount - l - 2], 
//                    Bias[LayerCount - l - 2], 
//                    Deltas[LayerCount - l - 2], 
//                    Deltas[LayerCount - l - 1]);
//            }
//        }
//        return true;
//    }
//
//    double L2_Penalty()
//    {
//        double Result = 0.0;
//        for (auto& w :Weights)
//        {
//            Result += w.squariance();
//        }
//        return Attenuate * Result / 2;
//    }
//
//
//    double simulate(const Matrix<double>& input, Matrix<double>& output, const Matrix<double>& expect, double nor = 1)
//    {
//        Matrix<double>& InputLayer = Layers[0];
//        InputLayer = input;
//        InputLayer.normalize1(nor);
//
//        // 正向传播
//        for (size_t l = 0; l < Layers.size() - 1; ++l)
//        {
//            forward(Layers[l], Layers[l + 1], Weights[l], Bias[l]);
//        }
//        output = Layers[Layers.size() - 1];
//        // 判断误差
//        LastAberration = Aberration.subtract(expect, output).squariance() / 2;// / output.col();
//        return LastAberration;
//    }
//
//    std::string to_string()
//    {
//        std::string str = "Layers:{\n";
//        for (auto& val : Layers)
//        {
//            str += val.to_string();
//            str += "\n";
//            str += "------------------------------------------------------------------------\n";
//            str += "\n";
//        }
//        str += "}\nWeights:{\n";
//        for (auto& val : Weights)
//        {
//            str += val.to_string();
//            str += "\n";
//            str += "------------------------------------------------------------------------\n";
//            str += "\n";
//        }
//        str += "}\nBias:{\n";
//        for (auto& val : Bias)
//        {
//            str += val.to_string();
//            str += "\n";
//            str += "------------------------------------------------------------------------\n";
//            str += "\n";
//        }
//        str += "}\n";
//
//        str += "LastAberration:";
//        str += std::to_string(LastAberration);
//
//        return str;
//    }
//
//private:
//    std::vector<Matrix<double>> Layers;
//    std::vector<Matrix<double>> Weights;
//    std::vector<Matrix<double>> Bias;
//    std::vector<Matrix<double>> Deltas;
//    Matrix<double>              Aberration;
//    double                      AberrationThreshold;
//    double                      LastAberration;
//    double                      LearnRate;
//    double                      Attenuate;
//};

