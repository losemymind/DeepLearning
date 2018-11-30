/****************************************************************************
  Copyright (c) 2018 libo All rights reserved.
  
  losemymind.libo@gmail.com

****************************************************************************/
#ifndef _FUNCTIONAL_HPP
#define _FUNCTIONAL_HPP

#pragma once
#include <vector>

inline double ReLU(double input)
{
    return input > 0 ? input : 0;
}

inline double ReLUDerivative(double input)
{
    return input > 0 ? 1 : 0;
}

inline double Sigmoid(double input)
{
    return (1.0 / (1.0 + std::exp(-input)));
}


// f(x)' = f(x)(1 ? f(x))
inline double SigmoidDerivative(double input)
{
    //double val = Sigmoid(input);
    double val = input;
    return val * (1.0 - val);
}

// f(x) = tanh(x)=(exp(x)-exp( ? x))/(exp(x)+exp( ? x))
inline double Tanh(double input)
{
    return (2 * Sigmoid(2 * input) - 1);
}

// f(x)' = 1 ? (f(x))2
inline double TanhDerivative(double input)
{
    double val = Tanh(input);
    return (1.0 - val * val);
}


template<typename T>
inline double NormL1(const std::vector<T>& arr)
{
    double ret = 0.0;
    for (auto& val : arr)
    {
        if (val != 0.0)
        {
            ret += std::abs(val);
        }
    }
    return ret;
}

template<typename T>
inline double NormL1(const std::vector<T>& left, const std::vector<T>& right)
{
    double ret = 0.0;
    if (left.size() != right.size())
    {
        return ret;
    }
    for (size_t i = 0; i < left.size(); ++i)
    {
        T val = left[i] - right[i];
        if (val != 0.0)
        {
            ret += std::abs(val);
        }
    }
    return ret;
}

template<typename T>
inline double NormL2(const std::vector<T>& arr)
{
    double ret = 0.0;
    for (auto& val : arr)
    {
        ret += val * val;
    }
    return std::sqrt(ret);
}

template<typename T>
inline double NormL2(const std::vector<T>& left, const std::vector<T>& right)
{
    double ret = 0.0;
    if (left.size() != right.size())
    {
        return ret;
    }
    for (size_t i = 0; i < left.size(); ++i)
    {
        T val = left[i] - right[i];
        ret += val * val;
    }
    return std::sqrt(ret);
}


template<typename T>
void Softmax(const T* src, T* dst, int length)
{
    const T alpha = *std::max_element(src, src + length);
    T denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = std::exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
}

double RandomN(double min, double max)
{
    double len = (max - min) / (double)RAND_MAX;
    return min + (double)rand() * len;
}

double Normalize1(double val, double max = 0)
{
    if (max == 0)
    {
        if (std::abs(val) > max) 
            max = std::abs(val);
    }
    return (val / max);
}

#endif // END OF _FUNCTIONAL_HPP
