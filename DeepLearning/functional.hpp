/****************************************************************************
  Copyright (c) 2018 libo All rights reserved.
  
  losemymind.libo@gmail.com

****************************************************************************/
#ifndef _FUNCTIONAL_HPP
#define _FUNCTIONAL_HPP

#pragma once
#include <vector>
#include <random>

namespace DL
{
    inline double identity(double val)
    {
        return val;
    }

    inline double identity_d(double val)
    {
        return 1.0;
    }

    inline double relu(double val)
    {
        return val >= 0 ? val : 0;
    }

    inline double relu_d(double val)
    {
        return val >= 0 ? 1 : 0;
    }

    inline double sigmoid(double val)
    {
        return (1.0 / (1.0 + std::exp(-val)));
    }

    // f(x)' = f(x)(1 ? f(x))
    inline double sigmoid_d(double val)
    {
        return val * (1.0 - val);
    }

    inline double sigmoid_de(double val)
    {
        val = sigmoid(val);
        return val * (1.0 - val);
    }

    // f(x) = tanh(x)=(exp(x)-exp( ? x))/(exp(x)+exp( ? x))
    inline double tanh(double val)
    {
        return (2 * sigmoid(2 * val) - 1);
    }

    // f(x)' = 1 ? (f(x))2
    inline double tanh_d(double val)
    {
        return (1.0 - val * val);
    }

    inline double tanh_de(double val)
    {
        val = tanh(val);
        return (1.0 - val * val);
    }

    template<typename T>
    void softmax(const T* src, T* dst, int length)
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


    static std::default_random_engine& random_engine()
    {
        static std::random_device          _randomDevice;
        static std::default_random_engine  _randomEngine(_randomDevice());
        return _randomEngine;
    }

    template< class _Ty >
    static inline _Ty random(_Ty a, _Ty b, typename std::enable_if< std::is_floating_point<_Ty>::value >::type* = 0)
    {
        std::uniform_real_distribution<_Ty> randomResult(a, b);
        return randomResult(random_engine());
    }

    /**
     * Get random integral number in a range [a, b]
     * @return An integer type, by default the type is int.
     */
    template< typename _Ty >
    static inline _Ty random(_Ty a, _Ty b, typename std::enable_if< std::is_integral<_Ty>::value >::type* = 0)
    {
        std::uniform_int_distribution<_Ty> randomResult(a, b);
        return randomResult(random_engine());
    }

    /**
     * @return A random floating-point number in the range [0, 1.0].
     */
    template< typename _Ty >
    static inline _Ty random(typename std::enable_if< std::is_floating_point<_Ty>::value >::type* = 0)
    {
        return random<_Ty>((_Ty)0, /*std::*/nextafter(1, DBL_MAX));
    }
}

#endif // END OF _FUNCTIONAL_HPP
