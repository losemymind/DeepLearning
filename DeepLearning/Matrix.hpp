/****************************************************************************
  Copyright (c) 2018 libo All rights reserved.

  losemymind.libo@gmail.com

****************************************************************************/
#ifndef FOUNDATIONKIT_MATRIX_HPP
#define FOUNDATIONKIT_MATRIX_HPP

#include <stdexcept>	// for runtime_error
#include <string>
#include <initializer_list>
#include "functional.hpp"

enum class ENormType
{
    NORM_MINMAX,
    NORM_L1,
    NORM_L2,
};

template<typename T>
class Matrix
{
public:

    typedef Matrix<T> _Myt;
    typedef T         value_type;

    ~Matrix()
    {
        deallocate();
    }

    Matrix()
        : Row(0)
        , Col(0)
        , Data(nullptr)
    {

    }
    Matrix(size_t InRow, size_t InCol)
        : Row(InRow)
        , Col(InCol)
        , Data(nullptr)
    {
        allocate();
    }

    Matrix(std::initializer_list<std::initializer_list<value_type>> _Ilist): Matrix()
    {
        assign(_Ilist);
    }

    Matrix(std::initializer_list<value_type> _Ilist) : Matrix()
    {
        assign({ _Ilist });
    }

    Matrix(const _Myt& InOther) : Matrix()
    {
        copy(InOther);
    }

    Matrix(_Myt&& InOther) : Matrix()
    {
        move(std::forward<_Myt&&>(InOther));
    }

    _Myt& operator = (const _Myt& InOther)
    {
        return copy(InOther);
    }

    _Myt& operator = (_Myt&& InOther)
    {
        return move(std::forward<_Myt&&>(InOther));
    }

    _Myt& operator=(std::initializer_list<std::initializer_list<value_type>> _Ilist)
    {	// assign initializer_list
        assign(_Ilist);
        return (*this);
    }

    _Myt& operator=(std::initializer_list<value_type> _Ilist)
    {	// assign initializer_list
        assign({ _Ilist });
        return (*this);
    }

    _Myt operator + (const _Myt& InOther)
    {
        _Myt Result(Row, Col);
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Result.Data[i][j] = Data[i][j] + InOther.Data[i][j];
            }
        }
        return Result;
    }

    _Myt& operator +=(const _Myt& InOther)
    {
        if (!equal_size(InOther))
        {
            return *this;
        }
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] += InOther.Data[i][j];
            }
        }
        return *this;
    }
    _Myt& operator - ()
    {
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = -Data[i][j];
            }
        }
        return *this;
    }
    _Myt operator - (const _Myt& InOther)
    {
        _Myt Result(Row, Col);
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = Data[i][j] - InOther.Data[i][j];
            }
        }
        return Result;
    }

    _Myt& operator -=(const _Myt& InOther)
    {
        _Myt Result(Row, Col);
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] -= InOther.Data[i][j];
            }
        }
        return Result;
        return *this;
    }

    _Myt operator*(const _Myt& InOther)
    {
        _Myt Result(Row, InOther.Col);
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                for (size_t k = 0; k < InOther.Row; ++k)
                {
                    Result.Data[i][j] += Data[i][k] * InOther.Data[k][j];
                }
            }
        }
        return Result;
    }

    _Myt& operator *=(const _Myt& InOther)
    {
        this->multiply(*this, InOther);
        return *this;
    }

    value_type get(size_t RowIndex, size_t ColIndex)
    {
        if (RowIndex < 0 || RowIndex > Row || ColIndex < 0 || ColIndex > Col)
        {
            throw std::runtime_error("Out of range.");
        }
        return Data[RowIndex][ColIndex];
    }

    void assign(std::initializer_list<std::initializer_list<value_type>> _Ilist)
    {
        Row = _Ilist.size();
        if (Row == 0)
        {
            return;
        }
        Col = _Ilist.begin()->size();
        if (Col == 0)
        {
            return;
        }
        allocate();
        int i = 0;
        for (auto r : _Ilist)
        {
            int j = 0;
            for (auto c : r)
            {
                Data[i][j++] = c;
            }
            ++i;
        }
    }

    bool empty()
    {
        return (Row == 0 || Col == 0);
    }

    size_t row()const
    {
        return Row;
    }

    size_t col()const
    {
        return Col;
    }

    template<typename F>
    _Myt& foreach(F InFunc)
    {

        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = InFunc(Data[i][j]);
            }
        }
        return *this;
    }

    template<typename F>
    _Myt& foreach_c(F InFunc)
    {

        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                InFunc(Data[i][j]);
            }
        }
        return *this;
    }

    template<typename F>
    _Myt foreach_n(F InFunc)
    {
        _Myt Ret(this->Row, this->Col);
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Ret.Data[i][j] = InFunc(Data[i][j]);
            }
        }
        return Ret;
    }

    bool equal_size(const _Myt& InOther)const
    {
        return (this->Row == InOther.Row && this->Col == InOther.Col);
    }


    _Myt negate()
    {
        _Myt Ret(this->Row, this->Col);
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Ret.Data[i][j] = -Data[i][j];
            }
        }
        return Ret;
    }

    _Myt& add(const _Myt& InLeft, const _Myt& InRight)
    {
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = InLeft.Data[i][j] + InRight.Data[i][j];
            }
        }
        return *this;
    }

    _Myt& subtract(const _Myt& InLeft, const _Myt& InRight)
    {
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = InLeft.Data[i][j] - InRight.Data[i][j];
            }
        }
        return *this;
    }

    _Myt& multiply(const _Myt& InLeft, const _Myt& InRight)
    {
        // left matrix column must == right row
        if (InLeft.col() != InRight.row())
            return *this;

        Row = InLeft.row();
        Col = InRight.col();
        deallocate();
        allocate();
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = 0;
                for (size_t k = 0; k < InRight.Row; ++k)
                {
                    Data[i][j] += InLeft.Data[i][k] * InRight.Data[k][j];
                }
            }
        }
        return *this;
    }

    _Myt& multiply(value_type InValue)
    {
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] *= InValue;
            }
        }
        return *this;
    }

    _Myt& hadamard(const _Myt& InOther)
    {
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] *= InOther.Data[i][j];
            }
        }
        return *this;
    }

    _Myt& hadamard(const _Myt& InLeft, const _Myt& InRight)
    {
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = InLeft.Data[i][j]* InRight.Data[i][j];
            }
        }
        return *this;
    }

    _Myt kronecker(const _Myt& InOther)
    {
        Matrix Ret(this->Row*InOther.Row, this->Col*InOther.Col);
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                for (size_t k = 0; k < InOther.Row; ++k)
                {
                    for (size_t l = 0; l < InOther.Col; ++l)
                    {
                        Ret.Data[i*InOther.Row + k][j*InOther.Col + l] = Data[i][j] * InOther.Data[k][l];
                    }
                }
            }
        }
        return Ret;
    }

    value_type squariance()
    {
        value_type ret = 0;
        foreach_c([&ret](auto& e)
        {
            ret += e * e;
        });
        return ret;
    }

    void normalize()
    {
        value_type ret = 0;
        foreach_c([&ret](auto& e) { ret += e * e; });
        if (ret == 0) return;
        foreach_c([&ret](auto& e) { e = e / ret; });
    }

    void normalize1(value_type max = 0)
    {
        if (max == 0)
        {
            foreach_c([&max](auto &e) { if (std::abs(e) > max) max = std::abs(e); });
            return;
        }
        foreach([max](auto& e) { return (e / max); });
    }

    void random(value_type min, value_type max)
    {
        foreach([min, max](auto& e)
        {
            return DL::random(min, max);
        });
    }

    _Myt& update_weights(const _Myt& LayerX, const _Myt& DeltaY, float InLearnRate)
    {
        // Weights.Row == LayerX.Col, Weights.Col == DeltaY.Col
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] -= InLearnRate * LayerX.Data[0][i] * DeltaY.Data[0][j];
            }
        }
        return *this;
    }

    _Myt& update_bias(const _Myt& DeltaY, float InLearnRate)
    {
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] -= InLearnRate * DeltaY.Data[i][j];
            }
        }
        return *this;
    }

    _Myt& deltas(const _Myt& InWeights, const _Myt& DeltaY)
    {
        // InWeights的行数表示 L 层神经元的个数， 列数表示 L+1 层神经元的个数
        // DeltaX和 DeltaY 的结构和神经网络层的一样，都是1行 N 列，列数表示神经元个数
        // DeltaX表示L层的残差，DeltaY表示 L+1层的残差。
        // L 层的残差等于 L 层与 L+1 层之间的权重 乘以 L+1 层的残差，
        // 也就是说所有层的残差的行数都是相等的，L 层的残差的列数（L层神经元个数）和 L 层与 L+1 层之间的权重的行数（L层神经元个数）
        // 是相等的， L+1 层的残差的 列数和 L 层与 L+1 层之间的权重的列数（L+1层神经元个数）是相等的。
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = 0;
                for (size_t k = 0; k < InWeights.col(); ++k)
                {
                    Data[i][j] += InWeights.Data[j][k] * DeltaY.Data[i][k];
                }
            }
        }
        return *this;
    }

    std::string to_string()const
    {
        std::string Result = "{";
        for (size_t i = 0; i < Row; ++i)
        {
            Result += "{";
            for (size_t j = 0; j < Col; ++j)
            {
                Result += std::to_string(Data[i][j]);
                if (j < Col -1)
                {
                    Result += ",";
                }
            }
            Result += "}";
        }
        Result += "}";
        return Result;
    }

private:
    void allocate()
    {
        Data = new T*[Row];
        for (size_t i = 0; i < Row; ++i)
        {
            Data[i] = new T[Col];
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = static_cast<value_type>(0);
            }
        }
    }
    void deallocate()
    {
        if (Data)
        {
            for (size_t i = 0; i < Row; ++i)
            {
                delete[] Data[i];
            }
            delete[] Data;
            Data = nullptr;
        }
    }

    _Myt& copy(const _Myt& InOther)
    {
        deallocate();
        Row = InOther.Row;
        Col = InOther.Col;
        allocate();
        for (size_t i = 0; i < Row; ++i)
        {
            for (size_t j = 0; j < Col; ++j)
            {
                Data[i][j] = InOther.Data[i][j];
            }
        }
        return *this;
    }

    _Myt& move(_Myt&& InOther)
    {
        deallocate();
        Row = InOther.Row;
        Col = InOther.Col;
        Data = InOther.Data;
        InOther.Data = nullptr;
        return *this;
    }

    size_t       Row;
    size_t       Col;
    value_type** Data;
};

#endif // END OF FOUNDATIONKIT_MATRIX_HPP