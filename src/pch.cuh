#pragma once

#include <iostream>
#include<iomanip>
#include <cassert>
#include <vector>
#include <fstream>
#include "Utils.h"

template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr auto CeilToMultipleOf(T intValue, T multiple) noexcept
{
    return ((intValue + multiple - 1) / multiple) * multiple;
}

template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr auto IntDivideCeil(T intValue, T factor) noexcept
{
    return (intValue + factor - 1) / factor;
}

using uint = uint32_t;

#ifdef _DEBUG
#define SUCC(expr) assert((expr) == cudaSuccess);
#else
#define SUCC(expr) expr;
#endif
