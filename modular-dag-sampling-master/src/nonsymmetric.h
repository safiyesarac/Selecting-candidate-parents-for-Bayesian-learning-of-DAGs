#ifndef NONSYMMETRIC_H
#define NONSYMMETRIC_H

#include <vector>
#include <bitset>
#include "common.h"
#include "lognum.h"
#include <bitset>
#include "subtable.h" 

namespace nonsymmetric_ {
template <typename T>
std::vector<SubTable<T>> calculate_hat_weights(int size, const std::vector<std::vector<T>>& weights) {
    std::vector<SubTable<T>> hat_weights_2(size);

    for (int i = 0; i < size; ++i) {
        SubTable<T> vectB1(size);
        hat_weights_2[i] = vectB1;
    }

    for (int i = 0; i < size; ++i) {
        std::bitset<32> V((1 << size) - 1);
        V.set(i, 0);
        int V_sub_i = static_cast<int>(V.to_ulong());

        // Initialize hat_weights_2[i] with the initial weights
        hat_weights_2[i](0, 0) = weights[i][0];

        for (int t = V_sub_i; t != 0; t = (t - 1) & V_sub_i) {
            for (int R = t; R != 0; R = (R - 1) & t) {
                T sum = T(0);
                for (int k = 0; k < size; k++) {
                    if (((R >> k) & 1) == 1) {
                        sum = sum + hat_weights_2[i](R & ~(1 << k), t & ~(1 << k));
                    }
                }
                hat_weights_2[i](R, t) = sum * weights[i][__builtin_popcount(R)];
            }
        }
    }

    return hat_weights_2;
}


template <class T>
SubTable<T> monotone_calculate_fs(int size, const std::vector<SubTable<T>>& hws) {
    SubTable<T> fs(size);
    fs(0, 0) = T(1);  // Replace T::one() with T(1)

    std::bitset<32> V(((size_t)1 << size) - 1);
    int V_sub = (int)V.to_ulong();

    for (int U = 0; (U = (U - V_sub) & V_sub);) {
        for (int S_0 = 0; (S_0 = (S_0 - U) & U);) {
            int upmask = U & (~S_0);
            if (S_0 == U) {
                fs(S_0, U) = T(1);  // Replace T::one() with T(1)
            } else {
                T sum1 = T(0);
                for (int S_1 = 0; (S_1 = (S_1 - upmask) & upmask);) {
                    T product = T(1);  // Replace T::one() with T(1)
                    std::bitset<32> S_1_bits(S_1);

                    for (int i = 0; i < size; i++) {
                        if (S_1_bits[i] == 1) {
                            product = product * hws[i](S_0, V_sub & (~upmask));
                        }
                    }
                    product = product * fs(S_1, U & (~S_0));
                    sum1 = sum1 + product;
                }
                fs(S_0, U) = sum1;
            }
        }
    }

    return fs;
}

}  // namespace nonsymmetric_

#endif  // NONSYMMETRIC_H
