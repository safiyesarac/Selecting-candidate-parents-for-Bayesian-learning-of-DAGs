#include "array.h"
#include "bits.h"
#include "types.h"

#include "simple_common.h"

#include <algorithm>

namespace aps {

	static bool debugAPS = false;
extern "C" void setAPSdebug(bool value) {
    debugAPS = value;
}


namespace {



template <typename T>
void printDAG(const std::vector<size_t>& dag) {
    std::cerr << "[DAG] Edges: ";
    for (size_t v = 0; v < dag.size(); ++v) {
        size_t pMask = dag[v];
        while (pMask) {
            size_t p = bottomOneBitIdx(pMask);
            pMask ^= (1ULL << p);
            std::cerr << p << "->" << v << " ";
        }
    }
    std::cerr << "\n";
}

template <typename T>
void printArray(const char* name, const Array<T>& arr, size_t limit=16) {
    std::cerr << "[DEBUG] " << name << " (size=" << arr.size() << "): ";
    for (size_t i = 0; i < arr.size() && i < limit; ++i) {
        std::cerr << (double)arr[i] << " ";
    }
    if (arr.size() > limit) {
        std::cerr << "...";
    }
    std::cerr << "\n";
}
template <typename T>
Array<T> alphaSum(const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<T> alpha(S1 << n);
	if (debugAPS) {
        std::cerr << "[DEBUG] alphaSum: n=" << n << "\n";
    }

	alpha[0] = getOne<T>();
	for(size_t mask = 1; mask < alpha.size(); ++mask) {
		T plusVal = getZero<T>();
		T minusVal = getZero<T>();

		size_t maskPopCount = popCount(mask);

		for(size_t sub = 0; sub != mask; sub = (sub - mask) & mask) {
			T term = alpha[sub];
			size_t left = mask ^ sub;
			while(left) {
				size_t v = bottomOneBitIdx(left);
				left ^= S1 << v;
				term *= z[v][collapseBit(sub, v)];
			}
			if((popCount(sub) ^ maskPopCount) & 1) {
				plusVal += term;
			} else {
				minusVal += term;
			}
		}

		alpha[mask] = nonnegativeSubtraction(plusVal, minusVal);
		 if (debugAPS && mask < 32) {  // limit printing to first 32 for brevity
            std::cerr << " alpha[" << mask << "] = (plus=" << (double)plusVal
                      << ", minus=" << (double)minusVal 
                      << ") => " << (double)alpha[mask] << "\n";
        }
	}

	return alpha;
}

template <typename T>
Array<T> betaSum(const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<T> beta(S1 << n);

	size_t full = beta.size() - S1;
	if (debugAPS) {
        std::cerr << "[DEBUG] betaSum: n=" << n << ", full=" << full << "\n";
    }


	beta[0] = getOne<T>();
	for(size_t mask = 1; mask < beta.size(); ++mask) {
		T plusVal = getZero<T>();
		T minusVal = getZero<T>();

		size_t comp = full ^ mask;
		size_t maskPopCount = popCount(mask);

		for(size_t sub = 0; sub != mask; sub = (sub - mask) & mask) {
			T term = beta[sub];
			size_t left = mask ^ sub;
			while(left) {
				size_t v = bottomOneBitIdx(left);
				left ^= S1 << v;
				term *= z[v][collapseBit(comp, v)];
			}
			if((popCount(sub) ^ maskPopCount) & 1) {
				plusVal += term;
			} else {
				minusVal += term;
			}
		}

		beta[mask] = nonnegativeSubtraction(plusVal, minusVal);
		 if (debugAPS && mask < 32) {
            std::cerr << " beta[" << mask << "] = (plus=" << (double)plusVal
                      << ", minus=" << (double)minusVal
                      << ") => " << (double)beta[mask] << "\n";
        }
	}

	return beta;
}

template <typename T>
Array<Array<T>> gammaSum(const Array<T>& beta, const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<Array<T>> gamma(n);

	size_t full = (S1 << n) - S1;
	  if (debugAPS) {
        std::cerr << "[DEBUG] gammaSum: n=" << n << ", full=" << full << "\n";
    }

	for(size_t v = 0; v < n; ++v) {
		gamma[v] = Array<T>(S1 << (n - 1));

		for(size_t mask = 0; mask < gamma[v].size(); ++mask) {
			size_t expMask = expandBit(mask, v);
			size_t dom = full ^ (S1 << v) ^ expMask;

			T plusVal = getZero<T>();
			T minusVal = getZero<T>();

			size_t sub = 0;
			do {
				T term = beta[dom ^ sub];
				
				size_t left = sub;
				while(left) {
					size_t x = bottomOneBitIdx(left);
					left ^= S1 << x;
					term *= z[x][collapseBit(expMask, x)];
				}

				if(popCount(sub) & 1) {
					minusVal += term;
				} else {
					plusVal += term;
				}

				sub = (sub - dom) & dom;
			} while(sub);

			gamma[v][mask] = nonnegativeSubtraction(plusVal, minusVal);
			if (debugAPS && v < 1 && mask < 16) { // printing for first v & small mask
                std::cerr << " gamma[" << v << "][" << mask << "] => (plus="
                          << (double)plusVal << ", minus=" << (double)minusVal
                          << ") => " << (double)gamma[v][mask] << "\n";
            }
		}
	}

	return gamma;
}

}

template<typename T>
void printInput(const Array<Array<T>>& w) {
    std::cout << "[DEBUG] Printing input array:" << std::endl;
    for (size_t i = 0; i < w.size(); ++i) {
        std::cout << "Row " << i << ": ";
        for (size_t j = 0; j < w[i].size(); ++j) {
            std::cout << static_cast<double>(w[i][j]) << " ";
        }
        std::cout << std::endl;
    }
}



template <typename T>
Array<Array<T>> modularAPS_simple(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	
	if(debugAPS) {
        std::cerr << "[DEBUG] Called modularAPS_simple with n = " << n << std::endl;
        printInput(w);  // Print the input array
    }

	if(test && n > 11) {
		return Array<Array<T>>();
	}

	for(size_t v = 0; v < n; ++v) {
		if(w[v].size() != (S1 << (n - 1))) {
			fail("Internal error: weight array with invalid size");
		}
	}

	Array<Array<T>> z(n);
	for(size_t v = 0; v < n; ++v) {
		z[v] = downZeta(w[v]);
	}

	Array<T> alpha = alphaSum(z);
	Array<T> beta = betaSum(z);
	Array<Array<T>> ret = gammaSum(beta, z);

	for(size_t v = 0; v < n; ++v) {
		for(size_t i = 0; i < ret[v].size(); ++i) {
			ret[v][i] *= alpha[expandBit(i, v)];
		}
		upZeta(ret[v]);
		for(size_t i = 0; i < ret[v].size(); ++i) {
			ret[v][i] *= w[v][i];
		}
	}
	if(debugAPS) {
        std::cerr << "[DEBUG] Summation completed; returning results.\n";
    }

	return ret;
}

template <typename T>
Array<Array<T>> modularAR_simple(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	if(test && n > 11) {
		return Array<Array<T>>();
	}

	for(size_t v = 0; v < n; ++v) {
		if(w[v].size() != (S1 << (n - 1))) {
			fail("Internal error: weight array with invalid size");
		}
	}

	Array<Array<T>> z(n);
	for(size_t v = 0; v < n; ++v) {
		z[v] = downZeta(w[v]);
		  if(debugAPS && v < 2) { // just show first 2 for brevity
            std::cerr << "[DEBUG] after downZeta w[" << v << "], z[" << v << "] sample:\n";
            printArray("z[v]", z[v], 8);
        }
	}

	Array<T> alpha = alphaSum(z);
	Array<T> beta = betaSum(z);
	Array<Array<T>> gamma = gammaSum(beta, z);

	Array<Array<T>> ret(n);
	for(size_t i = 0; i < n; ++i) {
		ret[i] = Array<T>(n);
		ret[i].fill(getZero<T>());
	}
	for(size_t j = 0; j < n; ++j) {
		for(size_t u = 0; u < gamma[j].size(); ++u) {
			size_t ue = expandBit(u, j);
			T val = gamma[j][u] * z[j][u] * alpha[ue];
			size_t left = ((S1 << n) - S1) ^ ue;
			while(left) {
				size_t i = bottomOneBitIdx(left);
				left ^= S1 << i;
				ret[i][j] += val;
			}
		}
	}

	return ret;
}

#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	template Array<Array<T>> modularAPS_simple(const Array<Array<T>>&, bool); \
	template Array<Array<T>> modularAR_simple(const Array<Array<T>>&, bool);
APS_FOR_EACH_NUMBER_TYPE

}
