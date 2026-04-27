#pragma once
#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <vector>

namespace dsp_core {

/**
 * HarmonicLayer - Harmonic waveshaping evaluator with optimized computation
 *
 * Evaluates Chebyshev-based waveshaping functions using external coefficient data.
 * Uses precomputed basis functions for 10-50x performance improvement.
 *
 * Design: LayeredTransferFunction owns coefficients; HarmonicLayer is a pure evaluator.
 * This separates data ownership from computation logic.
 *
 * Expected coefficient layout (owned by caller):
 *   coefficients[0] = WT (wavetable mix) - controls base layer blending (unused by this class)
 *   coefficients[1..N] = harmonic amplitudes (used by evaluate())
 */
class HarmonicLayer {
  public:
    enum class Algorithm {
        Trig,      // Default: sin(n*asin(x)) for odd, cos(n*acos(x)) for even
        Polynomial // Alternative: Clenshaw's algorithm for Chebyshev polynomials
    };

    explicit HarmonicLayer(int numHarmonics = 40);

    // Configuration
    int getNumHarmonics() const {
        return numHarmonics;
    }

    // Evaluation (thread-safe read)

    /**
     * Evaluate harmonic contribution at normalized position x ∈ [-1, 1]
     *
     * @param x Normalized position in transfer function domain
     * @param coefficients External coefficient array (must have size >= numHarmonics + 1)
     *                     coefficients[0] is ignored (WT mix, used by caller)
     *                     coefficients[1..N] are harmonic amplitudes
     * @param tableSize Size of the transfer function table (for basis lookup)
     * @return Sum of weighted harmonics (does NOT include WT * base)
     */
    double evaluate(double x, const std::vector<double>& coefficients, int tableSize) const;

    // Precomputation (call once, or when table size changes)

    /**
     * Precompute basis functions for all harmonics across table
     *
     * Stores sin(n*asin(x)) and cos(n*acos(x)) for all x values,
     * eliminating trig calculations during slider movements.
     *
     * @param tableSize Number of points in transfer function table
     * @param minVal Minimum value of transfer function domain (typically -1.0)
     * @param maxVal Maximum value of transfer function domain (typically +1.0)
     */
    void precomputeBasisFunctions(int tableSize, double minVal, double maxVal);

    // Serialization
    juce::ValueTree toValueTree() const;
    void fromValueTree(const juce::ValueTree& vt);

    bool operator==(const HarmonicLayer& other) const;

  private:
    int numHarmonics;
    Algorithm algorithm = Algorithm::Trig;

    // Precomputed basis functions: [harmonic_index][table_index]
    mutable std::vector<std::vector<double>> basisFunctions;
    mutable bool precomputed = false;
    mutable int lastTableSize = 0;

    // Chebyshev evaluation functions
    static double evaluateChebyshevTrig(double x, const std::vector<double>& coeffs);
    static double evaluateChebyshevPolynomial(double x, const std::vector<double>& coeffs);

    static int xToTableIndex(double x, int tableSize, double minVal, double maxVal);
};

} // namespace dsp_core
