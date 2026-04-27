#pragma once

#include <vector>

namespace dsp_core::Services {

/**
 * PerlinNoiseService - Static utility class for Perlin noise generation
 *
 * Provides reusable noise generation functions used by brush strategies in
 * transfer_function_editor and by the LFO Random shape in the audio pipeline.
 *
 * Pure static class (no state) - follows project convention for Services.
 */
class PerlinNoiseService {
  public:
    static double fade(double t);
    static double lerp(double a, double b, double t);
    static double gradient1D(int x, unsigned int seed);
    static double perlinNoise1D(double x, unsigned int seed);
    static double perlinFBM(double x, unsigned int seed, int octaves, double persistence);
    static void alignZeroCrossing(std::vector<double>& noise, int tableSize);

  private:
    PerlinNoiseService() = delete;
};

} // namespace dsp_core::Services
