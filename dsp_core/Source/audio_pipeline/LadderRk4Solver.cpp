#include "LadderRk4Solver.h"
#include "../SeamlessTransferFunction.h"
#include <algorithm>

namespace dsp_core::audio_pipeline {

namespace {
constexpr double kSixth = 1.0 / 6.0;

inline double clampAbs(double v, double lim) noexcept {
    if (v < -lim) {
        return -lim;
    }
    if (v > lim) {
        return lim;
    }
    return v;
}
} // namespace

void LadderRk4Solver::prepare(double sampleRate, int numChannels) {
    sampleRate_ = sampleRate;
    T_ = 1.0 / sampleRate;
    const int ch = std::max(1, numChannels);
    channels_.assign(static_cast<std::size_t>(ch), ChannelState{});
}

void LadderRk4Solver::reset() {
    for (auto& c : channels_) {
        c = ChannelState{};
    }
}

void LadderRk4Solver::pokeStateForTest(int ch, double value) {
    if (ch < 0 || ch >= static_cast<int>(channels_.size())) {
        return;
    }
    auto& st = channels_[static_cast<std::size_t>(ch)];
    for (auto& v : st.s) {
        v = value;
    }
    st.Vn_1 = value;
    st.yFbPrev = value;
}

// One RK4 derivative evaluation of the whole cascade.
//
// Generalizes VirtualAnalogFilterStage::cascade. Walk the chain once, tracking
// each stage's input x_i, its derivative contribution d_i = T*g*Phi_i(x_i - s_i),
// and its peeked output o_i (LP -> s_i+d_i; HP -> x_i - (s_i+d_i)). o_i feeds the
// next stage's input. For an all-LP cascade this reduces to the prototype's
// `x_{i+1} = s[i-1] + d[i-1]` peek-ahead exactly.
//
// yFb is the feedback tap value chosen by the caller (s_M for LP-final,
// yFbPrev for HP-final) so this eval never closes a delay-free algebraic loop.
inline std::array<double, LadderRk4Solver::kMaxStages>
LadderRk4Solver::cascade(double V, const std::array<double, kMaxStages>& s, double gVal, double R,
                         double yFb, const LadderTopologySnapshot& topo,
                         const std::array<const SeamlessTransferFunction*, 5>& nls,
                         int ch) const noexcept {
    const int M = topo.numStages;

    // Feedback nonlinearity Phi_fb(R * y_fb) (slot 0), clamped for loop safety.
    // Reduces to the prototype's tanh(R * s[M-1]) when the FB curve is tanh and
    // the final stage is LP (then yFb == s[M-1]).
    const double fb = clampAbs(nls[0]->applyTransferFunction(R * yFb, ch), kFbOutMax);

    std::array<double, kMaxStages> d{};
    double xi = V - fb; // input to stage 1 (index 0)

    for (int i = 0; i < kMaxStages; ++i) {
        const auto u = static_cast<std::size_t>(i);
        if (i < M) {
            const double zi = xi - s[u]; // (x_i - s_i)
            d[u] = T_ * gVal * nls[static_cast<std::size_t>(i + 1)]->applyTransferFunction(zi, ch);

            // Peeked output feeds the next stage's input.
            const double peekedState = s[u] + d[u];
            xi = (topo.modes[u] == Mode::Lowpass) ? peekedState : (xi - peekedState);
        } else {
            // Unused upper stages mirror the last active stage's derivative
            // (matches the prototype's short-circuit — no extra LUT lookups).
            d[u] = d[static_cast<std::size_t>(M - 1)];
        }
    }

    return d;
}

double LadderRk4Solver::processSample(double x, int ch, double cutoffRad, double R,
                                      const LadderTopologySnapshot& topo,
                                      const std::array<const SeamlessTransferFunction*, 5>& nls) {
    if (ch < 0 || ch >= static_cast<int>(channels_.size())) {
        return x;
    }
    auto& state = channels_[static_cast<std::size_t>(ch)];

    const int M = topo.numStages;
    const auto uM = static_cast<std::size_t>(M - 1);
    const bool finalIsHp = topo.modes[uM] == Mode::Highpass;

    const double Vn = x;
    const double g = cutoffRad;
    const double g_mid = 0.5 * (g + state.g_n1);
    const double Vn_mid = 0.5 * (Vn + state.Vn_1);

    // Feedback tap: LP-final taps the (live, per-sub-step) last LP state — exactly
    // like the prototype which reads s[M-1] inside each cascade(). HP-final taps
    // the last committed output (yFbPrev), held fixed across the 4 sub-steps, to
    // break the delay-free loop (a future ZDF solver resolves this exactly).
    const double yFbK1 = finalIsHp ? state.yFbPrev : state.s[uM];
    const auto k1 = cascade(state.Vn_1, state.s, state.g_n1, R, yFbK1, topo, nls, ch);

    std::array<double, kMaxStages> s2{};
    for (int j = 0; j < kMaxStages; ++j) {
        s2[static_cast<std::size_t>(j)] =
            state.s[static_cast<std::size_t>(j)] + 0.5 * k1[static_cast<std::size_t>(j)];
    }
    const double yFbK2 = finalIsHp ? state.yFbPrev : s2[uM];
    const auto k2 = cascade(Vn_mid, s2, g_mid, R, yFbK2, topo, nls, ch);

    std::array<double, kMaxStages> s3{};
    for (int j = 0; j < kMaxStages; ++j) {
        s3[static_cast<std::size_t>(j)] =
            state.s[static_cast<std::size_t>(j)] + 0.5 * k2[static_cast<std::size_t>(j)];
    }
    const double yFbK3 = finalIsHp ? state.yFbPrev : s3[uM];
    const auto k3 = cascade(Vn_mid, s3, g_mid, R, yFbK3, topo, nls, ch);

    std::array<double, kMaxStages> s4{};
    for (int j = 0; j < kMaxStages; ++j) {
        s4[static_cast<std::size_t>(j)] =
            state.s[static_cast<std::size_t>(j)] + k3[static_cast<std::size_t>(j)];
    }
    const double yFbK4 = finalIsHp ? state.yFbPrev : s4[uM];
    const auto k4 = cascade(Vn, s4, g, R, yFbK4, topo, nls, ch);

    // Integrate + commit. Unused upper slots mirror the last active state so
    // s[M-1] stays the feedback source (prototype-compatible).
    for (int j = 0; j < kMaxStages; ++j) {
        const auto u = static_cast<std::size_t>(j);
        if (j < M) {
            state.s[u] += (k1[u] + 2.0 * k2[u] + 2.0 * k3[u] + k4[u]) * kSixth;
            state.s[u] = clampAbs(state.s[u], kStateMax);
        } else {
            state.s[u] = state.s[uM];
        }
    }

    state.Vn_1 = Vn;
    state.g_n1 = g;

    // Output o_M. For an all-LP cascade this is exactly s[M-1] (bit-compatible
    // with the prototype's `y[kMaxStages-1]`). If any active stage is HP we walk
    // the committed states forward from the input node. The input node uses the
    // same feedback as the final derivative sub-step (yFbK4) so the tap is
    // consistent with the integrator that produced these states.
    bool pureLp = true;
    for (int i = 0; i < M; ++i) {
        if (topo.modes[static_cast<std::size_t>(i)] != Mode::Lowpass) {
            pureLp = false;
            break;
        }
    }

    double oM = state.s[uM];
    if (!pureLp) {
        double xi = Vn - clampAbs(nls[0]->applyTransferFunction(R * yFbK4, ch), kFbOutMax);
        for (int i = 0; i < M; ++i) {
            const auto u = static_cast<std::size_t>(i);
            const double si = state.s[u];
            const double oi = (topo.modes[u] == Mode::Lowpass) ? si : (xi - si);
            xi = oi;
            if (i == M - 1) {
                oM = oi;
            }
        }
    }

    state.yFbPrev = oM;
    return oM;
}

} // namespace dsp_core::audio_pipeline
