/*******************************************************************************
 The block below describes the properties of this module, and is read by
 the Projucer to automatically generate project code that uses it.
 For details about the syntax and how to create or use a module, see the
 JUCE Module Format.md file.


 BEGIN_JUCE_MODULE_DECLARATION

  ID:                 dsp_core
  vendor:             tbd
  version:            1.0.0
  name:               DSP Core
  description:        A module for creating and editing transfer functions in audio applications.
  website:            tbd
  license:            proprietary/commercial
  minimumCppStandard: 17

  dependencies:       juce_core, juce_data_structures, juce_audio_processors, juce_dsp, juce_audio_formats

 END_JUCE_MODULE_DECLARATION

*******************************************************************************/

#pragma once
#define DSP_CORE_H_INCLUDED

#include <juce_data_structures/juce_data_structures.h>
#include <juce_dsp/juce_dsp.h>

#include "Source/AudioHistoryBuffer.h"
#include "Source/StereoHistoryBuffer.h"
#include "Source/HarmonicLayer.h"
#include "Source/Lane.h"
#include "Source/LaneMixer.h"
#include "Source/LayeredTransferFunction.h"
#include "Source/SeamlessTransferFunction.h"
#include "Source/SeamlessTransferFunctionImpl.h"
#include "Source/MorphGesture.h"
#include "Source/SplineLayer.h"
#include "Source/SplineTypes.h"
#include "Source/ExpressionEvaluator.h"
#include "Source/TransferFunction.h"
#include "Source/Services/AdaptiveToleranceCalculator.h"
#include "Source/Services/CoordinateSnapper.h"
#include "Source/Services/CurveFeatureDetector.h"
#include "Source/Services/GestureSmoother.h"
#include "Source/Services/SplineFitter.h"
#include "Source/Services/SplineEvaluator.h"
#include "Source/Services/SymmetryAnalyzer.h"
#include "Source/Services/TransferFunctionOperations.h"
#include "Source/Services/PerlinNoiseService.h"
#include "Source/Services/ExpressionParser.h"
#include "Source/Services/EquationLaneSynthesizer.h"
#include "Source/Services/AnchorConstraintService.h"
#include "Source/Services/SplineLaneSynthesizer.h"
#include "Source/HysteresisProcessor.h"
#include "Source/audio_pipeline/AudioProcessingStage.h"
#include "Source/audio_pipeline/AudioPipeline.h"
#include "Source/audio_pipeline/GainStage.h"
#include "Source/audio_pipeline/DryWetMixStage.h"
#include "Source/audio_pipeline/WaveshapingStage.h"
#include "Source/audio_pipeline/OversamplingWrapper.h"
#include "Source/audio_pipeline/DCBlockingFilter.h"
#include "Source/audio_pipeline/LowpassStage.h"
#include "Source/audio_pipeline/Tanh2xLUT.h"
#include "Source/audio_pipeline/LadderTPTStage.h"
#include "Source/audio_pipeline/LadderHighpassTPTStage.h"
#include "Source/audio_pipeline/FilterStage.h"
#include "Source/audio_pipeline/VirtualAnalogFilterStage.h"
#include "Source/audio_pipeline/LadderSolver.h"
#include "Source/audio_pipeline/LadderRk4Solver.h"
#include "Source/audio_pipeline/NonlinearLadderStage.h"
#include "Source/audio_pipeline/LowShelfStage.h"
#include "Source/audio_pipeline/FirstOrderShelfStage.h"
#include "Source/audio_pipeline/FatStage.h"
#include "Source/audio_pipeline/ToneFilterStrategy.h"
#include "Source/audio_pipeline/LowpassStrategy.h"
#include "Source/audio_pipeline/HighpassStrategy.h"
#include "Source/audio_pipeline/LowShelfStrategy.h"
#include "Source/audio_pipeline/HysteresisStrategy.h"
#include "Source/audio_pipeline/ImpulseResponseStrategy.h"
#include "Source/audio_pipeline/SmileStrategy.h"
#include "Source/audio_pipeline/HighShelfStage.h"
#include "Source/audio_pipeline/HighShelfStrategy.h"
#include "Source/audio_pipeline/BellStage.h"
#include "Source/audio_pipeline/BellStrategy.h"
#include "Source/audio_pipeline/SoftClippingStage.h"
#include "Source/audio_pipeline/SurgeStage.h"
#include "Source/audio_pipeline/BiasStage.h"
#include "Source/audio_pipeline/HysteresisStage.h"
#include "Source/audio_pipeline/ToneStage.h"
#include "Source/audio_pipeline/ToneFrequencyResponse.h"
#include "Source/audio_pipeline/AudioInputBuffer.h"
#include "Source/audio_pipeline/AudioInputWriter.h"
#include "Source/audio_pipeline/SignalRangeTap.h"
#include "Source/audio_pipeline/AutoGainState.h"
#include "Source/audio_pipeline/AutoSquashStage.h"
#include "Source/audio_pipeline/AutoRestoreStage.h"
#include "Source/audio_pipeline/SlotVisualizerPublisher.h"
#include "Source/audio_pipeline/EnvelopeFollowerStage.h"
#include "Source/audio_pipeline/LfoStage.h"
#include "Source/audio_pipeline/ModulatorSlotStage.h"
#include "Source/audio_pipeline/StageHandles.h"
#include "Source/audio_pipeline/AudioPipelineBuilder.h"
