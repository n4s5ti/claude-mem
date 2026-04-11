/**
 * @component janus-harness
 * @stage planning
 * @priority P2
 */
/**
 * Janus Harness — The deterministic half of the ATL4S control kernel.
 *
 * Agent provides: delta_s, j_harm, j_drift, j_anchor (semantic, needs LLM)
 * Harness computes: zone, lambda, W_c, guards, action, everything else (free)
 */

export { JanusHarness } from "./harness.js";
export type {
  SemanticSignals,
  CognitiveState,
  Zone,
  LambdaPattern,
  RecommendedAction,
  GuardResult,
  HarnessConfig,
  StepRecord,
} from "./types.js";
export {
  getZone,
  computeResidual,
  computeResonance,
  computeProgress,
  updateAlternation,
  computeCoupler,
  classifyLambda,
  computeEBar,
  canBridge,
  computeAlphaBlend,
  computeJt,
  executeGuards,
  guardWRI,
  guardWAI,
  guardWAY,
  guardWDT,
  guardWTF,
  clamp,
  resolveConfig,
} from "./formulas.js";
