/**
 * @component janus-harness
 * @stage planning
 * @priority P2
 */
/**
 * Janus Harness — Deterministic Formula Engine
 *
 * Every formula from the ATL4S Control Kernel (god_formulas.md).
 * Pure functions. Zero tokens. Zero API calls.
 *
 * The LLM estimates semantic signals (delta_s, j_harm, j_drift, j_anchor).
 * This module computes EVERYTHING else.
 */

import type {
  Zone,
  LambdaPattern,
  RecommendedAction,
  GuardResult,
  HarnessConfig,
  ZoneActionEntry,
} from "./types.js";

import { detectRepeatPattern } from "./loop_guard.js";

// ─── DEFAULT CONSTANTS ───

const DEFAULTS: Required<HarnessConfig> = {
  w_e: 0.50,
  w_r: 0.30,
  w_c: 0.20,
  zone_safe: 0.40,
  zone_transit: 0.60,
  zone_risk: 0.85,
  B_c: 0.85,
  gamma: 0.618,
  theta_c: 0.75,
  zeta_min: 0.10,
  omega: 1.0,
  phi_delta: 0.15,
  epsilon: 0.0,
  k_bias: 0.01,
  lambda_C_bonus: 0.10,
  lambda_V_bonus: 0.10,
  k_c: 0.25,
  tau_wri: 0.60,
  rho_wai: 0.75,
  sigma_wai: 0.70,
  eta_prog: 0.03,
  h_anchor: 0.02,
  window: 5,
  max_turns: 50,
};

export function resolveConfig(partial?: HarnessConfig): Required<HarnessConfig> {
  return { ...DEFAULTS, ...partial };
}

// ─── UTILITY ───

export function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function rollingMean(values: number[], window: number): number {
  if (values.length === 0) return 0;
  const w = Math.min(values.length, window);
  const slice = values.slice(-w);
  return slice.reduce((a, b) => a + b, 0) / w;
}

// ─── §3: ZONE CLASSIFICATION ───

export function getZone(delta_s: number, cfg: Required<HarnessConfig>): Zone {
  if (delta_s < cfg.zone_safe) return "safe";
  if (delta_s < cfg.zone_transit) return "transit";
  if (delta_s < cfg.zone_risk) return "risk";
  return "danger";
}

// ─── §4: RESIDUAL & RESONANCE ───

export function computeResidual(
  delta_s: number,
  cfg: Required<HarnessConfig>,
  delta_C_est: number = 0,
  delta_V_est: number = 0,
): number {
  // B_t = (I_t - G + k_bias) - lambda_C_bonus * ΔC_est - lambda_V_bonus * ΔV_est
  // We use delta_s as the proxy for ||I_t - G|| since the LLM already computed it
  const base = delta_s + cfg.k_bias;
  return base - cfg.lambda_C_bonus * delta_C_est - cfg.lambda_V_bonus * delta_V_est;
}

export function computeResonance(
  R_t: number,
  history: number[],
  cfg: Required<HarnessConfig>,
): number {
  // E_resonance_t = rolling_mean(R_1 ... R_t, window = min(t, 5))
  const all = [...history, R_t];
  return rollingMean(all, cfg.window);
}

// ─── §5: COUPLER & PROGRESSION (W_c) ───

export function computeProgress(
  delta_s_t: number,
  delta_s_prev: number | null,
  cfg: Required<HarnessConfig>,
): number {
  // prog_t = max(zeta_min, Δs_{t-1} - Δs_t)
  // P_t = prog_t^omega
  if (delta_s_prev === null) return Math.pow(cfg.zeta_min, cfg.omega);
  const prog = Math.max(cfg.zeta_min, delta_s_prev - delta_s_t);
  return Math.pow(prog, cfg.omega);
}

export function updateAlternation(
  alt_prev: 1 | -1,
  anchor_score_t: number,
  anchor_score_prev: number | null,
  cfg: Required<HarnessConfig>,
): 1 | -1 {
  if (anchor_score_prev === null) return alt_prev;
  const diff = Math.abs(anchor_score_t - anchor_score_prev);
  if (diff >= cfg.h_anchor) {
    const sign_changed =
      (anchor_score_t > anchor_score_prev) !== (anchor_score_prev > 0.5);
    if (sign_changed) return (alt_prev === 1 ? -1 : 1) as 1 | -1;
  }
  return alt_prev;
}

export function computeCoupler(
  delta_s_t: number,
  P_t: number,
  alt: 1 | -1,
  cfg: Required<HarnessConfig>,
): number {
  // Phi_t = phi_delta * alt_t + epsilon
  const Phi = cfg.phi_delta * alt + cfg.epsilon;
  // W_c_t = clip(B_s * P_t + Phi_t, -theta_c, +theta_c)
  // B_s = delta_s_t (semantic tension as base signal)
  const W_c = delta_s_t * P_t + Phi;
  return clamp(W_c, -cfg.theta_c, cfg.theta_c);
}

// ─── §6: LAMBDA PATTERN CLASSIFICATION ───

export function classifyLambda(
  delta_s_t: number,
  delta_s_prev: number | null,
  E_res_t: number,
  E_res_prev: number | null,
  anchor_conflict: boolean = false,
): LambdaPattern {
  if (delta_s_prev === null) return "convergent"; // first step, optimistic default

  const Delta_t = delta_s_t - delta_s_prev;

  // §6.4: chaotic — Delta > +0.04 or anchor conflicts
  if (Delta_t > 0.04 || anchor_conflict) return "chaotic";

  // §6.1: convergent — Delta <= -0.02 and E_bar non-increasing
  if (Delta_t <= -0.02 && (E_res_prev === null || E_res_t <= E_res_prev)) {
    return "convergent";
  }

  // §6.2: recursive — |Delta| < 0.02
  if (Math.abs(Delta_t) < 0.02) return "recursive";

  // §6.3: divergent — everything else
  return "divergent";
}

// ─── §6 (supplement): E_bar — rolling mean of delta_s ───

export function computeEBar(
  delta_s_history: number[],
  delta_s_t: number,
  cfg: Required<HarnessConfig>,
): number {
  const all = [...delta_s_history, delta_s_t];
  return rollingMean(all, cfg.window);
}

// ─── §7: BRIDGE LOGIC ───

export function canBridge(
  delta_s_t: number,
  delta_s_prev: number | null,
  W_c_t: number,
  cfg: Required<HarnessConfig>,
): boolean {
  if (delta_s_prev === null) return false;
  return (delta_s_t < delta_s_prev) && (W_c_t < 0.5 * cfg.theta_c);
}

// ─── §8: ATTENTION BLENDING (BBAM Lite) ───

export function computeAlphaBlend(W_c_t: number, cfg: Required<HarnessConfig>): number {
  // alpha_blend_t = clip(0.50 + k_c * tanh(W_c_t), 0.35, 0.65)
  return clamp(0.50 + cfg.k_c * Math.tanh(W_c_t), 0.35, 0.65);
}

// ─── §9 supplement: J_t composite ───

export function computeJt(j_harm: number, j_drift: number, j_anchor: number): number {
  return (j_harm + j_drift + j_anchor) / 3;
}

// ─── DT GUARDS (from dt-guards.ts spec) ───

/** WRI: Anchor Retention — fires when drifting from anchors */
export function guardWRI(
  anchor_score: number,
  delta_s_t: number,
  delta_s_prev: number | null,
  E_res_t: number,
  E_res_prev: number | null,
  cfg: Required<HarnessConfig>,
): { triggered: boolean; loss: number } {
  const below_threshold = anchor_score < cfg.tau_wri;
  const drifting =
    delta_s_prev !== null &&
    E_res_prev !== null &&
    delta_s_t > delta_s_prev &&
    E_res_t > E_res_prev;
  const triggered = below_threshold || drifting;
  const loss = Math.max(0, cfg.tau_wri - anchor_score);
  return { triggered, loss };
}

/** WAI: Head Identity — fires when stuck (redundant + low quality) */
export function guardWAI(
  redundancy: number,
  quality: number,
  cfg: Required<HarnessConfig>,
): { triggered: boolean } {
  return { triggered: redundancy > cfg.rho_wai && quality < cfg.sigma_wai };
}

/** WAY: Entropy Pump — fires when progress stalls */
export function guardWAY(
  progress: number,
  lambda: LambdaPattern,
  W_c: number,
  cfg: Required<HarnessConfig>,
): { triggered: boolean; h_target: number } {
  const stalled = progress < cfg.eta_prog && lambda !== "chaotic";
  // H_target = clamp(3.5 + 0.80*(0.03 - prog)*(1 + 0.5*|W_c|), 2.5, 5.0)
  const h_target = stalled
    ? clamp(3.5 + 0.80 * (cfg.eta_prog - progress) * (1 + 0.5 * Math.abs(W_c)), 2.5, 5.0)
    : 3.5;
  return { triggered: stalled, h_target };
}

/** WDT: Path Guard — fires when reasoning path deviates too far */
export function guardWDT(
  d_path: number,
  W_c: number,
): { triggered: boolean } {
  // Threshold adapts based on coupler strength
  const threshold = 0.25 * (1 - 0.60 * sigmoid(Math.abs(W_c)));
  return { triggered: d_path > threshold };
}

/** WTL: Turn Limit — fires when max_turns threshold is exceeded */
export function guardWTL(
  t: number,
  cfg: Required<HarnessConfig>,
): { triggered: boolean } {
  return { triggered: t > cfg.max_turns };
}

/** WTF: Collapse Detection — accumulates bad signals, triggers rollback */
export function guardWTF(
  delta_s_t: number,
  delta_s_prev: number | null,
  E_res_t: number,
  E_res_prev: number | null,
  contradiction: boolean,
  chi_prev: number,
): { triggered: boolean; chi: number } {
  let chi = 0;
  if (delta_s_prev !== null && delta_s_t > delta_s_prev) chi += 1;
  if (E_res_prev !== null && E_res_t > E_res_prev) chi += 1;
  if (contradiction) chi += 1;
  // Fires when chi_t + chi_prev >= 3
  const triggered = (chi + chi_prev) >= 3;
  return { triggered, chi };
}

/** Execute all guards and determine recommended action */
export function executeGuards(
  signals: {
    anchor_score: number;
    delta_s_t: number;
    delta_s_prev: number | null;
    E_res_t: number;
    E_res_prev: number | null;
    redundancy: number;
    quality: number;
    progress: number;
    lambda: LambdaPattern;
    W_c: number;
    d_path: number;
    contradiction: boolean;
    chi_prev: number;
    t: number;
    wrp_history: ZoneActionEntry[];
  },
  cfg: Required<HarnessConfig>,
): { guards: GuardResult; action: RecommendedAction } {
  const wri = guardWRI(
    signals.anchor_score, signals.delta_s_t, signals.delta_s_prev,
    signals.E_res_t, signals.E_res_prev, cfg,
  );
  const wai = guardWAI(signals.redundancy, signals.quality, cfg);
  const way = guardWAY(signals.progress, signals.lambda, signals.W_c, cfg);
  const wdt = guardWDT(signals.d_path, signals.W_c);
  const wtf = guardWTF(
    signals.delta_s_t, signals.delta_s_prev,
    signals.E_res_t, signals.E_res_prev,
    signals.contradiction, signals.chi_prev,
  );
  const wtl = guardWTL(signals.t, cfg);
  const wrp = detectRepeatPattern(signals.wrp_history, cfg.window);

  const triggered_count = [wri.triggered, wai.triggered, way.triggered, wdt.triggered, wtf.triggered]
    .filter(Boolean).length;
  const collapse_risk = triggered_count / 5;

  // Action priority (highest to lowest): WTL > WRP > WTF > WDT/collapse > WAY/WRI > continue
  let action: RecommendedAction = "continue";
  if (way.triggered || wri.triggered) action = "slow";
  if (wdt.triggered || collapse_risk > 0.6) action = "pause";
  if (wtf.triggered) action = "rollback";
  if (wrp.triggered) action = "pause";
  if (wtl.triggered) action = "rollback";

  const turn_limit_exceeded = wtl.triggered;

  return {
    guards: { wri, wai, way, wdt, wtf, wtl, wrp, turn_limit_exceeded, collapse_risk },
    action,
  };
}
