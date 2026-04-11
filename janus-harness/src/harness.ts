/**
 * @component janus-harness
 * @stage planning
 * @priority P2
 */
/**
 * Janus Harness — Stateful Control Kernel
 *
 * WHAT THE AGENT DOES:
 *   Estimates semantic signals: delta_s, j_harm, j_drift, j_anchor
 *   (This is the ONLY thing that needs an LLM / reasoning)
 *
 * WHAT THIS HARNESS DOES (zero tokens, pure math):
 *   - Zone classification
 *   - Lambda trajectory pattern
 *   - Coupler W_c dynamics
 *   - Progress tracking
 *   - Resonance energy
 *   - All 5 DT guards
 *   - Bridge logic
 *   - Attention blending
 *   - Action recommendation (continue / slow / pause / rollback)
 *
 * USAGE:
 *   const harness = new JanusHarness();
 *   const state = harness.step({ delta_s: 0.42, j_harm: 0.05, j_drift: 0.18, j_anchor: 0.23 });
 *   // state.zone === "transit"
 *   // state.action === "continue"
 *   // ... all formulas computed, zero cost
 */

import type {
  SemanticSignals,
  CognitiveState,
  HarnessConfig,
  StepRecord,
  ZoneActionEntry,
} from "./types.js";

import {
  resolveConfig,
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
} from "./formulas.js";

export class JanusHarness {
  private cfg: Required<HarnessConfig>;
  private t: number = 0;
  private history: StepRecord[] = [];
  private alt: 1 | -1 = 1;
  private chi_prev: number = 0;

  constructor(config?: HarnessConfig) {
    this.cfg = resolveConfig(config);
  }

  /**
   * Process one step of semantic signals.
   * Returns the complete cognitive state — all formulas computed.
   *
   * Call this at each pause point / reasoning checkpoint.
   * The orchestrator / oracle / embedded model provides the signals.
   * Everything else is free.
   */
  step(signals: SemanticSignals): CognitiveState {
    this.t += 1;
    const t = this.t;

    const prev = this.history.length > 0 ? this.history[this.history.length - 1] : null;
    const delta_s = signals.delta_s;
    const delta_s_prev = prev?.delta_s ?? null;

    // ─── §3: Zone ───
    const zone = getZone(delta_s, this.cfg);

    // ─── §4: Residual & Resonance ───
    const R_t = Math.abs(computeResidual(
      delta_s, this.cfg,
      signals.delta_C_est ?? 0,
      signals.delta_V_est ?? 0,
    ));
    const R_history = this.history.map((h) => h.R_t);
    const E_resonance = computeResonance(R_t, R_history, this.cfg);
    const E_res_prev = prev?.E_resonance ?? null;

    // ─── §5: Progress & Coupler ───
    const P_t = computeProgress(delta_s, delta_s_prev, this.cfg);

    const anchor_score = signals.anchor_score ?? 1.0;
    const anchor_score_prev = prev?.anchor_score ?? null;
    this.alt = updateAlternation(this.alt, anchor_score, anchor_score_prev, this.cfg);

    const W_c = computeCoupler(delta_s, P_t, this.alt, this.cfg);

    // ─── §6: Lambda Pattern ───
    const lambda_observe = classifyLambda(
      delta_s, delta_s_prev,
      E_resonance, E_res_prev,
      signals.anchor_conflict ?? false,
    );

    // ─── §6 supplement: E_bar ───
    const delta_s_history = this.history.map((h) => h.delta_s);
    const E_bar = computeEBar(delta_s_history, delta_s, this.cfg);

    // ─── §7: Bridge Logic ───
    const bridge = canBridge(delta_s, delta_s_prev, W_c, this.cfg);

    // ─── §8: Attention Blend ───
    const alpha_blend = computeAlphaBlend(W_c, this.cfg);

    // ─── §9: J_t ───
    const J_t = computeJt(signals.j_harm, signals.j_drift, signals.j_anchor);

    // ─── DT Guards ───
    // Compute raw progress for guard threshold (delta between steps)
    const raw_progress = delta_s_prev !== null ? delta_s_prev - delta_s : this.cfg.zeta_min;

    // Build WRP history from past (zone, action) pairs before this step
    const wrp_history: ZoneActionEntry[] = this.history.map((h) => ({
      zone: h.zone,
      action: h.action,
    }));

    const { guards, action } = executeGuards(
      {
        anchor_score,
        delta_s_t: delta_s,
        delta_s_prev,
        E_res_t: E_resonance,
        E_res_prev,
        redundancy: signals.redundancy ?? 0,
        quality: signals.quality ?? 1.0,
        progress: raw_progress,
        lambda: lambda_observe,
        W_c,
        d_path: signals.d_path ?? 0,
        contradiction: signals.contradiction ?? false,
        chi_prev: this.chi_prev,
        t,
        wrp_history,
      },
      this.cfg,
    );

    // Update state for next step
    this.chi_prev = guards.wtf.chi;
    this.history.push({
      t,
      delta_s,
      R_t,
      E_resonance,
      anchor_score,
      chi: guards.wtf.chi,
      zone,
      action,
    });

    // Keep rolling window bounded
    if (this.history.length > this.cfg.window * 2) {
      this.history = this.history.slice(-this.cfg.window);
    }

    return {
      t,
      delta_s,
      zone,
      lambda_observe,
      W_c,
      J_t,
      E_resonance,
      P_t,
      can_bridge: bridge,
      alpha_blend,
      guards,
      action,
      alt: this.alt,
      E_bar,
    };
  }

  /**
   * Emit a YAML cognitive header from the current state.
   * This is what J4NUS reads from the stream.
   */
  static toYAML(state: CognitiveState): string {
    const lambda_symbols: Record<string, string> = {
      convergent: "->",
      recursive: "<>",
      divergent: "<-",
      chaotic: "x",
    };
    return [
      "cognitive_state:",
      `  delta_s: ${state.delta_s.toFixed(2)}`,
      `  zone: ${state.zone}`,
      `  lambda: "${lambda_symbols[state.lambda_observe]}"`,
      `  w_c: ${state.W_c.toFixed(2)}`,
      `  j_t: ${state.J_t.toFixed(2)}`,
      `  action: ${state.action}`,
      `  guards:`,
      `    collapse_risk: ${state.guards.collapse_risk.toFixed(2)}`,
      `    wtf: ${state.guards.wtf.triggered}`,
      `    wri: ${state.guards.wri.triggered}`,
      `  can_bridge: ${state.can_bridge}`,
      `  t: ${state.t}`,
    ].join("\n");
  }

  /**
   * Check if the current state warrants a pause point
   * (for the orchestrator to evaluate / mentor to intervene).
   */
  static shouldPause(state: CognitiveState): boolean {
    return (
      state.action === "pause" ||
      state.action === "rollback" ||
      state.zone === "danger" ||
      (state.zone === "risk" && state.lambda_observe === "chaotic") ||
      state.J_t > 0.4
    );
  }

  /**
   * Get the best rollback target from history (argmin delta_s over last window).
   */
  getBestRollbackTarget(): number | null {
    if (this.history.length < 2) return null;
    const window = this.history.slice(-this.cfg.window);
    let minDs = Infinity;
    let minT = 0;
    for (const rec of window) {
      if (rec.delta_s < minDs) {
        minDs = rec.delta_s;
        minT = rec.t;
      }
    }
    return minT;
  }

  /** Reset to initial state */
  reset(): void {
    this.t = 0;
    this.history = [];
    this.alt = 1;
    this.chi_prev = 0;
  }

  /** Get step count */
  get stepCount(): number {
    return this.t;
  }

  /** Get last delta_s for external comparisons */
  get lastDeltaS(): number | null {
    return this.history.length > 0
      ? this.history[this.history.length - 1].delta_s
      : null;
  }
}
