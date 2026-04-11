/**
 * @component janus-harness
 * @stage planning
 * @priority P2
 */
/**
 * Janus Harness — Type Definitions
 *
 * The harness splits the ATL4S control kernel into two halves:
 *   SEMANTIC (LLM-produced)  →  signals that require reasoning about meaning
 *   DETERMINISTIC (hooks)    →  everything else, pure math, zero tokens
 *
 * Agents estimate the semantic half. Hooks compute the rest.
 */

// ─── SEMANTIC SIGNALS (produced by LLM / oracle / embedded model) ───

export interface SemanticSignals {
  /** Semantic tension between current state and goal — [0, 1] */
  delta_s: number;
  /** Harm risk signal — [0, 1] */
  j_harm: number;
  /** Drift from goal signal — [0, 1] */
  j_drift: number;
  /** Anchor violation signal — [0, 1] */
  j_anchor: number;
  /** Overall anchor satisfaction score — [0, 1], optional */
  anchor_score?: number;
  /** Structural degradation estimate — [0, 1], optional */
  delta_C_est?: number;
  /** Value degradation estimate — [0, 1], optional */
  delta_V_est?: number;
  /** Whether anchor constraints are in conflict */
  anchor_conflict?: boolean;
  /** Redundancy score for WAI guard — [0, 1], optional */
  redundancy?: number;
  /** Quality score for WAI guard — [0, 1], optional */
  quality?: number;
  /** Path deviation distance for WDT guard — [0, 1], optional */
  d_path?: number;
  /** Contradiction detected for WTF guard */
  contradiction?: boolean;
}

// ─── DETERMINISTIC OUTPUT (computed by hooks, zero cost) ───

export type Zone = "safe" | "transit" | "risk" | "danger";
export type LambdaPattern = "convergent" | "recursive" | "divergent" | "chaotic";
export type RecommendedAction = "continue" | "slow" | "pause" | "rollback";

export interface GuardResult {
  /** WRI: Anchor retention guard */
  wri: { triggered: boolean; loss: number };
  /** WAI: Head identity / stuck detection */
  wai: { triggered: boolean };
  /** WAY: Entropy pump for stalled progress */
  way: { triggered: boolean; h_target: number };
  /** WDT: Path deviation guard */
  wdt: { triggered: boolean };
  /** WTF: Collapse detection — nuclear rollback */
  wtf: { triggered: boolean; chi: number };
  /** WTL: Turn limit exceeded guard */
  wtl: { triggered: boolean };
  /** WRP: Deterministic loop detection — repeat-pattern guard */
  wrp: LoopGuardResult;
  /** Whether the configured max_turns limit has been reached */
  turn_limit_exceeded: boolean;
  /** Collapse risk: proportion of triggered guards [0, 1] */
  collapse_risk: number;
}

export interface CognitiveState {
  /** Step number */
  t: number;
  /** Semantic tension (passthrough from LLM) */
  delta_s: number;
  /** Zone classification */
  zone: Zone;
  /** Trajectory pattern */
  lambda_observe: LambdaPattern;
  /** Coupler strength [-theta_c, +theta_c] */
  W_c: number;
  /** Deception/drift composite signal [0, 1] */
  J_t: number;
  /** Resonance energy (rolling mean of residual norms) */
  E_resonance: number;
  /** Progress metric */
  P_t: number;
  /** Whether agent is allowed to bridge (major plan change) */
  can_bridge: boolean;
  /** Attention blend factor [0.35, 0.65] */
  alpha_blend: number;
  /** DT Guard results */
  guards: GuardResult;
  /** Recommended action from guards */
  action: RecommendedAction;
  /** Alternation flag state */
  alt: 1 | -1;
  /** Rolling mean of delta_s */
  E_bar: number;
}

// ─── CONFIGURATION ───

export interface HarnessConfig {
  // Semantic tension weights
  w_e?: number; // default 0.50
  w_r?: number; // default 0.30
  w_c?: number; // default 0.20

  // Zone thresholds
  zone_safe?: number;    // default 0.40
  zone_transit?: number; // default 0.60
  zone_risk?: number;    // default 0.85

  // Coupler parameters
  B_c?: number;        // default 0.85
  gamma?: number;      // default 0.618 (golden ratio)
  theta_c?: number;    // default 0.75
  zeta_min?: number;   // default 0.10
  omega?: number;      // default 1.0
  phi_delta?: number;  // default 0.15
  epsilon?: number;    // default 0.0

  // Residual parameters
  k_bias?: number;           // default 0.01
  lambda_C_bonus?: number;   // default 0.10
  lambda_V_bonus?: number;   // default 0.10

  // Attention blending
  k_c?: number; // default 0.25

  // Guard thresholds
  tau_wri?: number;    // default 0.60 — anchor retention threshold
  rho_wai?: number;    // default 0.75 — redundancy threshold
  sigma_wai?: number;  // default 0.70 — quality threshold
  eta_prog?: number;   // default 0.03 — progress stall threshold
  h_anchor?: number;   // default 0.02 — anchor flip threshold

  // Rolling window size
  window?: number; // default 5

  // Turn limit
  max_turns?: number; // default 50
}

// ─── STEP HISTORY (internal state) ───

export interface StepRecord {
  t: number;
  delta_s: number;
  R_t: number;
  E_resonance: number;
  anchor_score: number;
  chi: number;
  zone: Zone;
  action: RecommendedAction;
}

// ─── LOOP GUARD ───

/** A single (zone, action) observation used for loop detection */
export interface ZoneActionEntry {
  zone: Zone;
  action: RecommendedAction;
}

/** Result of WRP (repeat-pattern) loop detection */
export interface LoopGuardResult {
  /** Whether a deterministic loop was detected */
  triggered: boolean;
  /** Human-readable key for the repeating pattern, e.g. "safe:continue" */
  pattern: string;
  /** Number of times the repeating pattern appeared in the window */
  repeat_count: number;
}
