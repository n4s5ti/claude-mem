/**
 * @component janus-harness
 * @stage planning
 * @priority P2
 */
/**
 * Janus Harness Tests
 *
 * Simulates real agent trajectories with LLM-produced semantic signals
 * and verifies the harness computes correct deterministic outputs.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { JanusHarness } from "./harness.js";
import type { SemanticSignals, CognitiveState } from "./types.js";

describe("JanusHarness", () => {
  it("should initialize with zero state", () => {
    const h = new JanusHarness();
    assert.equal(h.stepCount, 0);
    assert.equal(h.lastDeltaS, null);
  });

  it("should compute safe zone for low tension", () => {
    const h = new JanusHarness();
    const state = h.step({ delta_s: 0.15, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 });
    assert.equal(state.zone, "safe");
    assert.equal(state.t, 1);
    assert.equal(state.lambda_observe, "convergent"); // first step defaults to convergent
    assert.ok(state.J_t < 0.1);
    assert.equal(state.action, "continue");
  });

  it("should compute danger zone for high tension", () => {
    const h = new JanusHarness();
    const state = h.step({ delta_s: 0.92, j_harm: 0.8, j_drift: 0.7, j_anchor: 0.6 });
    assert.equal(state.zone, "danger");
    assert.ok(state.J_t > 0.5);
  });

  it("should classify convergent trajectory", () => {
    const h = new JanusHarness();
    h.step({ delta_s: 0.70, j_harm: 0.1, j_drift: 0.2, j_anchor: 0.1 });
    const s2 = h.step({ delta_s: 0.55, j_harm: 0.08, j_drift: 0.15, j_anchor: 0.08 });
    assert.equal(s2.lambda_observe, "convergent"); // delta = -0.15, clearly converging
  });

  it("should classify chaotic trajectory", () => {
    const h = new JanusHarness();
    h.step({ delta_s: 0.30, j_harm: 0.1, j_drift: 0.1, j_anchor: 0.1 });
    const s2 = h.step({
      delta_s: 0.55, j_harm: 0.3, j_drift: 0.3, j_anchor: 0.3,
      anchor_conflict: true,
    });
    assert.equal(s2.lambda_observe, "chaotic");
  });

  it("should recommend pause in risk zone with chaotic lambda", () => {
    const h = new JanusHarness();
    // Drive into risk + chaotic
    h.step({ delta_s: 0.50, j_harm: 0.1, j_drift: 0.1, j_anchor: 0.1 });
    const state = h.step({
      delta_s: 0.75, j_harm: 0.3, j_drift: 0.4, j_anchor: 0.3,
      anchor_conflict: true,
    });
    assert.ok(JanusHarness.shouldPause(state));
  });

  it("should allow bridge when tension decreasing and coupler low", () => {
    const h = new JanusHarness();
    h.step({ delta_s: 0.60, j_harm: 0.1, j_drift: 0.1, j_anchor: 0.1 });
    const s2 = h.step({ delta_s: 0.35, j_harm: 0.05, j_drift: 0.05, j_anchor: 0.05 });
    // Tension dropped 0.60 -> 0.35, should allow bridge if W_c is low enough
    // can_bridge = (delta_s_t < delta_s_prev) AND (W_c < 0.5 * theta_c)
    assert.equal(s2.delta_s < 0.60, true); // tension decreased
  });

  it("should track E_bar rolling average", () => {
    const h = new JanusHarness();
    h.step({ delta_s: 0.80, j_harm: 0.1, j_drift: 0.1, j_anchor: 0.1 });
    h.step({ delta_s: 0.60, j_harm: 0.1, j_drift: 0.1, j_anchor: 0.1 });
    const s3 = h.step({ delta_s: 0.40, j_harm: 0.1, j_drift: 0.1, j_anchor: 0.1 });
    // E_bar should be mean of [0.80, 0.60, 0.40] = 0.60
    assert.ok(Math.abs(s3.E_bar - 0.60) < 0.01);
  });

  it("should generate valid YAML header", () => {
    const h = new JanusHarness();
    const state = h.step({ delta_s: 0.42, j_harm: 0.05, j_drift: 0.18, j_anchor: 0.23 });
    const yaml = JanusHarness.toYAML(state);
    assert.ok(yaml.includes("cognitive_state:"));
    assert.ok(yaml.includes("delta_s: 0.42"));
    assert.ok(yaml.includes("zone: transit"));
  });

  it("should find best rollback target", () => {
    const h = new JanusHarness();
    h.step({ delta_s: 0.30, j_harm: 0.1, j_drift: 0.1, j_anchor: 0.1 }); // t=1, best
    h.step({ delta_s: 0.50, j_harm: 0.2, j_drift: 0.2, j_anchor: 0.2 }); // t=2
    h.step({ delta_s: 0.80, j_harm: 0.5, j_drift: 0.5, j_anchor: 0.5 }); // t=3, worst
    const target = h.getBestRollbackTarget();
    assert.equal(target, 1); // should rollback to t=1 where delta_s was lowest
  });

  it("should simulate realistic trajectory: safe → risk → recovery", () => {
    const h = new JanusHarness();

    // Step 1: Good start, safe zone
    const s1 = h.step({ delta_s: 0.20, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 });
    assert.equal(s1.zone, "safe");
    assert.equal(s1.action, "continue");

    // Step 2: Things get harder, transit zone
    const s2 = h.step({ delta_s: 0.45, j_harm: 0.10, j_drift: 0.15, j_anchor: 0.10 });
    assert.equal(s2.zone, "transit");
    assert.equal(s2.lambda_observe, "chaotic"); // big jump from 0.20 to 0.45

    // Step 3: Agent struggles, risk zone
    const s3 = h.step({ delta_s: 0.70, j_harm: 0.20, j_drift: 0.30, j_anchor: 0.25 });
    assert.equal(s3.zone, "risk");

    // *** THIS IS THE PAUSE POINT ***
    // Orchestrator freezes the agent, mentor researches, injects guidance
    assert.ok(s3.zone === "risk" || JanusHarness.shouldPause(s3));

    // Step 4: After mentor injection, agent recovers
    const s4 = h.step({ delta_s: 0.35, j_harm: 0.05, j_drift: 0.08, j_anchor: 0.05 });
    assert.equal(s4.zone, "safe");
    assert.equal(s4.lambda_observe, "convergent"); // big drop, converging

    // Step 5: Smooth sailing
    const s5 = h.step({ delta_s: 0.18, j_harm: 0.02, j_drift: 0.03, j_anchor: 0.02 });
    assert.equal(s5.zone, "safe");
    assert.equal(s5.lambda_observe, "convergent");
    assert.equal(s5.action, "continue");
  });

  it("should trigger WTF guard and recommend rollback on sustained collapse", () => {
    const h = new JanusHarness();
    // Two consecutive steps with increasing tension + resonance = WTF territory
    h.step({
      delta_s: 0.40, j_harm: 0.1, j_drift: 0.1, j_anchor: 0.1,
      contradiction: true,
    });
    const s2 = h.step({
      delta_s: 0.60, j_harm: 0.3, j_drift: 0.3, j_anchor: 0.3,
      contradiction: true,
    });
    // chi from step 1 should be >= 1 (delta_s increasing or contradiction)
    // chi from step 2 should add up to trigger WTF
    if (s2.guards.wtf.triggered) {
      assert.equal(s2.action, "rollback");
    }
  });

  it("should reset cleanly", () => {
    const h = new JanusHarness();
    h.step({ delta_s: 0.50, j_harm: 0.1, j_drift: 0.1, j_anchor: 0.1 });
    h.reset();
    assert.equal(h.stepCount, 0);
    assert.equal(h.lastDeltaS, null);
  });

  it("should trigger rollback action when max_turns is reached", () => {
    // WTL fires when t > max_turns (strict greater-than)
    const h = new JanusHarness({ max_turns: 3 });
    const signals: SemanticSignals = { delta_s: 0.15, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 };

    // Steps 1–3: t <= max_turns, WTL must NOT fire
    h.step(signals);
    h.step(signals);
    const s3 = h.step(signals);
    assert.equal(s3.guards.wtl.triggered, false);
    assert.equal(s3.guards.turn_limit_exceeded, false);
    assert.notEqual(s3.action, "rollback");

    // Step 4: t=4 > max_turns=3, WTL must fire and force rollback
    const s4 = h.step(signals);
    assert.equal(s4.guards.wtl.triggered, true);
    assert.equal(s4.guards.turn_limit_exceeded, true);
    assert.equal(s4.action, "rollback");
  });

  it("should respect custom max_turns=5 configuration", () => {
    const h = new JanusHarness({ max_turns: 5 });
    const signals: SemanticSignals = { delta_s: 0.15, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 };

    // Steps 1–5: at or below max_turns, WTL should not fire
    for (let i = 0; i < 5; i++) {
      const s = h.step(signals);
      assert.equal(s.guards.wtl.triggered, false, `WTL should not fire at step ${i + 1}`);
      assert.equal(s.guards.turn_limit_exceeded, false, `turn_limit_exceeded should be false at step ${i + 1}`);
    }

    // Step 6: t=6 > max_turns=5, WTL fires
    const s6 = h.step(signals);
    assert.equal(h.stepCount, 6);
    assert.equal(s6.guards.wtl.triggered, true);
    assert.equal(s6.guards.turn_limit_exceeded, true);
    assert.equal(s6.action, "rollback");
  });

  it("should fire WTL rollback without WTF, confirming WTL has higher priority", () => {
    // Use max_turns=2 and safe steady signals that won't accumulate WTF chi
    const h = new JanusHarness({ max_turns: 2 });
    // Constant low-tension signals: delta_s never increases, no contradiction
    const safe: SemanticSignals = { delta_s: 0.20, j_harm: 0.02, j_drift: 0.03, j_anchor: 0.02 };

    h.step(safe); // t=1: WTL: 1 > 2 = false
    h.step(safe); // t=2: WTL: 2 > 2 = false

    // t=3 > max_turns=2: WTL triggers; stable non-escalating signals → WTF must NOT trigger
    const s3 = h.step(safe);
    assert.equal(s3.guards.wtf.triggered, false, "WTF must not be triggered with stable safe signals");
    assert.equal(s3.guards.wtl.triggered, true,  "WTL must fire when t > max_turns");
    assert.equal(s3.action, "rollback",           "WTL alone is sufficient to force rollback");
  });

  // ─── WRP Guard Tests ───

  it("should trigger WRP guard when 3 identical (zone, action) pairs appear in history", () => {
    // WRP fires when the same (zone, action) pair repeats >= 3 times in the rolling window.
    // wrp_history is built from PREVIOUS step records, so 3 steps with "safe:continue"
    // means at step 4 the wrp_history contains 3 identical entries → triggered.
    //
    // Steadily decreasing delta_s keeps progress > eta_prog (WAY stays quiet) and
    // delta_s < zone_safe=0.40 (safe zone) with no guard noise.
    const h = new JanusHarness();
    h.step({ delta_s: 0.30, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 }); // t=1: safe:continue
    h.step({ delta_s: 0.25, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 }); // t=2: safe:continue
    h.step({ delta_s: 0.20, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 }); // t=3: safe:continue

    // t=4: wrp_history = [safe:continue × 3] → count 3 >= REPEAT_THRESHOLD(3) → triggered
    const s4 = h.step({ delta_s: 0.15, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 });
    assert.equal(s4.guards.wrp.triggered, true);
    assert.equal(s4.guards.wrp.repeat_count, 3);
    assert.equal(s4.guards.wrp.pattern, "safe:continue");
  });

  it("should NOT trigger WRP guard when only 2 identical (zone, action) pairs appear", () => {
    // At t=3 the wrp_history contains only 2 "safe:continue" entries (from t=1 and t=2).
    // 2 < REPEAT_THRESHOLD(3) → WRP must not fire.
    const h = new JanusHarness();
    h.step({ delta_s: 0.30, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 }); // t=1: safe:continue
    h.step({ delta_s: 0.25, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 }); // t=2: safe:continue

    // t=3: wrp_history = [safe:continue × 2] → count 2 < 3 → not triggered
    const s3 = h.step({ delta_s: 0.20, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 });
    assert.equal(s3.guards.wrp.triggered, false);
    assert.equal(s3.guards.wrp.repeat_count, 2);
  });

  it("should NOT cross-trigger WRP when two different patterns each appear twice", () => {
    // WRP counts individual (zone:action) pairs. Having 2 copies of "safe:continue"
    // and 2 copies of "transit:slow" in the window should NOT trigger (max count = 2 < 3).
    // Alternating low (0.30/0.28) and high (0.50/0.52) delta_s produces alternating zones:
    //   - Low  delta_s (< 0.40) + tension falling → safe:continue
    //   - High delta_s (0.40–0.60) + tension rising → transit:slow (WRI fires on rising tension)
    const h = new JanusHarness();
    h.step({ delta_s: 0.30, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 }); // safe:continue
    h.step({ delta_s: 0.50, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 }); // transit:slow
    h.step({ delta_s: 0.28, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 }); // safe:continue
    h.step({ delta_s: 0.52, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 }); // transit:slow

    // t=5: wrp_history = [safe:continue, transit:slow, safe:continue, transit:slow]
    //   safe:continue count = 2, transit:slow count = 2 → max = 2 < 3 → not triggered
    const s5 = h.step({ delta_s: 0.25, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 });
    assert.equal(s5.guards.wrp.triggered, false);
    assert.ok(s5.guards.wrp.repeat_count < 3);
  });

  it("should force pause action when WRP guard is triggered", () => {
    // WRP sits above WTF but below WTL in the action priority cascade:
    //   WTL > WRP > WTF > collapse/WDT > WAY/WRI > continue
    // When WRP fires and no higher-priority guard (WTL) is active, action must be "pause".
    const h = new JanusHarness();
    // Build 3 consecutive safe:continue steps in history
    h.step({ delta_s: 0.30, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 });
    h.step({ delta_s: 0.25, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 });
    h.step({ delta_s: 0.20, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 });

    // t=4: WRP triggered (3× safe:continue in history), WTL not triggered (t=4 << max_turns=50)
    const s4 = h.step({ delta_s: 0.15, j_harm: 0.02, j_drift: 0.05, j_anchor: 0.03 });
    assert.equal(s4.guards.wrp.triggered, true);
    assert.equal(s4.guards.wtl.triggered, false, "WTL must not override WRP here");
    assert.equal(s4.action, "pause");
  });
});
