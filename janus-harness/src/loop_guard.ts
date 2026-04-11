/**
 * @component janus-harness
 * @stage planning
 * @priority P2
 */
/**
 * Janus Harness — WRP Loop Guard
 *
 * Deterministic loop-prevention via repeat-pattern detection.
 * Pure function. Zero tokens. Zero API calls.
 *
 * Detects when the same (zone, action) pair repeats 3+ times within
 * the last `window` steps, indicating a stuck cognitive loop.
 */

import type {
  ZoneActionEntry,
  LoopGuardResult,
} from "./types.js";

// ─── CONSTANTS ───

/** Default rolling window size (matches HarnessConfig default) */
const DEFAULT_WINDOW = 5;

/** Minimum repeat count to trigger the guard */
const REPEAT_THRESHOLD = 3;

// ─── WRP: REPEAT-PATTERN GUARD ───

/**
 * Detect whether the agent is stuck in a deterministic (zone, action) loop.
 *
 * Scans the last `window` entries in `history` and checks whether any
 * single (zone, action) pair appears `REPEAT_THRESHOLD` or more times.
 * When multiple pairs tie, the one with the highest count is reported.
 *
 * @param history  Ordered list of (zone, action) observations — oldest first
 * @param window   How many tail entries to inspect (default 5)
 * @returns        LoopGuardResult with triggered flag, pattern key, and count
 */
export function detectRepeatPattern(
  history: ZoneActionEntry[],
  window: number = DEFAULT_WINDOW,
): LoopGuardResult {
  if (history.length === 0) {
    return { triggered: false, pattern: "", repeat_count: 0 };
  }

  const slice = history.slice(-window);

  // Tally occurrences of each (zone:action) key within the window
  const counts = new Map<string, number>();
  for (const entry of slice) {
    const key = `${entry.zone}:${entry.action}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  // Find the most-repeated pattern
  let topKey = "";
  let topCount = 0;
  for (const [key, count] of counts) {
    if (count > topCount) {
      topKey = key;
      topCount = count;
    }
  }

  const triggered = topCount >= REPEAT_THRESHOLD;

  return {
    triggered,
    pattern: triggered ? topKey : "",
    repeat_count: topCount,
  };
}
