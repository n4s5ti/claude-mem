import { describe, it, expect, beforeEach, afterEach, spyOn, mock } from 'bun:test';
import { OpenRouterAgent } from '../src/services/worker/OpenRouterAgent';
import { DatabaseManager } from '../src/services/worker/DatabaseManager';
import { SessionManager } from '../src/services/worker/SessionManager';
import { ModeManager } from '../src/services/domain/ModeManager';
import { SettingsDefaultsManager } from '../src/shared/SettingsDefaultsManager';

const mockMode = {
  name: 'code',
  prompts: { init: 'init prompt', observation: 'obs prompt', summary: 'summary prompt' },
  observation_types: [{ id: 'discovery' }, { id: 'bugfix' }],
  observation_concepts: []
};

let loadFromFileSpy: ReturnType<typeof spyOn>;
let getSpy: ReturnType<typeof spyOn>;
let modeManagerSpy: ReturnType<typeof spyOn>;

function makeOpenRouterResponse(content: string, tokens = 100) {
  return new Response(JSON.stringify({
    choices: [{ message: { role: 'assistant', content }, finish_reason: 'stop' }],
    usage: { prompt_tokens: tokens * 0.7, completion_tokens: tokens * 0.3, total_tokens: tokens }
  }));
}

function makeSession(overrides: Record<string, any> = {}) {
  return {
    sessionDbId: 1,
    contentSessionId: 'test-session',
    memorySessionId: 'mem-session-123',
    project: 'test-project',
    userPrompt: 'test prompt',
    conversationHistory: [],
    lastPromptNumber: 1,
    cumulativeInputTokens: 0,
    cumulativeOutputTokens: 0,
    pendingMessages: [],
    abortController: new AbortController(),
    generatorPromise: null,
    earliestPendingTimestamp: null,
    currentProvider: null,
    startTime: Date.now(),
    processingMessageIds: [],
    ...overrides
  } as any;
}

describe('OpenRouterAgent', () => {
  let agent: OpenRouterAgent;
  let originalFetch: typeof global.fetch;
  let mockDbManager: DatabaseManager;
  let mockSessionManager: SessionManager;

  beforeEach(() => {
    modeManagerSpy = spyOn(ModeManager, 'getInstance').mockImplementation(() => ({
      getActiveMode: () => mockMode,
      loadMode: () => {},
    } as any));

    loadFromFileSpy = spyOn(SettingsDefaultsManager, 'loadFromFile').mockImplementation(() => ({
      ...SettingsDefaultsManager.getAllDefaults(),
      CLAUDE_MEM_OPENROUTER_API_KEY: 'sk-or-key-1',
      CLAUDE_MEM_OPENROUTER_API_KEYS: 'sk-or-key-2,sk-or-key-3',
      CLAUDE_MEM_OPENROUTER_MODEL: 'xiaomi/mimo-v2-flash:free',
      CLAUDE_MEM_DATA_DIR: '/tmp/claude-mem-test',
    }));

    getSpy = spyOn(SettingsDefaultsManager, 'get').mockImplementation((key: string) => {
      if (key === 'CLAUDE_MEM_OPENROUTER_API_KEY') return 'sk-or-key-1';
      if (key === 'CLAUDE_MEM_OPENROUTER_API_KEYS') return 'sk-or-key-2,sk-or-key-3';
      if (key === 'CLAUDE_MEM_OPENROUTER_MODEL') return 'xiaomi/mimo-v2-flash:free';
      if (key === 'CLAUDE_MEM_DATA_DIR') return '/tmp/claude-mem-test';
      return SettingsDefaultsManager.getAllDefaults()[key as keyof ReturnType<typeof SettingsDefaultsManager.getAllDefaults>] ?? '';
    });

    const mockStoreObservations = mock(() => ({
      observationIds: [1],
      summaryId: 1,
      createdAtEpoch: Date.now()
    }));

    const mockSessionStore = {
      storeObservation: mock(() => ({ id: 1, createdAtEpoch: Date.now() })),
      storeObservations: mockStoreObservations,
      storeSummary: mock(() => ({ id: 1, createdAtEpoch: Date.now() })),
      markSessionCompleted: mock(() => {}),
      updateMemorySessionId: mock(() => {}),
      getSessionById: mock(() => ({ memory_session_id: 'mem-session-123' })),
      ensureMemorySessionIdRegistered: mock(() => {})
    };

    const mockChromaSync = {
      syncObservation: mock(() => Promise.resolve()),
      syncSummary: mock(() => Promise.resolve())
    };

    mockDbManager = {
      getSessionStore: () => mockSessionStore,
      getChromaSync: () => mockChromaSync
    } as unknown as DatabaseManager;

    const mockPendingMessageStore = {
      markProcessed: mock(() => {}),
      confirmProcessed: mock(() => {}),
      cleanupProcessed: mock(() => 0),
      resetStuckMessages: mock(() => 0)
    };

    mockSessionManager = {
      getMessageIterator: async function* () { yield* []; },
      getPendingMessageStore: () => mockPendingMessageStore
    } as unknown as SessionManager;

    agent = new OpenRouterAgent(mockDbManager, mockSessionManager);
    originalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
    if (modeManagerSpy) modeManagerSpy.mockRestore();
    if (loadFromFileSpy) loadFromFileSpy.mockRestore();
    if (getSpy) getSpy.mockRestore();
    mock.restore();
  });

  it('should initialize with correct config and call OpenRouter API', async () => {
    const session = makeSession();

    global.fetch = mock(() => Promise.resolve(
      makeOpenRouterResponse('<observation><type>discovery</type><title>Test</title></observation>')
    ));

    await agent.startSession(session);

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [url, opts] = (global.fetch as any).mock.calls[0];
    expect(url).toBe('https://openrouter.ai/api/v1/chat/completions');
    expect(opts.headers['Authorization']).toBe('Bearer sk-or-key-1');
  });

  it('should add assistant response to conversation history', async () => {
    const session = makeSession();

    global.fetch = mock(() => Promise.resolve(
      makeOpenRouterResponse('<observation><type>discovery</type><title>Test</title></observation>')
    ));

    await agent.startSession(session);

    // Should have at least user (init prompt) and assistant (response) in history
    expect(session.conversationHistory.length).toBeGreaterThanOrEqual(2);
    // Verify assistant response is present (multi-turn coherence)
    const hasAssistant = session.conversationHistory.some((m: any) => m.role === 'assistant');
    expect(hasAssistant).toBe(true);
  });

  it('should round-robin keys across consecutive sessions', async () => {
    const usedKeys: string[] = [];

    global.fetch = mock((url: string, opts: any) => {
      usedKeys.push(opts.headers['Authorization']);
      return Promise.resolve(
        makeOpenRouterResponse('<observation><type>discovery</type><title>Test</title></observation>')
      );
    });

    // Run 3 sessions — each should use a different key
    for (let i = 0; i < 3; i++) {
      await agent.startSession(makeSession());
    }

    expect(usedKeys).toEqual([
      'Bearer sk-or-key-1',
      'Bearer sk-or-key-2',
      'Bearer sk-or-key-3',
    ]);
  });

  it('should wrap around to first key after cycling all keys', async () => {
    const usedKeys: string[] = [];

    global.fetch = mock((url: string, opts: any) => {
      usedKeys.push(opts.headers['Authorization']);
      return Promise.resolve(
        makeOpenRouterResponse('<observation><type>discovery</type><title>Test</title></observation>')
      );
    });

    // Run 4 sessions — 4th should wrap to key 1
    for (let i = 0; i < 4; i++) {
      await agent.startSession(makeSession());
    }

    expect(usedKeys[3]).toBe('Bearer sk-or-key-1');
  });

  it('should rotate to next key on 429 rate limit', async () => {
    let callCount = 0;
    const usedKeys: string[] = [];

    global.fetch = mock((url: string, opts: any) => {
      callCount++;
      usedKeys.push(opts.headers['Authorization']);

      if (callCount === 1) {
        // First key returns 429
        return Promise.resolve(new Response('Rate limited', { status: 429 }));
      }
      return Promise.resolve(
        makeOpenRouterResponse('<observation><type>discovery</type><title>Test</title></observation>')
      );
    });

    await agent.startSession(makeSession());

    // Should have tried 2 keys
    expect(usedKeys.length).toBe(2);
    expect(usedKeys[0]).toBe('Bearer sk-or-key-1');
    expect(usedKeys[1]).toBe('Bearer sk-or-key-2');
  });

  it('should skip cooled-down key on subsequent call', async () => {
    let callCount = 0;
    const usedKeys: string[] = [];

    global.fetch = mock((url: string, opts: any) => {
      callCount++;
      usedKeys.push(opts.headers['Authorization']);

      if (callCount === 1) {
        // First key returns 429 — will be cooled down
        return Promise.resolve(new Response('Rate limited', { status: 429 }));
      }
      return Promise.resolve(
        makeOpenRouterResponse('<observation><type>discovery</type><title>Test</title></observation>')
      );
    });

    // Session 1: key-1 gets 429, falls to key-2 (succeeds)
    await agent.startSession(makeSession());

    // Session 2: should skip key-1 (in cooldown) and go to key-3
    await agent.startSession(makeSession());

    // Call 1: key-1 (429), Call 2: key-2 (success), Call 3: key-3 (success, skipped key-1)
    expect(usedKeys[2]).toBe('Bearer sk-or-key-3');
  });

  it('should throw when all keys fail with non-retryable error', async () => {
    global.fetch = mock(() => Promise.resolve(new Response('Bad request', { status: 400 })));

    // 400 is not retryable, should fail on first key without rotation
    await expect(agent.startSession(makeSession())).rejects.toThrow('OpenRouter API error: 400');
  });

  it('should throw after exhausting all keys on retryable errors', async () => {
    global.fetch = mock(() => Promise.resolve(new Response('Rate limited', { status: 429 })));

    // All 3 keys return 429
    await expect(agent.startSession(makeSession())).rejects.toThrow('429');
  });

  it('should generate synthetic memorySessionId when not set', async () => {
    const session = makeSession({ memorySessionId: null });

    global.fetch = mock(() => Promise.resolve(
      makeOpenRouterResponse('<observation><type>discovery</type><title>Test</title></observation>')
    ));

    await agent.startSession(session);

    expect(session.memorySessionId).not.toBeNull();
    expect(session.memorySessionId).toContain('openrouter-');
  });

  it('should not regenerate memorySessionId if already present', async () => {
    const existingId = 'existing-memory-id';
    const session = makeSession({ memorySessionId: existingId });

    global.fetch = mock(() => Promise.resolve(
      makeOpenRouterResponse('<observation><type>discovery</type><title>Test</title></observation>')
    ));

    await agent.startSession(session);

    expect(session.memorySessionId).toBe(existingId);
  });
});
