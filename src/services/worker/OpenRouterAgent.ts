/**
 * OpenRouterAgent: OpenRouter-based observation extraction
 *
 * Alternative to SDKAgent that uses OpenRouter's unified API
 * for accessing 100+ models from different providers.
 *
 * Responsibility:
 * - Call OpenRouter REST API for observation extraction
 * - Parse XML responses (same format as Claude/Gemini)
 * - Sync to database and Chroma
 * - Support dynamic model selection across providers
 */

import { buildContinuationPrompt, buildInitPrompt, buildObservationPrompt, buildSummaryPrompt } from '../../sdk/prompts.js';
import { getCredential } from '../../shared/EnvManager.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../shared/paths.js';
import { logger } from '../../utils/logger.js';
import { ModeManager } from '../domain/ModeManager.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import {
  isAbortError,
  processAgentResponse,
  shouldFallbackToClaude,
  type FallbackAgent,
  type WorkerRef
} from './agents/index.js';

// OpenRouter API endpoint.
// Override via OPENROUTER_BASE_URL env var (e.g. http://127.0.0.1:4000/openrouter/v1 for CCProxy).
// See atl5s LLM routing standard: all providers funnel through CCProxy :4000.
const OPENROUTER_API_URL = process.env.OPENROUTER_BASE_URL
  ? `${process.env.OPENROUTER_BASE_URL.replace(/\/$/, '')}/chat/completions`
  : 'https://openrouter.ai/api/v1/chat/completions';

// Context window management constants (defaults, overridable via settings)
const DEFAULT_MAX_CONTEXT_MESSAGES = 20;  // Maximum messages to keep in conversation history
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;  // ~100k tokens max context (safety limit)
const CHARS_PER_TOKEN_ESTIMATE = 4;  // Conservative estimate: 1 token = 4 chars

/**
 * Collect all configured OpenRouter API keys in priority order:
 * 1. Primary key (CLAUDE_MEM_OPENROUTER_API_KEY)
 * 2. Pooled keys (CLAUDE_MEM_OPENROUTER_API_KEYS, comma-separated)
 * 3. Environment credential (OPENROUTER_API_KEY)
 * Deduplicates while preserving order.
 */
function getAllOpenRouterApiKeys(settings: ReturnType<typeof SettingsDefaultsManager.loadFromFile>): string[] {
  const keys: string[] = [];
  const seen = new Set<string>();

  const add = (k: string) => {
    const trimmed = k.trim();
    if (trimmed && !seen.has(trimmed)) {
      seen.add(trimmed);
      keys.push(trimmed);
    }
  };

  add(settings.CLAUDE_MEM_OPENROUTER_API_KEY || '');

  (settings.CLAUDE_MEM_OPENROUTER_API_KEYS || '')
    .split(',')
    .forEach(k => add(k));

  add(getCredential('OPENROUTER_API_KEY') || '');

  return keys;
}

/** Return the first available key (used by availability checks). */
function getOpenRouterApiKey(settings: ReturnType<typeof SettingsDefaultsManager.loadFromFile>): string {
  return getAllOpenRouterApiKeys(settings)[0] || '';
}

// OpenAI-compatible message format
interface OpenAIMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface OpenRouterResponse {
  choices?: Array<{
    message?: {
      role?: string;
      content?: string;
    };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  error?: {
    message?: string;
    code?: string;
  };
}

class OpenRouterApiError extends Error {
  public readonly statusCode: number;
  constructor(statusCode: number, message: string) {
    super(`OpenRouter API error: ${statusCode} - ${message}`);
    this.name = 'OpenRouterApiError';
    this.statusCode = statusCode;
  }
}

export class OpenRouterAgent {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;
  private fallbackAgent: FallbackAgent | null = null;
  private _nextKeyIndex: number = 0;
  private _keyCooldowns: Map<number, number> = new Map();
  private static readonly COOLDOWN_MS = 60_000;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  /**
   * Set the backup agent for when OpenRouter API fails
   * Must be set after construction to avoid circular dependency
   */
  setFallbackAgent(agent: FallbackAgent): void {
    this.fallbackAgent = agent;
  }

  /**
   * Start OpenRouter agent for a session
   * Uses multi-turn conversation to maintain context across messages
   */
  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    try {
      // Get OpenRouter configuration
      const { apiKeys, model, siteUrl, appName } = this.getOpenRouterConfig();

      if (apiKeys.length === 0) {
        throw new Error('OpenRouter API key not configured. Set CLAUDE_MEM_OPENROUTER_API_KEY in settings or OPENROUTER_API_KEY environment variable.');
      }

      // Generate synthetic memorySessionId (OpenRouter is stateless, doesn't return session IDs)
      if (!session.memorySessionId) {
        const syntheticMemorySessionId = `openrouter-${session.contentSessionId}-${Date.now()}`;
        session.memorySessionId = syntheticMemorySessionId;
        this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
        logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=OpenRouter`);
      }

      // Load active mode
      const mode = ModeManager.getInstance().getActiveMode();

      // Build initial prompt
      const initPrompt = session.lastPromptNumber === 1
        ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
        : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

      // Add to conversation history and query OpenRouter with full context
      session.conversationHistory.push({ role: 'user', content: initPrompt });
      const initResponse = await this.queryOpenRouterWithRotation(session.conversationHistory, apiKeys, model, siteUrl, appName);

      if (initResponse.content) {
        // Add response to conversation history
        session.conversationHistory.push({ role: 'assistant', content: initResponse.content });

        // Track token usage
        const tokensUsed = initResponse.tokensUsed || 0;
        session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);  // Rough estimate
        session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);

        // Process response using shared ResponseProcessor (no original timestamp for init - not from queue)
        await processAgentResponse(
          initResponse.content,
          session,
          this.dbManager,
          this.sessionManager,
          worker,
          tokensUsed,
          null,
          'OpenRouter',
          undefined  // No lastCwd yet - before message processing
        );
      } else {
        logger.error('SDK', 'Empty OpenRouter init response - session may lack context', {
          sessionId: session.sessionDbId,
          model
        });
      }

      // Track lastCwd from messages for CLAUDE.md generation
      let lastCwd: string | undefined;

      // Process pending messages
      for await (const message of this.sessionManager.getMessageIterator(session.sessionDbId)) {
        // CLAIM-CONFIRM: Track message ID for confirmProcessed() after successful storage
        // The message is now in 'processing' status in DB until ResponseProcessor calls confirmProcessed()
        session.processingMessageIds.push(message._persistentId);

        // Capture cwd from messages for proper worktree support
        if (message.cwd) {
          lastCwd = message.cwd;
        }
        // Capture earliest timestamp BEFORE processing (will be cleared after)
        const originalTimestamp = session.earliestPendingTimestamp;

        if (message.type === 'observation') {
          // Update last prompt number
          if (message.prompt_number !== undefined) {
            session.lastPromptNumber = message.prompt_number;
          }

          // CRITICAL: Check memorySessionId BEFORE making expensive LLM call
          // This prevents wasting tokens when we won't be able to store the result anyway
          if (!session.memorySessionId) {
            throw new Error('Cannot process observations: memorySessionId not yet captured. This session may need to be reinitialized.');
          }

          // Build observation prompt
          const obsPrompt = buildObservationPrompt({
            id: 0,
            tool_name: message.tool_name!,
            tool_input: JSON.stringify(message.tool_input),
            tool_output: JSON.stringify(message.tool_response),
            created_at_epoch: originalTimestamp ?? Date.now(),
            cwd: message.cwd
          });

          // Add to conversation history and query OpenRouter with full context
          session.conversationHistory.push({ role: 'user', content: obsPrompt });
          const obsResponse = await this.queryOpenRouterWithRotation(session.conversationHistory, apiKeys, model, siteUrl, appName);

          let tokensUsed = 0;
          if (obsResponse.content) {
            // Add response to conversation history
            session.conversationHistory.push({ role: 'assistant', content: obsResponse.content });

            tokensUsed = obsResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          // Process response using shared ResponseProcessor
          await processAgentResponse(
            obsResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'OpenRouter',
            lastCwd
          );

        } else if (message.type === 'summarize') {
          // CRITICAL: Check memorySessionId BEFORE making expensive LLM call
          if (!session.memorySessionId) {
            throw new Error('Cannot process summary: memorySessionId not yet captured. This session may need to be reinitialized.');
          }

          // Build summary prompt
          const summaryPrompt = buildSummaryPrompt({
            id: session.sessionDbId,
            memory_session_id: session.memorySessionId,
            project: session.project,
            user_prompt: session.userPrompt,
            last_assistant_message: message.last_assistant_message || ''
          }, mode);

          // Add to conversation history and query OpenRouter with full context
          session.conversationHistory.push({ role: 'user', content: summaryPrompt });
          const summaryResponse = await this.queryOpenRouterWithRotation(session.conversationHistory, apiKeys, model, siteUrl, appName);

          let tokensUsed = 0;
          if (summaryResponse.content) {
            // Add response to conversation history
            session.conversationHistory.push({ role: 'assistant', content: summaryResponse.content });

            tokensUsed = summaryResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          // Process response using shared ResponseProcessor
          await processAgentResponse(
            summaryResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'OpenRouter',
            lastCwd
          );
        }
      }

      // Mark session complete
      const sessionDuration = Date.now() - session.startTime;
      logger.success('SDK', 'OpenRouter agent completed', {
        sessionId: session.sessionDbId,
        duration: `${(sessionDuration / 1000).toFixed(1)}s`,
        historyLength: session.conversationHistory.length,
        model
      });

    } catch (error: unknown) {
      if (isAbortError(error)) {
        logger.warn('SDK', 'OpenRouter agent aborted', { sessionId: session.sessionDbId });
        throw error;
      }

      // Check if we should fail over to a backup provider
      if (shouldFallbackToClaude(error) && this.fallbackAgent) {
        logger.warn('SDK', 'OpenRouter API failed, falling back to backup provider', {
          sessionDbId: session.sessionDbId,
          error: error instanceof Error ? error.message : String(error),
          historyLength: session.conversationHistory.length
        });

        // Reuse the same session and conversation history during provider failover.
        // Note: With claim-and-delete queue pattern, messages are already deleted on claim
        return this.fallbackAgent.startSession(session, worker);
      }

      logger.failure('SDK', 'OpenRouter agent error', { sessionDbId: session.sessionDbId }, error as Error);
      throw error;
    }
  }

  /**
   * Estimate token count from text (conservative estimate)
   */
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  /**
   * Truncate conversation history to prevent runaway context costs
   * Keeps most recent messages within token budget
   */
  private truncateHistory(history: ConversationMessage[]): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const MAX_CONTEXT_MESSAGES = parseInt(settings.CLAUDE_MEM_OPENROUTER_MAX_CONTEXT_MESSAGES) || DEFAULT_MAX_CONTEXT_MESSAGES;
    const MAX_ESTIMATED_TOKENS = parseInt(settings.CLAUDE_MEM_OPENROUTER_MAX_TOKENS) || DEFAULT_MAX_ESTIMATED_TOKENS;

    if (history.length <= MAX_CONTEXT_MESSAGES) {
      // Check token count even if message count is ok
      const totalTokens = history.reduce((sum, m) => sum + this.estimateTokens(m.content), 0);
      if (totalTokens <= MAX_ESTIMATED_TOKENS) {
        return history;
      }
    }

    // Sliding window: keep most recent messages within limits
    const truncated: ConversationMessage[] = [];
    let tokenCount = 0;

    // Process messages in reverse (most recent first)
    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      const msgTokens = this.estimateTokens(msg.content);

      if (truncated.length >= MAX_CONTEXT_MESSAGES || tokenCount + msgTokens > MAX_ESTIMATED_TOKENS) {
        logger.warn('SDK', 'Context window truncated to prevent runaway costs', {
          originalMessages: history.length,
          keptMessages: truncated.length,
          droppedMessages: i + 1,
          estimatedTokens: tokenCount,
          tokenLimit: MAX_ESTIMATED_TOKENS
        });
        break;
      }

      truncated.unshift(msg);  // Add to beginning
      tokenCount += msgTokens;
    }

    return truncated;
  }

  /**
   * Convert shared ConversationMessage array to OpenAI-compatible message format
   */
  private conversationToOpenAIMessages(history: ConversationMessage[]): OpenAIMessage[] {
    return history.map(msg => ({
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      content: msg.content
    }));
  }

  /**
   * Query OpenRouter with round-robin key rotation and cooldown tracking.
   * Distributes load across keys; on retryable errors (401, 402, 429, 504), advances to next key.
   * Rate-limited keys (429) are placed in a 60s cooldown to avoid hammering them.
   */
  private async queryOpenRouterWithRotation(
    history: ConversationMessage[],
    apiKeys: string[],
    model: string,
    siteUrl?: string,
    appName?: string
  ): Promise<{ content: string; tokensUsed?: number }> {
    let lastError: Error | null = null;
    const startIndex = this._nextKeyIndex % apiKeys.length;
    let attemptsLeft = apiKeys.length;

    for (let attempt = 0; attempt < apiKeys.length; attempt++) {
      const keyIndex = (startIndex + attempt) % apiKeys.length;

      if (this.isKeyInCooldown(keyIndex)) {
        logger.debug('SDK', `OpenRouter key ${keyIndex + 1}/${apiKeys.length} in cooldown, skipping`);
        attemptsLeft--;
        continue;
      }

      try {
        const result = await this.queryOpenRouterMultiTurn(history, apiKeys[keyIndex], model, siteUrl, appName);
        // Advance to next key for subsequent calls (round-robin)
        this._nextKeyIndex = (keyIndex + 1) % apiKeys.length;
        return result;
      } catch (err) {
        lastError = err as Error;
        attemptsLeft--;
        const statusCode = this.extractHttpStatus(lastError);
        const isRetryable = statusCode !== null && [401, 402, 429, 504].includes(statusCode);

        if (statusCode === 429) {
          this.cooldownKey(keyIndex);
        }

        if (isRetryable && attemptsLeft > 0) {
          logger.warn('SDK', `OpenRouter key ${keyIndex + 1}/${apiKeys.length} failed (${statusCode}), rotating to next key`);
          continue;
        }
        throw lastError;
      }
    }

    // All keys exhausted (likely all in cooldown)
    if (this._keyCooldowns.size >= apiKeys.length) {
      logger.warn('SDK', 'All OpenRouter keys in cooldown, clearing cooldowns');
      this._keyCooldowns.clear();
    }

    throw lastError || new Error('No OpenRouter API keys available');
  }

  /** Extract HTTP status code from error (typed or string-based fallback) */
  private extractHttpStatus(error: Error): number | null {
    if (error instanceof OpenRouterApiError) {
      return error.statusCode;
    }
    const match = (error.message || '').match(/\b(4\d{2}|5\d{2})\b/);
    return match ? parseInt(match[1], 10) : null;
  }

  /** Check if a key is in cooldown (auto-expires) */
  private isKeyInCooldown(keyIndex: number): boolean {
    const cooldownUntil = this._keyCooldowns.get(keyIndex);
    if (!cooldownUntil) return false;
    if (Date.now() >= cooldownUntil) {
      this._keyCooldowns.delete(keyIndex);
      return false;
    }
    return true;
  }

  /** Put a key into cooldown after rate limiting */
  private cooldownKey(keyIndex: number): void {
    this._keyCooldowns.set(keyIndex, Date.now() + OpenRouterAgent.COOLDOWN_MS);
    logger.info('SDK', `OpenRouter key ${keyIndex + 1} placed in ${OpenRouterAgent.COOLDOWN_MS / 1000}s cooldown`);
  }

  /**
   * Query OpenRouter via REST API with full conversation history (multi-turn)
   * Sends the entire conversation context for coherent responses
   */
  private async queryOpenRouterMultiTurn(
    history: ConversationMessage[],
    apiKey: string,
    model: string,
    siteUrl?: string,
    appName?: string
  ): Promise<{ content: string; tokensUsed?: number }> {
    // Truncate history to prevent runaway costs
    const truncatedHistory = this.truncateHistory(history);
    const messages = this.conversationToOpenAIMessages(truncatedHistory);
    const totalChars = truncatedHistory.reduce((sum, m) => sum + m.content.length, 0);
    const estimatedTokens = this.estimateTokens(truncatedHistory.map(m => m.content).join(''));

    logger.debug('SDK', `Querying OpenRouter multi-turn (${model})`, {
      turns: truncatedHistory.length,
      totalChars,
      estimatedTokens
    });

    const response = await fetch(OPENROUTER_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'HTTP-Referer': siteUrl || 'https://github.com/thedotmack/claude-mem',
        'X-Title': appName || 'claude-mem',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.3,  // Lower temperature for structured extraction
        max_tokens: parseInt(SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH).CLAUDE_MEM_OPENROUTER_MAX_RESPONSE_TOKENS) || 1024,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new OpenRouterApiError(response.status, errorText);
    }

    const data = await response.json() as OpenRouterResponse;

    // Check for API error in response body
    if (data.error) {
      const code = parseInt(data.error.code || '0', 10) || 500;
      throw new OpenRouterApiError(code, data.error.message || 'Unknown error');
    }

    if (!data.choices?.[0]?.message?.content) {
      logger.error('SDK', 'Empty response from OpenRouter');
      return { content: '' };
    }

    const content = data.choices[0].message.content;
    const tokensUsed = data.usage?.total_tokens;

    // Log actual token usage for cost tracking
    if (tokensUsed) {
      const inputTokens = data.usage?.prompt_tokens || 0;
      const outputTokens = data.usage?.completion_tokens || 0;
      // Token usage (cost varies by model - many OpenRouter models are free)
      const estimatedCost = (inputTokens / 1000000 * 3) + (outputTokens / 1000000 * 15);

      logger.info('SDK', 'OpenRouter API usage', {
        model,
        inputTokens,
        outputTokens,
        totalTokens: tokensUsed,
        estimatedCostUSD: estimatedCost.toFixed(4),
        messagesInContext: truncatedHistory.length
      });

      // Warn if costs are getting high
      if (tokensUsed > 50000) {
        logger.warn('SDK', 'High token usage detected - consider reducing context', {
          totalTokens: tokensUsed,
          estimatedCost: estimatedCost.toFixed(4)
        });
      }
    }

    return { content, tokensUsed };
  }

  /**
   * Get OpenRouter configuration from settings or environment
   * Issue #733: Uses centralized ~/.claude-mem/.env for credentials, not random project .env files
   */
  private getOpenRouterConfig(): { apiKeys: string[]; model: string; siteUrl?: string; appName?: string } {
    const settingsPath = USER_SETTINGS_PATH;
    const settings = SettingsDefaultsManager.loadFromFile(settingsPath);

    // API keys: collect all configured keys for rotation on failure
    // Priority: primary key > pooled keys > env credential
    // This prevents Issue #733 where random project .env files could interfere
    const apiKeys = getAllOpenRouterApiKeys(settings);

    // Model: from settings or default
    const model = settings.CLAUDE_MEM_OPENROUTER_MODEL || 'xiaomi/mimo-v2-flash:free';

    // Optional analytics headers
    const siteUrl = settings.CLAUDE_MEM_OPENROUTER_SITE_URL || '';
    const appName = settings.CLAUDE_MEM_OPENROUTER_APP_NAME || 'claude-mem';

    return { apiKeys, model, siteUrl, appName };
  }
}

/**
 * Check if OpenRouter is available (has API key configured)
 * Issue #733: Uses centralized ~/.claude-mem/.env, not random project .env files
 */
export function isOpenRouterAvailable(): boolean {
  const settingsPath = USER_SETTINGS_PATH;
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
  return !!getOpenRouterApiKey(settings);
}

/**
 * Check if OpenRouter is the selected provider
 */
export function isOpenRouterSelected(): boolean {
  const settingsPath = USER_SETTINGS_PATH;
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
  return settings.CLAUDE_MEM_PROVIDER === 'openrouter';
}
