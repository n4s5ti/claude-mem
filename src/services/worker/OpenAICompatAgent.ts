/**
 * OpenAICompatAgent: Generalized OpenAI-compatible API agent
 *
 * Works with any provider that implements the OpenAI chat/completions API:
 * - OpenRouter (100+ models via unified API)
 * - Groq (fast inference, free tier)
 * - Any future OpenAI-compatible provider
 *
 * Responsibility:
 * - Call provider REST API for observation extraction
 * - Parse XML responses (same format as Claude/Gemini)
 * - Sync to database and Chroma
 * - Support dynamic model selection and key rotation
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

// Context window management constants (defaults, overridable via settings)
const DEFAULT_MAX_CONTEXT_MESSAGES = 20;
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;
const CHARS_PER_TOKEN_ESTIMATE = 4;

// ============================================================================
// Provider Configuration
// ============================================================================

/**
 * Configuration for an OpenAI-compatible API provider.
 * Define one of these per provider to parameterize the agent.
 */
export interface OpenAICompatProviderConfig {
  /** Display name for logging (e.g., 'OpenRouter', 'Groq') */
  name: string;
  /** Chat completions API endpoint */
  apiUrl: string;
  /** Build request headers for a given API key */
  getHeaders(apiKey: string, extra?: Record<string, string>): Record<string, string>;
  /** Default model if none configured in settings */
  defaultModel: string;
  /** Settings key prefix (e.g., 'CLAUDE_MEM_OPENROUTER' → reads CLAUDE_MEM_OPENROUTER_API_KEY, etc.) */
  settingsPrefix: string;
  /** Environment variable name for fallback credential lookup (e.g., 'OPENROUTER_API_KEY') */
  envCredentialKey?: string;
  /** When false, skip API key checks — credentials managed externally (e.g., CCProxy). Defaults to true. */
  requiresApiKey?: boolean;
}

// ============================================================================
// Base URL resolution helpers.
// atl5s LLM routing standard: all providers funnel through CCProxy at :4000.
// Env vars follow standard SDK conventions (OPENROUTER_BASE_URL, GROQ_API_BASE)
// so setting OPENROUTER_BASE_URL=http://127.0.0.1:4000/openrouter/v1 in the
// shared op-mcp secrets.env redirects both direct and CCProxy-variant providers
// without a rebuild. Empty env var falls back to the upstream default.
// ============================================================================

function resolveOpenRouterUrl(fallback: string): string {
  const base = process.env.OPENROUTER_BASE_URL;
  if (!base) return fallback;
  return `${base.replace(/\/$/, '')}/chat/completions`;
}

function resolveGroqUrl(fallback: string): string {
  const base = process.env.GROQ_API_BASE;
  if (!base) return fallback;
  return `${base.replace(/\/$/, '')}/chat/completions`;
}

export const OPENROUTER_PROVIDER: OpenAICompatProviderConfig = {
  name: 'OpenRouter',
  apiUrl: resolveOpenRouterUrl('https://openrouter.ai/api/v1/chat/completions'),
  getHeaders: (apiKey, extra) => ({
    'Authorization': `Bearer ${apiKey}`,
    'HTTP-Referer': extra?.siteUrl || 'https://github.com/thedotmack/claude-mem',
    'X-Title': extra?.appName || 'claude-mem',
    'Content-Type': 'application/json',
  }),
  defaultModel: 'xiaomi/mimo-v2-flash:free',
  settingsPrefix: 'CLAUDE_MEM_OPENROUTER',
  envCredentialKey: 'OPENROUTER_API_KEY',
};

export const GROQ_PROVIDER: OpenAICompatProviderConfig = {
  name: 'Groq',
  apiUrl: resolveGroqUrl('https://api.groq.com/openai/v1/chat/completions'),
  getHeaders: (apiKey) => ({
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  }),
  defaultModel: 'openai/gpt-oss-120b',
  settingsPrefix: 'CLAUDE_MEM_GROQ',
  envCredentialKey: 'GROQ_API_KEY',
};

export const CCPROXY_OPENROUTER_PROVIDER: OpenAICompatProviderConfig = {
  name: 'CCProxy-OpenRouter',
  apiUrl: resolveOpenRouterUrl('http://127.0.0.1:4000/openrouter/v1/chat/completions'),
  getHeaders: () => ({
    'Content-Type': 'application/json',
  }),
  defaultModel: 'xiaomi/mimo-v2-flash:free',
  settingsPrefix: 'CLAUDE_MEM_OPENROUTER',
  requiresApiKey: false,
};

export const CCPROXY_GROQ_PROVIDER: OpenAICompatProviderConfig = {
  name: 'CCProxy-Groq',
  apiUrl: resolveGroqUrl('http://127.0.0.1:4000/groq/v1/chat/completions'),
  getHeaders: () => ({
    'Content-Type': 'application/json',
  }),
  defaultModel: 'openai/gpt-oss-120b',
  settingsPrefix: 'CLAUDE_MEM_GROQ',
  requiresApiKey: false,
};

// ============================================================================
// OpenAI-compatible message and response types
// ============================================================================

interface OpenAIMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface OpenAICompatResponse {
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

class OpenAICompatApiError extends Error {
  public readonly statusCode: number;
  constructor(statusCode: number, message: string, providerName: string) {
    super(`${providerName} API error: ${statusCode} - ${message}`);
    this.name = 'OpenAICompatApiError';
    this.statusCode = statusCode;
  }
}

// ============================================================================
// Key collection helpers
// ============================================================================

/**
 * Collect all configured API keys for a provider in priority order:
 * 1. Primary key ({prefix}_API_KEY)
 * 2. Pooled keys ({prefix}_API_KEYS, comma-separated)
 * 3. Environment credential (envCredentialKey)
 * Deduplicates while preserving order.
 */
function getAllApiKeys(
  settings: ReturnType<typeof SettingsDefaultsManager.loadFromFile>,
  provider: OpenAICompatProviderConfig
): string[] {
  if (provider.requiresApiKey === false) {
    return ['ccproxy-managed'];  // Sentinel — CCProxy handles credentials
  }

  const keys: string[] = [];
  const seen = new Set<string>();

  const add = (k: string) => {
    const trimmed = k.trim();
    if (trimmed && !seen.has(trimmed)) {
      seen.add(trimmed);
      keys.push(trimmed);
    }
  };

  // Primary key from settings
  const primaryKey = `${provider.settingsPrefix}_API_KEY` as keyof typeof settings;
  add((settings as any)[primaryKey] || '');

  // Pooled keys from settings (comma-separated)
  const poolKey = `${provider.settingsPrefix}_API_KEYS` as keyof typeof settings;
  ((settings as any)[poolKey] || '')
    .split(',')
    .forEach((k: string) => add(k));

  // Environment credential fallback
  if (provider.envCredentialKey) {
    add(getCredential(provider.envCredentialKey as any) || '');
  }

  return keys;
}

/** Return the first available key (used by availability checks). */
function getFirstApiKey(
  settings: ReturnType<typeof SettingsDefaultsManager.loadFromFile>,
  provider: OpenAICompatProviderConfig
): string {
  return getAllApiKeys(settings, provider)[0] || '';
}

// ============================================================================
// OpenAICompatAgent
// ============================================================================

export class OpenAICompatAgent {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;
  private provider: OpenAICompatProviderConfig;
  private fallbackAgent: FallbackAgent | null = null;
  private _nextKeyIndex: number = 0;
  private _keyCooldowns: Map<number, number> = new Map();
  private _waitRetried: boolean = false;
  private static readonly COOLDOWN_MS = 60_000;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager, provider: OpenAICompatProviderConfig) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
    this.provider = provider;
  }

  /**
   * Set the backup agent for when this provider fails
   */
  setFallbackAgent(agent: FallbackAgent): void {
    this.fallbackAgent = agent;
  }

  /**
   * Start agent for a session
   */
  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    try {
      const { apiKeys, model, extraHeaders } = this.getProviderConfig();

      if (apiKeys.length === 0) {
        throw new Error(`${this.provider.name} API key not configured. Set ${this.provider.settingsPrefix}_API_KEY in settings or ${this.provider.envCredentialKey} environment variable.`);
      }

      // Generate synthetic memorySessionId (stateless API, no session IDs returned)
      if (!session.memorySessionId) {
        const syntheticMemorySessionId = `${this.provider.name.toLowerCase()}-${session.contentSessionId}-${Date.now()}`;
        session.memorySessionId = syntheticMemorySessionId;
        this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
        logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=${this.provider.name}`);
      }

      // Load active mode
      const mode = ModeManager.getInstance().getActiveMode();

      // Build initial prompt
      const initPrompt = session.lastPromptNumber === 1
        ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
        : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

      // Add to conversation history and query with full context
      session.conversationHistory.push({ role: 'user', content: initPrompt });
      const initResponse = await this.queryWithRotation(session.conversationHistory, apiKeys, model, extraHeaders);

      if (initResponse.content) {
        session.conversationHistory.push({ role: 'assistant', content: initResponse.content });

        const tokensUsed = initResponse.tokensUsed || 0;
        session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
        session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);

        await processAgentResponse(
          initResponse.content,
          session,
          this.dbManager,
          this.sessionManager,
          worker,
          tokensUsed,
          null,
          this.provider.name,
          undefined
        );
      } else {
        logger.error('SDK', `Empty ${this.provider.name} init response - session may lack context`, {
          sessionId: session.sessionDbId,
          model
        });
      }

      let lastCwd: string | undefined;

      // Process pending messages
      for await (const message of this.sessionManager.getMessageIterator(session.sessionDbId)) {
        session.processingMessageIds.push(message._persistentId);

        if (message.cwd) {
          lastCwd = message.cwd;
        }
        const originalTimestamp = session.earliestPendingTimestamp;

        if (message.type === 'observation') {
          if (message.prompt_number !== undefined) {
            session.lastPromptNumber = message.prompt_number;
          }

          if (!session.memorySessionId) {
            throw new Error('Cannot process observations: memorySessionId not yet captured.');
          }

          const obsPrompt = buildObservationPrompt({
            id: 0,
            tool_name: message.tool_name!,
            tool_input: JSON.stringify(message.tool_input),
            tool_output: JSON.stringify(message.tool_response),
            created_at_epoch: originalTimestamp ?? Date.now(),
            cwd: message.cwd
          });

          session.conversationHistory.push({ role: 'user', content: obsPrompt });
          const obsResponse = await this.queryWithRotation(session.conversationHistory, apiKeys, model, extraHeaders);

          let tokensUsed = 0;
          if (obsResponse.content) {
            session.conversationHistory.push({ role: 'assistant', content: obsResponse.content });
            tokensUsed = obsResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          await processAgentResponse(
            obsResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            this.provider.name,
            lastCwd
          );

          // Reset context: keep only the last exchange to stay within provider limits.
          // Observations are already stored in DB — no context loss.
          if (session.conversationHistory.length > 4) {
            const lastTwo = session.conversationHistory.slice(-2); // last user+assistant pair
            session.conversationHistory = lastTwo;
          }

        } else if (message.type === 'summarize') {
          if (!session.memorySessionId) {
            throw new Error('Cannot process summary: memorySessionId not yet captured.');
          }

          const summaryPrompt = buildSummaryPrompt({
            id: session.sessionDbId,
            memory_session_id: session.memorySessionId,
            project: session.project,
            user_prompt: session.userPrompt,
            last_assistant_message: message.last_assistant_message || ''
          }, mode);

          session.conversationHistory.push({ role: 'user', content: summaryPrompt });
          const summaryResponse = await this.queryWithRotation(session.conversationHistory, apiKeys, model, extraHeaders);

          let tokensUsed = 0;
          if (summaryResponse.content) {
            session.conversationHistory.push({ role: 'assistant', content: summaryResponse.content });
            tokensUsed = summaryResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          await processAgentResponse(
            summaryResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            this.provider.name,
            lastCwd
          );

          // Reset context: keep only the last exchange to stay within provider limits.
          // Summary is already stored in DB — no context loss.
          if (session.conversationHistory.length > 4) {
            const lastTwo = session.conversationHistory.slice(-2); // last user+assistant pair
            session.conversationHistory = lastTwo;
          }
        }
      }

      const sessionDuration = Date.now() - session.startTime;
      logger.success('SDK', `${this.provider.name} agent completed`, {
        sessionId: session.sessionDbId,
        duration: `${(sessionDuration / 1000).toFixed(1)}s`,
        historyLength: session.conversationHistory.length,
        model
      });

    } catch (error: unknown) {
      if (isAbortError(error)) {
        logger.warn('SDK', `${this.provider.name} agent aborted`, { sessionId: session.sessionDbId });
        throw error;
      }

      if (shouldFallbackToClaude(error) && this.fallbackAgent) {
        logger.warn('SDK', `${this.provider.name} API failed, falling back to backup provider`, {
          sessionDbId: session.sessionDbId,
          error: error instanceof Error ? error.message : String(error),
          historyLength: session.conversationHistory.length
        });

        return this.fallbackAgent.startSession(session, worker);
      }

      logger.failure('SDK', `${this.provider.name} agent error`, { sessionDbId: session.sessionDbId }, error as Error);
      throw error;
    }
  }

  private estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  private truncateHistory(history: ConversationMessage[]): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const maxMsgsKey = `${this.provider.settingsPrefix}_MAX_CONTEXT_MESSAGES`;
    const maxToksKey = `${this.provider.settingsPrefix}_MAX_TOKENS`;

    const MAX_CONTEXT_MESSAGES = parseInt((settings as any)[maxMsgsKey]) || DEFAULT_MAX_CONTEXT_MESSAGES;
    const MAX_ESTIMATED_TOKENS = parseInt((settings as any)[maxToksKey]) || DEFAULT_MAX_ESTIMATED_TOKENS;

    if (history.length <= MAX_CONTEXT_MESSAGES) {
      const totalTokens = history.reduce((sum, m) => sum + this.estimateTokens(m.content), 0);
      if (totalTokens <= MAX_ESTIMATED_TOKENS) {
        return history;
      }
    }

    const truncated: ConversationMessage[] = [];
    let tokenCount = 0;

    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      const msgTokens = this.estimateTokens(msg.content);

      if (truncated.length >= MAX_CONTEXT_MESSAGES || tokenCount + msgTokens > MAX_ESTIMATED_TOKENS) {
        logger.warn('SDK', 'Context window truncated to prevent runaway costs', {
          provider: this.provider.name,
          originalMessages: history.length,
          keptMessages: truncated.length,
          droppedMessages: i + 1,
          estimatedTokens: tokenCount,
          tokenLimit: MAX_ESTIMATED_TOKENS
        });
        break;
      }

      truncated.unshift(msg);
      tokenCount += msgTokens;
    }

    return truncated;
  }

  private conversationToOpenAIMessages(history: ConversationMessage[]): OpenAIMessage[] {
    return history.map(msg => ({
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      content: msg.content
    }));
  }

  /**
   * Query with round-robin key rotation and cooldown tracking.
   */
  private async queryWithRotation(
    history: ConversationMessage[],
    apiKeys: string[],
    model: string,
    extraHeaders?: Record<string, string>
  ): Promise<{ content: string; tokensUsed?: number }> {
    let lastError: Error | null = null;
    const startIndex = this._nextKeyIndex % apiKeys.length;
    let attemptsLeft = apiKeys.length;

    for (let attempt = 0; attempt < apiKeys.length; attempt++) {
      const keyIndex = (startIndex + attempt) % apiKeys.length;

      if (this.isKeyInCooldown(keyIndex)) {
        logger.debug('SDK', `${this.provider.name} key ${keyIndex + 1}/${apiKeys.length} in cooldown, skipping`);
        attemptsLeft--;
        continue;
      }

      try {
        const result = await this.queryMultiTurn(history, apiKeys[keyIndex], model, extraHeaders);
        this._nextKeyIndex = (keyIndex + 1) % apiKeys.length;
        // Trickle pacing: delay between requests to avoid burst-exhausting rate limits
        const paceSettings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
        const paceMs = parseInt(paceSettings.CLAUDE_MEM_PROVIDER_PACE_MS) || 2000;
        if (paceMs > 0) {
          await new Promise(resolve => setTimeout(resolve, paceMs));
        }
        return result;
      } catch (err) {
        lastError = err as Error;
        attemptsLeft--;
        const statusCode = this.extractHttpStatus(lastError);
        const isRetryable = statusCode !== null && [401, 402, 413, 429, 504].includes(statusCode);

        if (statusCode === 429) {
          this.cooldownKey(keyIndex);
        }

        // Always advance key index on failure so next call starts at a different key
        this._nextKeyIndex = (keyIndex + 1) % apiKeys.length;

        if (isRetryable && attemptsLeft > 0) {
          logger.warn('SDK', `${this.provider.name} key ${keyIndex + 1}/${apiKeys.length} failed (${statusCode}), rotating to next key`);
          continue;
        }
        throw lastError;
      }
    }

    if (this._keyCooldowns.size >= apiKeys.length) {
      const now = Date.now();
      const shortestWait = Math.min(...Array.from(this._keyCooldowns.values()).map(t => Math.max(t - now, 0)));
      if (!this._waitRetried && shortestWait > 0 && shortestWait <= 65000) {
        this._waitRetried = true;
        logger.info('SDK', `All ${this.provider.name} keys in cooldown, waiting ${Math.ceil(shortestWait / 1000)}s for next available key`);
        await new Promise(resolve => setTimeout(resolve, shortestWait + 500));
        this._keyCooldowns.clear();
        const result = await this.queryWithRotation(history, apiKeys, model, extraHeaders);
        this._waitRetried = false;
        return result;
      }
      this._waitRetried = false;
      logger.warn('SDK', `All ${this.provider.name} keys in cooldown, clearing cooldowns`);
      this._keyCooldowns.clear();
    }

    throw lastError || new Error(`No ${this.provider.name} API keys available`);
  }

  private extractHttpStatus(error: Error): number | null {
    if (error instanceof OpenAICompatApiError) {
      return error.statusCode;
    }
    const match = (error.message || '').match(/\b(4\d{2}|5\d{2})\b/);
    return match ? parseInt(match[1], 10) : null;
  }

  private isKeyInCooldown(keyIndex: number): boolean {
    const cooldownUntil = this._keyCooldowns.get(keyIndex);
    if (!cooldownUntil) return false;
    if (Date.now() >= cooldownUntil) {
      this._keyCooldowns.delete(keyIndex);
      return false;
    }
    return true;
  }

  private cooldownKey(keyIndex: number): void {
    this._keyCooldowns.set(keyIndex, Date.now() + OpenAICompatAgent.COOLDOWN_MS);
    logger.info('SDK', `${this.provider.name} key ${keyIndex + 1} placed in ${OpenAICompatAgent.COOLDOWN_MS / 1000}s cooldown`);
  }

  /**
   * Query provider via REST API with full conversation history
   */
  private async queryMultiTurn(
    history: ConversationMessage[],
    apiKey: string,
    model: string,
    extraHeaders?: Record<string, string>
  ): Promise<{ content: string; tokensUsed?: number }> {
    const truncatedHistory = this.truncateHistory(history);
    const messages = this.conversationToOpenAIMessages(truncatedHistory);
    const totalChars = truncatedHistory.reduce((sum, m) => sum + m.content.length, 0);
    const estimatedTokens = this.estimateTokens(truncatedHistory.map(m => m.content).join(''));

    logger.debug('SDK', `Querying ${this.provider.name} multi-turn (${model})`, {
      turns: truncatedHistory.length,
      totalChars,
      estimatedTokens
    });

    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
    const maxResponseKey = `${this.provider.settingsPrefix}_MAX_RESPONSE_TOKENS`;
    const maxResponseTokens = parseInt((settings as any)[maxResponseKey]) || 1024;

    const headers = this.provider.getHeaders(apiKey, extraHeaders);

    const response = await fetch(this.provider.apiUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.3,
        max_tokens: maxResponseTokens,
        stream: false,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new OpenAICompatApiError(response.status, errorText, this.provider.name);
    }

    const data = await response.json() as OpenAICompatResponse;

    if (data.error) {
      const code = parseInt(data.error.code || '0', 10) || 500;
      throw new OpenAICompatApiError(code, data.error.message || 'Unknown error', this.provider.name);
    }

    if (!data.choices?.[0]?.message?.content) {
      logger.error('SDK', `Empty response from ${this.provider.name}`);
      return { content: '' };
    }

    const content = data.choices[0].message.content;
    const tokensUsed = data.usage?.total_tokens;

    if (tokensUsed) {
      const inputTokens = data.usage?.prompt_tokens || 0;
      const outputTokens = data.usage?.completion_tokens || 0;
      const estimatedCost = (inputTokens / 1000000 * 3) + (outputTokens / 1000000 * 15);

      logger.info('SDK', `${this.provider.name} API usage`, {
        model,
        inputTokens,
        outputTokens,
        totalTokens: tokensUsed,
        estimatedCostUSD: estimatedCost.toFixed(4),
        messagesInContext: truncatedHistory.length
      });

      if (tokensUsed > 50000) {
        logger.warn('SDK', 'High token usage detected - consider reducing context', {
          provider: this.provider.name,
          totalTokens: tokensUsed,
          estimatedCost: estimatedCost.toFixed(4)
        });
      }
    }

    return { content, tokensUsed };
  }

  /**
   * Get provider configuration from settings
   */
  private getProviderConfig(): { apiKeys: string[]; model: string; extraHeaders?: Record<string, string> } {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const apiKeys = getAllApiKeys(settings, this.provider);

    const modelKey = `${this.provider.settingsPrefix}_MODEL`;
    const model = (settings as any)[modelKey] || this.provider.defaultModel;

    // Build extra headers for providers that need them (e.g., OpenRouter analytics)
    const extraHeaders: Record<string, string> = {};
    const siteUrlKey = `${this.provider.settingsPrefix}_SITE_URL`;
    const appNameKey = `${this.provider.settingsPrefix}_APP_NAME`;
    const siteUrl = (settings as any)[siteUrlKey];
    const appName = (settings as any)[appNameKey];
    if (siteUrl) extraHeaders.siteUrl = siteUrl;
    if (appName) extraHeaders.appName = appName;

    return { apiKeys, model, extraHeaders: Object.keys(extraHeaders).length > 0 ? extraHeaders : undefined };
  }
}

// ============================================================================
// Factory functions
// ============================================================================

export function createOpenRouterAgent(dbManager: DatabaseManager, sessionManager: SessionManager): OpenAICompatAgent {
  return new OpenAICompatAgent(dbManager, sessionManager, OPENROUTER_PROVIDER);
}

export function createGroqAgent(dbManager: DatabaseManager, sessionManager: SessionManager): OpenAICompatAgent {
  return new OpenAICompatAgent(dbManager, sessionManager, GROQ_PROVIDER);
}

export function createCCProxyOpenRouterAgent(
  dbManager: DatabaseManager,
  sessionManager: SessionManager
): OpenAICompatAgent {
  return new OpenAICompatAgent(dbManager, sessionManager, CCPROXY_OPENROUTER_PROVIDER);
}

export function createCCProxyGroqAgent(
  dbManager: DatabaseManager,
  sessionManager: SessionManager
): OpenAICompatAgent {
  return new OpenAICompatAgent(dbManager, sessionManager, CCPROXY_GROQ_PROVIDER);
}

// ============================================================================
// Availability / selection checks
// ============================================================================

function isProviderAvailable(provider: OpenAICompatProviderConfig): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return !!getFirstApiKey(settings, provider);
}

function isProviderSelected(providerName: string): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === providerName;
}

export function isOpenRouterAvailable(): boolean {
  return isProviderAvailable(OPENROUTER_PROVIDER);
}

export function isOpenRouterSelected(): boolean {
  return isProviderSelected('openrouter');
}

export function isGroqAvailable(): boolean {
  return isProviderAvailable(GROQ_PROVIDER);
}

export function isGroqSelected(): boolean {
  return isProviderSelected('groq');
}

export function isCCProxyOpenRouterSelected(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === 'ccproxy-openrouter';
}

export function isCCProxyOpenRouterAvailable(): boolean {
  return true;  // CCProxy is local, always available
}

export function isCCProxyGroqSelected(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === 'ccproxy-groq';
}

export function isCCProxyGroqAvailable(): boolean {
  return true;
}
