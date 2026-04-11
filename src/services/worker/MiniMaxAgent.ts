/**
 * MiniMaxAgent: MiniMax observation extraction via CCProxy (Anthropic wire format)
 *
 * CCProxy exposes MiniMax through an Anthropic-compatible endpoint:
 *   http://127.0.0.1:4000/minimax/v1/messages
 *
 * This is NOT OpenAI-compatible format. Using the OpenAI /chat/completions
 * path causes tool-call XML hallucinations (see project memory:
 * project_minimax_tool_hallucination.md). This agent sends Anthropic Messages
 * API requests directly and parses Anthropic-format responses.
 *
 * CCProxy manages credentials — no API key handling needed here.
 *
 * Responsibility:
 * - Call CCProxy MiniMax endpoint using Anthropic Messages API format
 * - Parse Anthropic-format responses (content[0].text)
 * - Sync to database and Chroma
 * - Support fallback to backup provider on failure
 */

import { buildContinuationPrompt, buildInitPrompt, buildObservationPrompt, buildSummaryPrompt } from '../../sdk/prompts.js';
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

// ============================================================================
// Constants
// ============================================================================

// MiniMax endpoint via CCProxy (Anthropic wire format).
// Override via ANTHROPIC_BASE_URL env var — MiniMax mounts at the /minimax
// prefix on the same CCProxy host, so when ANTHROPIC_BASE_URL is set we
// retarget the Messages endpoint under that host.
// See atl5s LLM routing standard: all providers funnel through CCProxy :4000.
const MINIMAX_API_URL = (() => {
  const base = process.env.ANTHROPIC_BASE_URL;
  if (!base) return 'http://127.0.0.1:4000/minimax/v1/messages';
  return `${base.replace(/\/$/, '')}/minimax/v1/messages`;
})();
const PROVIDER_NAME = 'MiniMax';

const DEFAULT_MAX_CONTEXT_MESSAGES = 20;
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;
const CHARS_PER_TOKEN_ESTIMATE = 4;

// ============================================================================
// Anthropic wire-format types
// ============================================================================

interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface AnthropicResponse {
  content?: Array<{
    type: string;
    text?: string;
  }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
  stop_reason?: string;
  error?: {
    type?: string;
    message?: string;
  };
}

class MiniMaxApiError extends Error {
  public readonly statusCode: number;
  constructor(statusCode: number, message: string) {
    super(`${PROVIDER_NAME} API error: ${statusCode} - ${message}`);
    this.name = 'MiniMaxApiError';
    this.statusCode = statusCode;
  }
}

// ============================================================================
// MiniMaxAgent
// ============================================================================

export class MiniMaxAgent {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;
  private fallbackAgent: FallbackAgent | null = null;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  /**
   * Set the backup agent for when MiniMax fails
   */
  setFallbackAgent(agent: FallbackAgent): void {
    this.fallbackAgent = agent;
  }

  /**
   * Start agent for a session
   */
  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    try {
      const { model, maxResponseTokens } = this.getProviderConfig();

      // Generate synthetic memorySessionId (CCProxy/MiniMax is stateless — no session IDs returned)
      if (!session.memorySessionId) {
        const syntheticMemorySessionId = `minimax-${session.contentSessionId}-${Date.now()}`;
        session.memorySessionId = syntheticMemorySessionId;
        this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
        logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=${PROVIDER_NAME}`);
      }

      // Load active mode
      const mode = ModeManager.getInstance().getActiveMode();

      // Build initial prompt
      const initPrompt = session.lastPromptNumber === 1
        ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
        : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

      // Add to conversation history and query with full context
      session.conversationHistory.push({ role: 'user', content: initPrompt });
      const initResponse = await this.queryMultiTurn(session.conversationHistory, model, maxResponseTokens);

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
          PROVIDER_NAME,
          undefined
        );
      } else {
        logger.error('SDK', `Empty ${PROVIDER_NAME} init response - session may lack context`, {
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
          const obsResponse = await this.queryMultiTurn(session.conversationHistory, model, maxResponseTokens);

          let tokensUsed = 0;
          if (obsResponse.content) {
            session.conversationHistory.push({ role: 'assistant', content: obsResponse.content });
            tokensUsed = obsResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          if (obsResponse.content) {
            await processAgentResponse(
              obsResponse.content,
              session,
              this.dbManager,
              this.sessionManager,
              worker,
              tokensUsed,
              originalTimestamp,
              PROVIDER_NAME,
              lastCwd
            );
          } else {
            logger.warn('SDK', `Empty ${PROVIDER_NAME} observation response, skipping processing to preserve message`, {
              sessionId: session.sessionDbId,
              messageId: session.processingMessageIds[session.processingMessageIds.length - 1]
            });
            // Don't confirm — leave message for stale recovery
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
          const summaryResponse = await this.queryMultiTurn(session.conversationHistory, model, maxResponseTokens);

          let tokensUsed = 0;
          if (summaryResponse.content) {
            session.conversationHistory.push({ role: 'assistant', content: summaryResponse.content });
            tokensUsed = summaryResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          if (summaryResponse.content) {
            await processAgentResponse(
              summaryResponse.content,
              session,
              this.dbManager,
              this.sessionManager,
              worker,
              tokensUsed,
              originalTimestamp,
              PROVIDER_NAME,
              lastCwd
            );
          } else {
            logger.warn('SDK', `Empty ${PROVIDER_NAME} summary response, skipping processing to preserve message`, {
              sessionId: session.sessionDbId,
              messageId: session.processingMessageIds[session.processingMessageIds.length - 1]
            });
            // Don't confirm — leave message for stale recovery
          }
        }
      }

      const sessionDuration = Date.now() - session.startTime;
      logger.success('SDK', `${PROVIDER_NAME} agent completed`, {
        sessionId: session.sessionDbId,
        duration: `${(sessionDuration / 1000).toFixed(1)}s`,
        historyLength: session.conversationHistory.length,
        model
      });

    } catch (error: unknown) {
      if (isAbortError(error)) {
        logger.warn('SDK', `${PROVIDER_NAME} agent aborted`, { sessionId: session.sessionDbId });
        throw error;
      }

      if (shouldFallbackToClaude(error) && this.fallbackAgent) {
        logger.warn('SDK', `${PROVIDER_NAME} API failed, falling back to backup provider`, {
          sessionDbId: session.sessionDbId,
          error: error instanceof Error ? error.message : String(error),
          historyLength: session.conversationHistory.length
        });

        return this.fallbackAgent.startSession(session, worker);
      }

      logger.failure('SDK', `${PROVIDER_NAME} agent error`, { sessionDbId: session.sessionDbId }, error as Error);
      throw error;
    }
  }

  // ============================================================================
  // Private helpers
  // ============================================================================

  private estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  private truncateHistory(history: ConversationMessage[]): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const MAX_CONTEXT_MESSAGES = parseInt(settings.CLAUDE_MEM_MINIMAX_MAX_CONTEXT_MESSAGES) || DEFAULT_MAX_CONTEXT_MESSAGES;
    const MAX_ESTIMATED_TOKENS = parseInt(settings.CLAUDE_MEM_MINIMAX_MAX_TOKENS) || DEFAULT_MAX_ESTIMATED_TOKENS;

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
          provider: PROVIDER_NAME,
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

  private conversationToAnthropicMessages(history: ConversationMessage[]): AnthropicMessage[] {
    return history.map(msg => ({
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      content: msg.content
    }));
  }

  /**
   * Query CCProxy MiniMax endpoint using Anthropic Messages API format.
   * No key rotation needed — CCProxy owns credential management.
   */
  private async queryMultiTurn(
    history: ConversationMessage[],
    model: string,
    maxResponseTokens: number
  ): Promise<{ content: string; tokensUsed?: number }> {
    const truncatedHistory = this.truncateHistory(history);
    const messages = this.conversationToAnthropicMessages(truncatedHistory);
    const totalChars = truncatedHistory.reduce((sum, m) => sum + m.content.length, 0);
    const estimatedTokens = this.estimateTokens(truncatedHistory.map(m => m.content).join(''));

    logger.debug('SDK', `Querying ${PROVIDER_NAME} multi-turn (${model})`, {
      turns: truncatedHistory.length,
      totalChars,
      estimatedTokens
    });

    const response = await fetch(MINIMAX_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        max_tokens: maxResponseTokens,
        messages,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new MiniMaxApiError(response.status, errorText);
    }

    const data = await response.json() as AnthropicResponse;

    if (data.error) {
      throw new MiniMaxApiError(500, data.error.message || 'Unknown error');
    }

    if (!data.content?.[0]?.text) {
      logger.error('SDK', `Empty response from ${PROVIDER_NAME}`);
      return { content: '' };
    }

    const content = data.content[0].text;
    const inputTokens = data.usage?.input_tokens || 0;
    const outputTokens = data.usage?.output_tokens || 0;
    const tokensUsed = inputTokens + outputTokens || undefined;

    if (tokensUsed) {
      logger.info('SDK', `${PROVIDER_NAME} API usage`, {
        model,
        inputTokens,
        outputTokens,
        totalTokens: tokensUsed,
        messagesInContext: truncatedHistory.length
      });

      if (tokensUsed > 50000) {
        logger.warn('SDK', 'High token usage detected - consider reducing context', {
          provider: PROVIDER_NAME,
          totalTokens: tokensUsed
        });
      }
    }

    return { content, tokensUsed };
  }

  /**
   * Get MiniMax configuration from settings
   */
  private getProviderConfig(): { model: string; maxResponseTokens: number } {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const model = settings.CLAUDE_MEM_MINIMAX_MODEL || 'MiniMax-M2.5';
    const maxResponseTokens = parseInt(settings.CLAUDE_MEM_MINIMAX_MAX_RESPONSE_TOKENS) || 1024;

    return { model, maxResponseTokens };
  }
}

// ============================================================================
// Availability / selection checks
// ============================================================================

export function isMiniMaxSelected(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === 'minimax';
}

/**
 * MiniMax is routed via CCProxy — it is always available when CCProxy is
 * running locally on port 4000.  We treat it as unconditionally available
 * (same pattern as isCCProxyOpenRouterAvailable / isCCProxyGroqAvailable).
 */
export function isMiniMaxAvailable(): boolean {
  return true;
}
