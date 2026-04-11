// Backfill all historical Claude Code transcripts into claude-mem
//
// Discovers JSONL transcript files directly from ~/.claude/projects/,
// extracts metadata from the transcripts themselves (not stale indexes),
// classifies via heuristics, and imports via the worker API.
//
// Usage: bun run scripts/backfill-transcripts.ts [options]
//   --dry-run          Show what would be imported without importing
//   --project=<name>   Only backfill sessions for a specific project
//   --limit=<n>        Process at most N sessions
//   --include-plans    Also index ~/.claude/plans/*.md files
//   --verbose          Show per-session details

import { existsSync, readFileSync, readdirSync, statSync } from 'fs';
import { basename, join } from 'path';
import { homedir } from 'os';
import { TranscriptParser } from '../src/utils/transcript-parser.js';

// ── Types ──────────────────────────────────────────────────────────────────

interface DiscoveredSession {
  sessionId: string;
  filePath: string;
  projectDir: string;
}

interface ImportSession {
  content_session_id: string;
  memory_session_id: string;
  project: string;
  user_prompt: string;
  started_at: string;
  started_at_epoch: number;
  completed_at: string | null;
  completed_at_epoch: number | null;
  status: string;
}

interface ImportObservation {
  memory_session_id: string;
  project: string;
  text: string | null;
  type: string;
  title: string;
  subtitle: string | null;
  facts: string | null;
  narrative: string | null;
  concepts: string;
  files_read: string | null;
  files_modified: string | null;
  prompt_number: number;
  discovery_tokens: number;
  created_at: string;
  created_at_epoch: number;
}

interface ImportPrompt {
  content_session_id: string;
  prompt_number: number;
  prompt_text: string;
  created_at: string;
  created_at_epoch: number;
}

interface ImportPayload {
  sessions: ImportSession[];
  observations: ImportObservation[];
  prompts: ImportPrompt[];
  summaries: never[];
}

interface ClassificationResult {
  type: 'bugfix' | 'feature' | 'refactor' | 'discovery' | 'decision' | 'change';
  concepts: string[];
  confidence: number;
}

// ── CLI Args ───────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const DRY_RUN = args.includes('--dry-run');
const VERBOSE = args.includes('--verbose');
const INCLUDE_PLANS = args.includes('--include-plans');
const PROJECT_FILTER = args.find(a => a.startsWith('--project='))?.split('=')[1];
const LIMIT = parseInt(args.find(a => a.startsWith('--limit='))?.split('=')[1] ?? '0', 10);

const WORKER_PORT = process.env.CLAUDE_MEM_WORKER_PORT || '37777';
const WORKER_URL = `http://127.0.0.1:${WORKER_PORT}`;
const CLAUDE_DIR = join(homedir(), '.claude');
const PROJECTS_DIR = join(CLAUDE_DIR, 'projects');
const PLANS_DIR = join(CLAUDE_DIR, 'plans');

// ── Project Name Derivation ────────────────────────────────────────────────

// Claude Code stores projects under ~/.claude/projects/<mangled-path>/
// where mangled-path replaces / with -. We reverse this to get the project name.
// e.g., "-home-n4s5ti-Documents-dev-atl5s" -> "atl5s"
function deriveProjectName(mangledDir: string): string {
  // The mangled name is the full path with / replaced by -
  // Just take the last segment (after the last real path separator equivalent)
  // The pattern is: -home-user-...-projectName
  const parts = mangledDir.split('-').filter(Boolean);
  if (parts.length === 0) return 'unknown-project';
  return parts[parts.length - 1];
}

// ── Heuristic Classification ───────────────────────────────────────────────

const BUG_PATTERNS = /\b(fix|bug|error|crash|broken|fail|issue|regression|patch|hotfix|debug|traceback|exception|segfault|panic|undefined is not|cannot read|null pointer|ENOENT|ECONNREFUSED|TypeError|ReferenceError|SyntaxError)\b/i;
const FEATURE_PATTERNS = /\b(add|create|implement|build|new feature|scaffold|generate|introduce|setup|init|bootstrap)\b/i;
const REFACTOR_PATTERNS = /\b(refactor|rename|extract|move|reorganize|clean ?up|simplify|deduplicate|consolidate|migrate|upgrade|modernize)\b/i;
const DECISION_PATTERNS = /\b(decide|choose|option|approach|architecture|design|trade-?off|versus|vs\.?|alternative|strategy|should we|which)\b/i;

function classifySession(
  firstPrompt: string,
  toolUses: Array<{ name: string; input: any }>,
  assistantText: string,
): ClassificationResult {
  const allText = `${firstPrompt} ${assistantText}`.toLowerCase();
  const concepts: string[] = [];

  const readCount = toolUses.filter(t => t.name === 'Read' || t.name === 'Glob' || t.name === 'Grep').length;
  const editCount = toolUses.filter(t => t.name === 'Edit' || t.name === 'Write').length;
  const totalTools = toolUses.length;

  const filesRead = new Set<string>();
  const filesModified = new Set<string>();
  for (const t of toolUses) {
    if (t.name === 'Read' && t.input?.file_path) filesRead.add(t.input.file_path);
    if (t.name === 'Edit' && t.input?.file_path) filesModified.add(t.input.file_path);
    if (t.name === 'Write' && t.input?.file_path) filesModified.add(t.input.file_path);
  }

  let bugScore = 0;
  let featureScore = 0;
  let refactorScore = 0;
  let discoveryScore = 0;
  let decisionScore = 0;

  if (BUG_PATTERNS.test(allText)) bugScore += 3;
  if (FEATURE_PATTERNS.test(allText)) featureScore += 3;
  if (REFACTOR_PATTERNS.test(allText)) refactorScore += 3;
  if (DECISION_PATTERNS.test(allText)) decisionScore += 2;

  if (totalTools > 0 && readCount / totalTools > 0.7) discoveryScore += 3;
  if (filesModified.size > filesRead.size) featureScore += 2;
  if (editCount > 0 && filesModified.size > 0) {
    if (bugScore > 0) bugScore += 1;
    else refactorScore += 1;
  }

  if (bugScore > 0) concepts.push('problem-solution', 'gotcha');
  if (featureScore > 0) concepts.push('what-changed');
  if (refactorScore > 0) concepts.push('pattern', 'what-changed');
  if (discoveryScore > 0) concepts.push('how-it-works');
  if (decisionScore > 0) concepts.push('trade-off', 'why-it-exists');
  if (concepts.length === 0) concepts.push('what-changed');

  const scores = [
    { type: 'bugfix' as const, score: bugScore },
    { type: 'feature' as const, score: featureScore },
    { type: 'refactor' as const, score: refactorScore },
    { type: 'discovery' as const, score: discoveryScore },
    { type: 'decision' as const, score: decisionScore },
  ];
  scores.sort((a, b) => b.score - a.score);

  if (scores[0].score === 0) {
    return { type: 'change', concepts: ['what-changed'], confidence: 0.3 };
  }

  return {
    type: scores[0].type,
    concepts: [...new Set(concepts)],
    confidence: Math.min(scores[0].score / 5, 1.0),
  };
}

// ── Transcript Processing ──────────────────────────────────────────────────

function extractFilesFromToolUses(toolUses: Array<{ name: string; input: any }>): {
  filesRead: string[];
  filesModified: string[];
} {
  const filesRead = new Set<string>();
  const filesModified = new Set<string>();

  for (const t of toolUses) {
    const p = t.input?.file_path || t.input?.path;
    if (!p || typeof p !== 'string') continue;
    if (t.name === 'Read' || t.name === 'Glob' || t.name === 'Grep') filesRead.add(p);
    else if (t.name === 'Edit' || t.name === 'Write') filesModified.add(p);
  }

  return {
    filesRead: [...filesRead].slice(0, 50),
    filesModified: [...filesModified].slice(0, 50),
  };
}

function extractAssistantText(parser: TranscriptParser): string {
  const entries = parser.getAssistantEntries();
  const texts: string[] = [];
  for (const entry of entries.slice(0, 5)) {
    if (Array.isArray(entry.message.content)) {
      for (const item of entry.message.content) {
        if (item.type === 'text') texts.push(item.text.slice(0, 500));
      }
    }
  }
  return texts.join(' ').slice(0, 2000);
}

// Extract first user prompt text from a parsed transcript
function extractFirstPrompt(parser: TranscriptParser): string {
  const userEntries = parser.getUserEntries();
  for (const ue of userEntries) {
    let text = '';
    if (typeof ue.message.content === 'string') {
      text = ue.message.content;
    } else if (Array.isArray(ue.message.content)) {
      text = ue.message.content
        .filter((item): item is { type: 'text'; text: string } => item.type === 'text')
        .map(item => item.text)
        .join('\n');
    }
    // Strip system-reminder tags
    text = text.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '').trim();
    if (text.length > 3) return text;
  }
  return 'Unknown prompt';
}

// Get timestamps from transcript entries
function extractTimestamps(parser: TranscriptParser): { first: string; last: string } {
  const allEntries = parser.getAllEntries();
  let first = '';
  let last = '';
  for (const e of allEntries) {
    if ('timestamp' in e && e.timestamp) {
      if (!first) first = e.timestamp;
      last = e.timestamp;
    }
  }
  return { first: first || new Date().toISOString(), last: last || first || new Date().toISOString() };
}

function processTranscript(
  discovered: DiscoveredSession,
  projectName: string,
  existingSessionMap: Map<string, string>,
): { session: ImportSession; observations: ImportObservation[]; prompts: ImportPrompt[] } | null {
  let parser: TranscriptParser;
  try {
    parser = new TranscriptParser(discovered.filePath);
  } catch {
    if (VERBOSE) console.log(`  SKIP (parse error): ${discovered.sessionId}`);
    return null;
  }

  const stats = parser.getParseStats();
  if (stats.parsedEntries < 2) {
    if (VERBOSE) console.log(`  SKIP (${stats.parsedEntries} entries): ${discovered.sessionId}`);
    return null;
  }

  // Extract session ID from transcript entries (more reliable than filename)
  const userEntries = parser.getUserEntries();
  const firstUser = userEntries[0];
  const sessionId = (firstUser as any)?.sessionId || discovered.sessionId;

  // Get cwd from transcript to derive project name if needed
  const cwdFromTranscript = (firstUser as any)?.cwd as string | undefined;
  const resolvedProject = cwdFromTranscript ? basename(cwdFromTranscript) : projectName;

  const toolUses = parser.getToolUseHistory();
  const assistantText = extractAssistantText(parser);
  const firstPrompt = extractFirstPrompt(parser);
  const { filesRead, filesModified } = extractFilesFromToolUses(toolUses);
  const tokenUsage = parser.getTotalTokenUsage();
  const timestamps = extractTimestamps(parser);

  const classification = classifySession(firstPrompt, toolUses, assistantText);

  const createdEpoch = new Date(timestamps.first).getTime();
  const modifiedEpoch = new Date(timestamps.last).getTime();

  // Reuse existing memory_session_id if session was already captured by live hooks
  const memSessionId = existingSessionMap.get(sessionId) || `backfill-${sessionId}`;

  const session: ImportSession = {
    content_session_id: sessionId,
    memory_session_id: memSessionId,
    project: resolvedProject,
    user_prompt: firstPrompt.slice(0, 1000),
    started_at: timestamps.first,
    started_at_epoch: createdEpoch,
    completed_at: timestamps.last,
    completed_at_epoch: modifiedEpoch,
    status: 'completed',
  };

  const title = firstPrompt
    .replace(/<[^>]+>/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 120);

  const observation: ImportObservation = {
    memory_session_id: memSessionId,
    project: resolvedProject,
    text: null,
    type: classification.type,
    title: title || 'Untitled session',
    subtitle: `${userEntries.length} prompts, ${toolUses.length} tool uses`,
    facts: JSON.stringify({
      promptCount: userEntries.length,
      toolUseCount: toolUses.length,
      tokensUsed: tokenUsage.inputTokens + tokenUsage.outputTokens,
      confidence: classification.confidence,
      source: 'backfill',
    }),
    narrative: null,
    concepts: JSON.stringify(classification.concepts),
    files_read: filesRead.length > 0 ? JSON.stringify(filesRead) : null,
    files_modified: filesModified.length > 0 ? JSON.stringify(filesModified) : null,
    prompt_number: 1,
    discovery_tokens: tokenUsage.inputTokens + tokenUsage.outputTokens,
    created_at: timestamps.first,
    created_at_epoch: createdEpoch,
  };

  // Extract user prompts
  const prompts: ImportPrompt[] = [];
  for (let i = 0; i < userEntries.length && i < 20; i++) {
    const ue = userEntries[i];
    let promptText = '';
    if (typeof ue.message.content === 'string') {
      promptText = ue.message.content;
    } else if (Array.isArray(ue.message.content)) {
      promptText = ue.message.content
        .filter((item): item is { type: 'text'; text: string } => item.type === 'text')
        .map(item => item.text)
        .join('\n');
    }
    promptText = promptText
      .replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    if (!promptText || promptText.length < 3) continue;

    prompts.push({
      content_session_id: sessionId,
      prompt_number: i + 1,
      prompt_text: promptText.slice(0, 5000),
      created_at: ue.timestamp || timestamps.first,
      created_at_epoch: new Date(ue.timestamp || timestamps.first).getTime(),
    });
  }

  return { session, observations: [observation], prompts };
}

// ── Plan Indexing ──────────────────────────────────────────────────────────

function processPlans(allSessions: ImportSession[]): {
  observations: ImportObservation[];
  planSessions: ImportSession[];
} {
  if (!existsSync(PLANS_DIR)) return { observations: [], planSessions: [] };

  const planFiles = readdirSync(PLANS_DIR).filter(f => f.endsWith('.md'));
  const observations: ImportObservation[] = [];
  const planSessions: ImportSession[] = [];

  const sessionsByTime = allSessions
    .filter(s => s.started_at_epoch > 0)
    .sort((a, b) => a.started_at_epoch - b.started_at_epoch);

  for (const file of planFiles) {
    const fullPath = join(PLANS_DIR, file);
    let content: string;
    try {
      content = readFileSync(fullPath, 'utf-8');
    } catch {
      continue;
    }
    if (content.length < 20) continue;

    const stat = statSync(fullPath);
    const mtime = stat.mtimeMs;
    const mtimeIso = stat.mtime.toISOString();

    const titleMatch = content.match(/^#\s+(.+)$/m);
    const title = titleMatch ? titleMatch[1].trim().slice(0, 120) : file.replace('.md', '');

    const filePathPattern = /(?:^|\s)((?:\/[\w.-]+)+(?:\/[\w.-]+\.[\w]+))/gm;
    const mentionedFiles: string[] = [];
    let match;
    while ((match = filePathPattern.exec(content)) !== null) {
      mentionedFiles.push(match[1]);
    }

    let nearestSession: ImportSession | null = null;
    let minDist = Infinity;
    for (const s of sessionsByTime) {
      const dist = Math.abs(s.started_at_epoch - mtime);
      if (dist < minDist) {
        minDist = dist;
        nearestSession = s;
      }
    }

    const useSession = nearestSession && minDist < 86_400_000;
    const createdAt = useSession ? nearestSession!.started_at : mtimeIso;
    const createdAtEpoch = useSession ? nearestSession!.started_at_epoch : mtime;
    const project = useSession ? nearestSession!.project : 'unknown-project';
    const memorySessionId = useSession ? nearestSession!.memory_session_id : `backfill-plan-${file}`;

    // Create a session for uncorrelated plans so FK constraints are satisfied
    if (!useSession) {
      planSessions.push({
        content_session_id: `plan-${file}`,
        memory_session_id: memorySessionId,
        project,
        user_prompt: `Plan: ${title}`,
        started_at: createdAt,
        started_at_epoch: createdAtEpoch,
        completed_at: createdAt,
        completed_at_epoch: createdAtEpoch,
        status: 'completed',
      });
    }

    observations.push({
      memory_session_id: memorySessionId,
      project,
      text: null,
      type: 'decision',
      title: `[Plan] ${title}`,
      subtitle: `${mentionedFiles.length} files referenced`,
      facts: JSON.stringify({
        planFile: file,
        fileSize: content.length,
        correlatedSession: useSession ? nearestSession!.content_session_id : null,
        timestampSource: useSession ? 'session' : 'mtime',
      }),
      narrative: content.slice(0, 3000),
      concepts: '["trade-off","why-it-exists","pattern"]',
      files_read: null,
      files_modified: mentionedFiles.length > 0 ? JSON.stringify(mentionedFiles.slice(0, 50)) : null,
      prompt_number: 1,
      discovery_tokens: 0,
      created_at: createdAt,
      created_at_epoch: createdAtEpoch,
    });
  }

  return { observations, planSessions };
}

// ── Discovery──────────────────────────────────────────────────────────────

function discoverTranscripts(): Map<string, DiscoveredSession[]> {
  const projectMap = new Map<string, DiscoveredSession[]>();

  if (!existsSync(PROJECTS_DIR)) return projectMap;

  for (const dir of readdirSync(PROJECTS_DIR)) {
    const dirPath = join(PROJECTS_DIR, dir);
    if (!statSync(dirPath).isDirectory()) continue;

    // Find JSONL files directly in this project directory (skip subdirs like subagents/)
    const jsonlFiles = readdirSync(dirPath).filter(f => f.endsWith('.jsonl'));
    if (jsonlFiles.length === 0) continue;

    const projectName = deriveProjectName(dir);
    const sessions: DiscoveredSession[] = jsonlFiles.map(f => ({
      sessionId: f.replace('.jsonl', ''),
      filePath: join(dirPath, f),
      projectDir: dir,
    }));

    const existing = projectMap.get(projectName) || [];
    existing.push(...sessions);
    projectMap.set(projectName, existing);
  }

  return projectMap;
}

// ── Main ───────────────────────────────────────────────────────────────────

async function main() {
  console.log('=== Claude-Mem Transcript Backfill ===');
  console.log(`Worker: ${WORKER_URL}`);
  console.log(`Mode: ${DRY_RUN ? 'DRY RUN' : 'LIVE IMPORT'}`);
  console.log('');

  if (!DRY_RUN) {
    try {
      const health = await fetch(`${WORKER_URL}/api/stats`);
      if (!health.ok) throw new Error(`Worker returned ${health.status}`);
      console.log('Worker: healthy');
    } catch {
      console.error(`Worker not running at ${WORKER_URL}. Start it first.`);
      process.exit(1);
    }
  }

  // Pre-flight: query existing sessions to reuse their memory_session_ids
  // This prevents FK violations when observations reference live-captured sessions
  const existingSessionMap = new Map<string, string>(); // content_session_id -> memory_session_id
  try {
    const { execSync } = await import('child_process');
    const dbPath = join(homedir(), '.claude-mem', 'claude-mem.db');
    const rows = execSync(
      `sqlite3 "${dbPath}" "SELECT content_session_id, memory_session_id FROM sdk_sessions;"`,
      { encoding: 'utf-8', timeout: 10000 },
    ).trim();
    for (const row of rows.split('\n')) {
      const [cid, mid] = row.split('|');
      if (cid && mid) existingSessionMap.set(cid, mid);
    }
    console.log(`Pre-flight: ${existingSessionMap.size} existing sessions mapped`);
  } catch (err) {
    console.log(`Pre-flight: skipped (${err})`);
  }

  // Discover all JSONL files across project directories
  const projectMap = discoverTranscripts();

  let totalFiles = 0;
  for (const sessions of projectMap.values()) totalFiles += sessions.length;
  console.log(`Discovered ${totalFiles} JSONL files across ${projectMap.size} projects\n`);

  const allSessions: ImportSession[] = [];
  const allObservations: ImportObservation[] = [];
  const allPrompts: ImportPrompt[] = [];
  let skipped = 0;
  let errors = 0;
  let processed = 0;

  for (const [projectName, sessions] of projectMap) {
    if (PROJECT_FILTER && projectName !== PROJECT_FILTER) {
      if (VERBOSE) console.log(`  SKIP project: ${projectName} (filter: ${PROJECT_FILTER})`);
      continue;
    }

    console.log(`Project: ${projectName} (${sessions.length} files)`);

    for (const discovered of sessions) {
      if (LIMIT > 0 && processed >= LIMIT) break;

      try {
        const result = processTranscript(discovered, projectName, existingSessionMap);
        if (result) {
          allSessions.push(result.session);
          allObservations.push(...result.observations);
          allPrompts.push(...result.prompts);
          processed++;

          if (VERBOSE) {
            const obs = result.observations[0];
            console.log(`  ${obs.type.padEnd(10)} [${obs.concepts}] ${obs.title.slice(0, 60)}`);
          }
        } else {
          skipped++;
        }
      } catch (err) {
        errors++;
        if (VERBOSE) console.error(`  ERROR: ${discovered.sessionId}: ${err}`);
      }
    }

    if (LIMIT > 0 && processed >= LIMIT) break;
  }

  // Process plans
  let planCount = 0;
  if (INCLUDE_PLANS) {
    console.log(`\nIndexing plans from ${PLANS_DIR}...`);
    const { observations: planObs, planSessions } = processPlans(allSessions);
    planCount = planObs.length;
    allObservations.push(...planObs);
    allSessions.push(...planSessions);
    console.log(`  Found ${planCount} plan files (${planSessions.length} need new sessions)`);
  }

  // Summary
  console.log('\n=== Summary ===');
  console.log(`Sessions processed: ${processed}`);
  console.log(`Sessions skipped:   ${skipped}`);
  console.log(`Errors:             ${errors}`);
  console.log(`Observations:       ${allObservations.length} (${planCount} from plans)`);
  console.log(`User prompts:       ${allPrompts.length}`);

  const typeCounts: Record<string, number> = {};
  for (const obs of allObservations) {
    typeCounts[obs.type] = (typeCounts[obs.type] || 0) + 1;
  }
  console.log('\nObservation types:');
  for (const [type, count] of Object.entries(typeCounts).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${type.padEnd(12)} ${count}`);
  }

  // Project distribution
  const projectCounts: Record<string, number> = {};
  for (const obs of allObservations) {
    projectCounts[obs.project] = (projectCounts[obs.project] || 0) + 1;
  }
  console.log('\nProject distribution:');
  for (const [proj, count] of Object.entries(projectCounts).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${proj.padEnd(25)} ${count}`);
  }

  if (DRY_RUN) {
    console.log('\nDRY RUN — nothing imported. Remove --dry-run to import.');
    return;
  }

  // Phased import: sessions first (to satisfy FK constraints), then observations + prompts
  console.log('\nImporting...');

  const BATCH_SIZE = 50;
  const totalImported = { sessions: 0, observations: 0, prompts: 0 };
  const totalSkippedImport = { sessions: 0, observations: 0, prompts: 0 };

  // Phase 1: Import all sessions
  console.log('  Phase 1: Sessions...');
  for (let i = 0; i < allSessions.length; i += BATCH_SIZE) {
    const payload = {
      sessions: allSessions.slice(i, i + BATCH_SIZE),
      observations: [] as ImportObservation[],
      prompts: [] as ImportPrompt[],
      summaries: [],
    };
    try {
      const response = await fetch(`${WORKER_URL}/api/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        const result = await response.json() as { stats: Record<string, number> };
        totalImported.sessions += result.stats.sessionsImported || 0;
        totalSkippedImport.sessions += result.stats.sessionsSkipped || 0;
      } else {
        console.error(`  Sessions batch ${Math.floor(i / BATCH_SIZE) + 1} failed: ${response.status}`);
      }
    } catch (err) {
      console.error(`  Sessions batch error: ${err}`);
    }
    process.stdout.write(`\r  Sessions: ${Math.min(i + BATCH_SIZE, allSessions.length)}/${allSessions.length}`);
  }
  console.log('');

  // Phase 2: Import observations (FK to sessions now satisfied)
  console.log('  Phase 2: Observations...');
  for (let i = 0; i < allObservations.length; i += BATCH_SIZE) {
    const payload = {
      sessions: [] as ImportSession[],
      observations: allObservations.slice(i, i + BATCH_SIZE),
      prompts: [] as ImportPrompt[],
      summaries: [],
    };
    try {
      const response = await fetch(`${WORKER_URL}/api/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        const result = await response.json() as { stats: Record<string, number> };
        totalImported.observations += result.stats.observationsImported || 0;
        totalSkippedImport.observations += result.stats.observationsSkipped || 0;
      } else {
        const text = await response.text();
        console.error(`  Obs batch ${Math.floor(i / BATCH_SIZE) + 1} failed: ${response.status} ${text}`);
      }
    } catch (err) {
      console.error(`  Obs batch error: ${err}`);
    }
    process.stdout.write(`\r  Observations: ${Math.min(i + BATCH_SIZE, allObservations.length)}/${allObservations.length}`);
  }
  console.log('');

  // Phase 3: Import prompts
  console.log('  Phase 3: Prompts...');
  for (let i = 0; i < allPrompts.length; i += BATCH_SIZE) {
    const payload = {
      sessions: [] as ImportSession[],
      observations: [] as ImportObservation[],
      prompts: allPrompts.slice(i, i + BATCH_SIZE),
      summaries: [],
    };
    try {
      const response = await fetch(`${WORKER_URL}/api/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        const result = await response.json() as { stats: Record<string, number> };
        totalImported.prompts += result.stats.promptsImported || 0;
        totalSkippedImport.prompts += result.stats.promptsSkipped || 0;
      } else {
        console.error(`  Prompts batch ${Math.floor(i / BATCH_SIZE) + 1} failed: ${response.status}`);
      }
    } catch (err) {
      console.error(`  Prompts batch error: ${err}`);
    }
    process.stdout.write(`\r  Prompts: ${Math.min(i + BATCH_SIZE, allPrompts.length)}/${allPrompts.length}`);
  }
  console.log('');

  console.log('\n\n=== Import Results ===');
  console.log(`Sessions:     ${totalImported.sessions} imported, ${totalSkippedImport.sessions} skipped (duplicates)`);
  console.log(`Observations: ${totalImported.observations} imported, ${totalSkippedImport.observations} skipped (duplicates)`);
  console.log(`Prompts:      ${totalImported.prompts} imported, ${totalSkippedImport.prompts} skipped (duplicates)`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
