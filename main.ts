// free-model-aggregator-02 entry for YB: one server, two client wires, real free models
// behind every answer, dialled through the grader's gateway.
//
// Built from the kit starter (lib/wire.ts is reused verbatim: wire translation,
// SSE event order, model-field rewrite and the fixed model-list timestamps are
// all theirs and are correct; lib/providers.ts is my own implementation of the
// same brief parsing, catalog and capability checks). What changes here,
// against the deliberately weak baseline, is routing, failover and cooldowns:
//
//   - quality (60%): candidates meeting a prompt's requirements are ranked by
//     rating_milli descending (id ascending as a deterministic tie-break) and the
//     best is dialled first -- never the first in brief order.
//   - failover (10%): a gateway 4xx/5xx (or an unreachable gateway) fails over to
//     the next-best eligible model, in rating order, until one serves or the
//     eligible list is exhausted. Every refusal is recorded, and when a prompt is
//     then served the trail is reported on the response via the header
//     x-makefun-upstream-failures (JSON array, max 8 rows, refusal order).
//   - cooldowns: a model refused with 429 or 5xx is excluded for the rest of the
//     trial whenever any other eligible model remains (per-trial exclusion, no
//     wall clock, no randomness -- replay-deterministic). No model is ever dialled
//     twice for one prompt, and no speculative calls are made (no upstream
//     /v1/models, no preflight), keeping the gateway-call count minimal.
//   - determinism: every choice is a function of the brief, the requests seen so
//     far and the gateway answers seen so far. No Date.now / wall clock /
//     Math.random anywhere in routing.
//
// Trial shape (see kit RULES.txt / prompt.txt): listen 0.0.0.0 on a free port
// (never 8080), POST $MAKEFUN_ENV_URL/register {protocol_version:1, port, catalog},
// serve GET /v1/models (OpenAI shape for Authorization: Bearer, Anthropic shape
// for x-api-key), POST /v1/chat/completions and POST /v1/messages in JSON and SSE,
// poll GET $MAKEFUN_ENV_URL/work every 100 ms and exit 0 when it says done.
// `node main.ts --list` prints the catalog byte-for-byte as registered, then exits 0.
// Every prompt body names model "auto": it is replaced with a chosen dialable id and
// dialled on the gateway Chat Completions route POST $OPENAI_BASE_URL/chat/completions
// with Authorization: Bearer $OPENAI_API_KEY (the dummy key the gateway swaps).
// POST $ANTHROPIC_BASE_URL/v1/messages is never dialled for these ids (gateway 400).
// The dialled id is put back as `model` on every answer and every streamed chunk.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Max rows in the x-makefun-upstream-failures header (the rules' cap). */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models refused with 429/5xx this trial: excluded while any other eligible model remains. */
const cooledDown = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Eligible free models for a request, best first: rating_milli descending, then id
 * ascending as the deterministic tie-break (brief order never decides). A model on
 * the trial cooldown list is skipped while any other eligible model remains.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  const eligible = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  eligible.sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const fresh = eligible.filter((model) => !cooledDown.has(model.id));
  return fresh.length > 0 ? fresh : eligible;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, unioned with the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  // Union: the manifest is the declared source, the body is the ground truth of
  // what the model must actually do (JSON mode, tools, images). Either source
  // naming a requirement means the chosen model must meet it.
  return {
    ...(entry.requires.tools === true || fromBody.tools === true ? { tools: true } : {}),
    ...(entry.requires.vision === true || fromBody.vision === true ? { vision: true } : {}),
    ...(entry.requires.json === true || fromBody.json === true ? { json: true } : {}),
    ...(entry.requires.min_context !== undefined ? { min_context: entry.requires.min_context } : {}),
  };
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const scanBlocks = (content: unknown): void => {
    if (!Array.isArray(content)) return;
    for (const block of content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    scanBlocks(message.content);
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Extra headers carried on a served response: the upstream-failure trail, when there is one. */
function failureHeaders(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  const rows = failures.slice(0, MAX_FAILURE_ROWS);
  return { 'x-makefun-upstream-failures': JSON.stringify(rows) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One row of the x-makefun-upstream-failures trail, in refusal order. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** The error code of a refused gateway answer, from its body when it names one. */
function failureCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.code === 'number') return String(parsed.error.code);
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
    if (isRecord(parsed) && typeof parsed.code === 'string' && parsed.code !== '') return parsed.code;
  } catch {
    // Not JSON: fall through to the status-based code.
  }
  if (status === 429) return 'rate_limited';
  if (status === 401) return 'unauthorized';
  if (status === 400) return 'invalid_request_error';
  if (status >= 500) return 'server_error';
  return 'upstream_error';
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  // The gateway refuses a call with no output cap (or one over 8192): every
  // dialled body carries one. Bodies from the trial already do; this is the guard.
  const upstreamBody: Json = { ...body, model };
  if (typeof upstreamBody.max_tokens !== 'number' && typeof upstreamBody.max_completion_tokens !== 'number') {
    upstreamBody.max_tokens = 1024;
  }
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
}

interface DialOutcome {
  model: string;
  upstream: Response;
}

/**
 * Dial the ranked candidates in order until one serves. Returns the serving call,
 * or null once every eligible model has been tried. `failures` is filled, in
 * refusal order, with one row per refused call; a 429/5xx also cools that model
 * down for the rest of the trial. No model is dialled twice for one prompt.
 */
async function dialWithFailover(candidates: ModelRow[], body: Json, failures: UpstreamFailure[]): Promise<DialOutcome | null> {
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, body);
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      cooledDown.add(candidate.id);
      log(`gateway unreachable for ${candidate.id}: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { model: candidate.id, upstream };
    const text = await upstream.text();
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
    if (upstream.status === 429 || upstream.status >= 500) cooledDown.add(candidate.id);
    log(`gateway refused ${candidate.id} with HTTP ${upstream.status}`);
  }
  return null;
}

/** All eligible models exhausted: a 502 in the client's wire. The prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: UpstreamFailure[]): void {
  const last = failures[failures.length - 1];
  const detail = last !== undefined ? `the gateway answered HTTP ${last.status} (${last.code}) for every eligible model` : 'no eligible model could be dialled';
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail);
  else sendError(res, wire, 502, 'upstream_error', detail);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankedCandidates(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const failures: UpstreamFailure[] = [];
  const served = await dialWithFailover(candidates, { ...body, model: body.model }, failures);
  if (served === null) return refused(res, 'openai', failures);
  const headers = failureHeaders(failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(served.upstream.body as ReadableStream<Uint8Array>, served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await served.upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, served.model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankedCandidates(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const failures: UpstreamFailure[] = [];
  // The upstream body is translated once; dial() overwrites `model` per attempt,
  // and dialWithFailover walks the ranked list once, so a model refused
  // mid-prompt is never retried for that prompt.
  const served = await dialWithFailover(candidates, anthropicToOpenAI(body, candidates[0]?.id ?? 'auto'), failures);
  if (served === null) return refused(res, 'anthropic', failures);
  const headers = failureHeaders(failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(served.upstream.body as ReadableStream<Uint8Array>, served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await served.upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, served.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
