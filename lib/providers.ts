// Brief reading, model catalog and capability checks for the 02 entry.
//
// The trial brief (the file named by MAKEFUN_PROMPT_FILE) is prose with two JSON
// documents embedded in it. This module pulls those documents out, turns them
// into the model list and the prompt list the entry routes from, and answers
// the two questions routing keeps asking: which models are free, and does a
// model meet what a prompt requires.
//
// The two documents are found by their `schema` member, under this contest's
// prefix:
//
//   - the model list (`free-model-aggregator-02/models/v1`): one row per model,
//     keyed by the id the gateway routes (`openrouter/<vendor>/<model>:free`),
//     carrying the model's public rating in milli-points, its capabilities
//     (tools, vision, JSON mode, context length) and its free flag. A model
//     counts as free unless its row says `free: false`; a paid model is never
//     routed to.
//   - the prompt list (`free-model-aggregator-02/prompts/v1`): the six prompts
//     the house client sends, in order, each with its id, its wire (`openai`
//     or `anthropic`), whether it streams, its weight (1 or 2) and the
//     requirements a model must satisfy to serve it.
//
// A brief may use the sibling contest's spellings instead (`ratings` for the
// model list, `manifest` for the prompt list), and one document may carry both
// lists; documents are recognised by the arrays they hold, never by a key name
// alone. The prose around the documents is never read.
//
// Everything in this module is deterministic: no clock, no randomness. The
// catalog `node main.ts --list` prints and the catalog sent to /register are
// built by the same code path, so the two are byte for byte identical on a
// live pass and on its replay.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** The schema prefix every document in this contest's brief carries. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public endpoint of the provider the free models belong to. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable the entrant's OpenRouter key is pushed under. */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as read here. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/**
 * Scan `text` for JSON objects and return every one that carries a string
 * `schema` member, in the order they appear. A `{` that does not open a
 * parseable object is skipped and scanning resumes just past it, so objects
 * nested inside a document are found as well as the documents themselves.
 */
function extractDocuments(text: string): Json[] {
  const found: Json[] = [];
  let cursor = text.indexOf('{');
  while (cursor !== -1) {
    const close = matchingBrace(text, cursor);
    if (close !== -1) {
      try {
        const parsed: unknown = JSON.parse(text.slice(cursor, close + 1));
        if (isRecord(parsed) && typeof parsed.schema === 'string') {
          found.push(parsed);
        }
      } catch {
        // Not JSON on its own: move on to the next `{`.
      }
    }
    cursor = text.indexOf('{', cursor + 1);
  }
  return found;
}

/**
 * The index of the `}` that closes the object opening at `open`, or -1 when
 * the text runs out first. Characters inside strings do not count, and a
 * backslash inside a string skips the character after it.
 */
function matchingBrace(text: string, open: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i++;
      else if (ch === '"') quoted = false;
      continue;
    }
    if (ch === '"') quoted = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** `value` as a string, or '' when it is not one. */
function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** `value` as a finite number, or null when it is not one. */
function asNumberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** `value` as a boolean, or null when it is not one. */
function asBoolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/**
 * The first document whose schema names this contest and which carries one of
 * `keys` as an array, together with the key it was found under. The sibling
 * contest's spellings still read, because the search is by the arrays a
 * document carries rather than by one fixed name.
 */
function findArrayDocument(
  docs: readonly Json[],
  keys: readonly string[],
): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asString(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/**
 * One model row, or null when the row names no model. The id comes from the
 * row's `id`, or from its `model` under the provider that names it: a row
 * giving the bare OpenRouter id (`vendor/model:free`) while naming
 * `openrouter` is routed as the gateway spells it
 * (`openrouter/vendor/model:free`). The brief lists free models, so a row is
 * free unless it says `free: false`; a paid row is never routed.
 */
function toModelRow(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const bare = asString(value.id) !== '' ? asString(value.id) : asString(value.model);
  if (bare === '') return null;
  const named = asString(value.provider) !== '' ? asString(value.provider) : asString(value.wire);
  const id =
    named === 'openrouter' && !bare.toLowerCase().startsWith('openrouter/')
      ? `openrouter/${bare}`
      : bare;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: asString(value.display_name) !== '' ? asString(value.display_name) : bare,
    rating_milli: asNumberOrNull(value.rating_milli) ?? 0,
    context_length: asNumberOrNull(value.context_length),
    tools: asBoolOrNull(value.tools),
    vision: asBoolOrNull(value.vision),
    json: asBoolOrNull(value.json),
    free: value.free !== false,
  };
}

/** One prompt row, or null when the row names no prompt. */
function toPromptRow(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  const required = isRecord(value.requires) ? value.requires : {};
  const requires: Requirement = {};
  if (required.tools === true) requires.tools = true;
  if (required.vision === true) requires.vision = true;
  if (required.json === true) requires.json = true;
  const minContext = asNumberOrNull(required.min_context);
  if (minContext !== null) requires.min_context = minContext;
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires,
  };
}

/** Read the brief at `path` and split it into its model list and prompt list. */
export function readBrief(path: string): Brief {
  if (path === '') {
    return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = extractDocuments(text);
  const modelsFound = findArrayDocument(docs, ['models', 'ratings']);
  if (modelsFound === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const rows = modelsFound.doc[modelsFound.key] as unknown[];
  const models: ModelRow[] = [];
  for (const row of rows) {
    const model = toModelRow(row);
    if (model !== null) models.push(model);
  }
  const promptsFound = findArrayDocument(docs, ['prompts', 'manifest']);
  if (promptsFound === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const promptRows = promptsFound.doc[promptsFound.key] as unknown[];
  const prompts: PromptRow[] = [];
  for (const row of promptRows) {
    const prompt = toPromptRow(row);
    if (prompt !== null) prompts.push(prompt);
  }
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  const out: ModelRow[] = [];
  for (const model of brief.models) {
    if (model.free) out.push(model);
  }
  return out;
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/**
 * Whether a model's capabilities meet one prompt's requirements. A
 * capability the brief left unknown never meets a requirement for it.
 */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  const checks: Array<[boolean | undefined, boolean | null]> = [
    [requires.tools, caps.tools],
    [requires.vision, caps.vision],
    [requires.json, caps.json],
  ];
  for (const [wanted, has] of checks) {
    if (wanted === true && has !== true) return false;
  }
  if (requires.min_context !== undefined) {
    if (caps.context_length === null || caps.context_length < requires.min_context) return false;
  }
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it
 * serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's
 * trial-local address: the gateway's origin differs between a live pass and
 * its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte.
 *
 * Every free model in the brief is listed — exactly the fourteen dialable ids
 * the catalog column scores — so keep it that way: register this set and keep
 * GET /v1/models (under both auth headers) exactly that set, since listing
 * fewer loses the column as surely as listing more.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const byId = new Map<string, CatalogModel>();
  for (const model of brief.models) {
    if (!model.free || byId.has(model.id)) continue;
    byId.set(model.id, { id: model.id, free: true, context_length: model.context_length });
  }
  const models = [...byId.values()];
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
