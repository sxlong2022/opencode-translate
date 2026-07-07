import { createHash, randomUUID } from "node:crypto"
import { readFile } from "node:fs/promises"
import { homedir } from "node:os"
import path from "node:path"
import { setTimeout as sleep } from "node:timers/promises"
import { generateText } from "ai"
import { createCredentialResolver } from "./auth"
import {
  buildAuthUnavailableError,
  type FetchLike,
  normalizeReason,
  PLUGIN_NAME,
  type PluginClientLike,
  type ProviderInfo,
  parseTranslatorModel,
  type ResolvedTranslateOptions,
  unwrapData,
} from "./constants"
import { buildSystemPrompt, buildUserPrompt, buildBatchSystemPrompt, buildBatchUserPrompt, parseBatchSegments, unwrapEchoedTextEnvelope } from "./prompts"

interface OpenCodeProviderConfig {
  baseURL?: string
  apiKey?: string
  [key: string]: unknown
}

interface OpenCodeProviderEntry {
  npm?: string
  name?: string
  options?: OpenCodeProviderConfig
  models?: Record<string, unknown>
  [key: string]: unknown
}

interface OpenCodeConfig {
  provider?: Record<string, OpenCodeProviderEntry>
  [key: string]: unknown
}

function resolveEnvVar(value: string): string {
  // Resolve {env:VAR_NAME} syntax
  const match = value.match(/^\{env:(.+)}$/)
  if (match) {
    const envVar = match[1]
    return process.env[envVar] || ""
  }
  return value
}

// Robust JSONC parser: strips comments and trailing commas safely inside strings.
function stripJsoncComments(content: string): string {
  const result: string[] = []
  let i = 0
  while (i < content.length) {
    const ch = content[i]
    if (ch === '"' || ch === "'") {
      const quote = ch
      result.push(ch)
      i++
      while (i < content.length) {
        const c = content[i]
        result.push(c)
        if (c === '\\') {
          i++
          if (i < content.length) result.push(content[i])
        } else if (c === quote) {
          break
        }
        i++
      }
    } else if (ch === '/' && content[i + 1] === '*') {
      i += 2
      while (i < content.length && !(content[i] === '*' && content[i + 1] === '/')) {
        i++
      }
      if (i < content.length) i += 2
    } else if (ch === '/' && content[i + 1] === '/') {
      while (i < content.length && content[i] !== '\n') i++
    } else {
      result.push(ch)
      i++
    }
  }
  return result.join('')
}

function stripTrailingCommas(content: string): string {
  return content.replace(/,(\s*[}\]])/g, '$1')
}

async function readOpenCodeConfig(): Promise<OpenCodeConfig | undefined> {
  const configDirs: string[] = []
  const envDir = process.env.OPENCODE_CONFIG_DIR
  if (envDir) configDirs.push(envDir)
  let dir = process.cwd()
  while (true) {
    configDirs.push(path.join(dir, '.opencode'))
    const parent = path.dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  configDirs.push(path.join(homedir(), '.config', 'opencode'))
  configDirs.push(path.join(homedir(), '.opencode'))
  for (const configDir of configDirs) {
    for (const ext of ['.json', '.jsonc']) {
      const configPath = path.join(configDir, `opencode${ext}`)
      try {
        let content = await readFile(configPath, 'utf-8')
        if (ext === '.jsonc') {
          content = stripJsoncComments(content)
          content = stripTrailingCommas(content)
        }
        return JSON.parse(content) as OpenCodeConfig
      } catch { /* try next */ }
    }
  }
  return undefined
}

async function resolveProviderFromConfig(
  providerID: string,
): Promise<{ baseURL?: string; apiKey?: string }> {
  const config = await readOpenCodeConfig()
  if (!config?.provider?.[providerID]) return {}

  const provider = config.provider[providerID]
  const result: { baseURL?: string; apiKey?: string } = {}

  if (provider.options?.baseURL) {
    result.baseURL = provider.options.baseURL
  }

  if (provider.options?.apiKey) {
    result.apiKey = resolveEnvVar(provider.options.apiKey)
  }

  return result
}

interface TranslatorDependencies {
  generateTextImpl?: typeof generateText
  sleep?: (ms: number) => Promise<void>
  now?: () => number
  credentialResolver?: ReturnType<typeof createCredentialResolver>
  timeoutMs?: number
}

interface TranslateTextInput {
  text: string
  sourceLanguage: string
  targetLanguage: string
  direction: "inbound" | "outbound"
}

interface TranslateTextsInput {
  texts: readonly string[]
  sourceLanguage: string
  targetLanguage: string
  direction: "inbound" | "outbound"
}

// Hard timeout for a single generateText call. Without this, a stalled
// provider request can block the chat.message hook indefinitely.
const DEFAULT_TRANSLATE_TIMEOUT_MS = 15_000
// Wall-clock budget for an entire translation (primary + fallback combined).
// When a fallbackModel is configured, this is the total time allowed before
// giving up and returning the original text.
const TOTAL_BUDGET_MS = 8_000
// Per-call cap for the primary model when fallback is available — fail fast
// so the fallback gets a chance within the remaining budget.
const PRIMARY_TIMEOUT_MS = 5_000

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs)
    promise.then(
      (value) => {
        clearTimeout(timer)
        resolve(value)
      },
      (error) => {
        clearTimeout(timer)
        reject(error)
      },
    )
  })
}

const providerFactoryCache = new Map<string, unknown>()
let modelsDevCache: Record<string, { api?: string }> | undefined

async function resolveModelsDevBaseURL(providerID: string): Promise<string | undefined> {
  if (!modelsDevCache) {
    try {
      const res = await fetch("https://models.dev/api.json")
      if (res.ok) {
        const data = (await res.json()) as Record<string, { api?: string }>
        modelsDevCache = data
      }
    } catch {
      // Network unavailable - skip
    }
  }
  return modelsDevCache?.[providerID]?.api
}

export function __resetTranslatorCachesForTest() {
  providerFactoryCache.clear()
  modelsDevCache = undefined
}

function getStatus(error: unknown): number | undefined {
  if (!error || typeof error !== "object") return undefined
  const record = error as Record<string, unknown>
  if (typeof record.status === "number") return record.status
  if (typeof record.statusCode === "number") return record.statusCode
  const response = record.response
  if (response && typeof response === "object") {
    const status = (response as Record<string, unknown>).status
    if (typeof status === "number") return status
  }
  return undefined
}

function getRetryAfterMs(error: unknown): number {
  if (!error || typeof error !== "object") return 2000
  const record = error as Record<string, unknown>
  const response = record.response
  if (!response || typeof response !== "object") return 2000
  const headers = (response as { headers?: Headers }).headers
  if (!(headers instanceof Headers)) return 2000
  const retryAfter = headers.get("retry-after")
  if (!retryAfter) return 2000
  const seconds = Number(retryAfter)
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000)
  const date = Date.parse(retryAfter)
  return Number.isFinite(date) ? Math.max(0, date - Date.now()) : 2000
}

function isRetryable(error: unknown): boolean {
  const status = getStatus(error)
  if (status === 429) return true
  if (status !== undefined) return status >= 500
  const message = normalizeReason(error).toLowerCase()
  return (
    message.includes("network") ||
    message.includes("fetch") ||
    message.includes("timeout") ||
    message.includes("socket") ||
    message.includes("econn")
  )
}

async function withRetry<T>(task: () => Promise<T>, sleepImpl: (ms: number) => Promise<void>): Promise<T> {
  let lastError: unknown
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      return await task()
    } catch (error) {
      lastError = error
      if (!isRetryable(error)) throw error
      if (getStatus(error) === 429) {
        if (attempt >= 1) throw error
        await sleepImpl(getRetryAfterMs(error))
        continue
      }
      if (attempt >= 2) throw error
      await sleepImpl(attempt === 0 ? 500 : 1500)
    }
  }
  throw lastError
}

async function loadFactory(providerID: string): Promise<unknown> {
  const cached = providerFactoryCache.get(providerID)
  if (cached) return cached

  let factory: unknown
  if (providerID === "anthropic") {
    const mod = await import("@ai-sdk/anthropic")
    factory = mod.createAnthropic ?? mod.anthropic
  } else if (providerID === "openai") {
    const mod = await import("@ai-sdk/openai")
    factory = mod.createOpenAI ?? mod.openai
  } else if (providerID === "google") {
    const mod = await import("@ai-sdk/google")
    factory = mod.createGoogleGenerativeAI ?? mod.google
  } else if (providerID === "google-vertex" || providerID === "amazon-bedrock") {
    // These providers require non-standard credentials (GCP project/location or AWS
    // region/access keys) that cannot be inferred from a simple apiKey. Fail fast
    // with a clear message rather than silently falling back to openai-compatible.
    throw new Error(
      `Provider "${providerID}" requires credentials that opencode-translate cannot resolve automatically. ` +
        `Use a different translatorModel (e.g. anthropic/..., openai/..., or a custom openai-compatible provider).`,
    )
  } else if (providerID === "github-copilot" || providerID === "openai-compatible") {
    const mod = await import("@ai-sdk/openai-compatible")
    factory = mod.createOpenAICompatible
  } else {
    // Unknown provider: treat as openai-compatible (custom endpoints like LongCat, SiliconFlow)
    const mod = await import("@ai-sdk/openai-compatible")
    factory = mod.createOpenAICompatible
  }

  if (typeof factory !== "function") {
    throw new Error(`Unable to load provider factory for "${providerID}"`)
  }

  providerFactoryCache.set(providerID, factory)
  return factory
}

function instantiateProvider(
  factory: unknown,
  providerID: string,
  credentials: { apiKey?: string; fetch?: FetchLike; baseURL?: string },
): unknown {
  if (typeof factory !== "function") throw new Error(`Invalid provider factory for "${providerID}"`)

  const config = {
    ...(credentials.apiKey !== undefined ? { apiKey: credentials.apiKey } : {}),
    ...(credentials.fetch ? { fetch: credentials.fetch } : {}),
    ...(credentials.baseURL ? { baseURL: credentials.baseURL } : {}),
  }

  if (providerID === "github-copilot") {
    return (factory as (config: Record<string, unknown>) => unknown)({
      ...config,
      name: "github-copilot",
      baseURL: "https://api.githubcopilot.com",
    })
  }

  if (providerID === "openai-compatible") {
    if (!credentials.baseURL) {
      throw new Error(`openai-compatible provider requires a baseURL option`)
    }
    return (factory as (config: Record<string, unknown>) => unknown)({
      ...config,
      name: "openai-compatible",
      baseURL: credentials.baseURL,
    })
  }

  // Unknown provider: treated as openai-compatible (custom endpoints like LongCat, SiliconFlow).
  // Native SDK providers (anthropic, openai, google, etc.) have their own baseURL defaults,
  // so only require baseURL for genuinely custom/unknown providers.
  const NATIVE_PROVIDERS = new Set(["anthropic", "openai", "google", "google-vertex", "amazon-bedrock"])
  if (!NATIVE_PROVIDERS.has(providerID) && !credentials.baseURL) {
    throw new Error(`Custom provider "${providerID}" requires a baseURL. Add it under provider.${providerID}.options.baseURL in opencode.json.`)
  }
  return (factory as (config: Record<string, unknown>) => unknown)({
    ...config,
    ...(!NATIVE_PROVIDERS.has(providerID) ? { name: providerID } : {}),
  })
}

function instantiateModel(provider: unknown, modelID: string): unknown {
  if (typeof provider === "function") return provider(modelID)
  if (provider && typeof provider === "object") {
    const record = provider as Record<string, unknown>
    if (typeof record.chatModel === "function") return (record.chatModel as (id: string) => unknown)(modelID)
    if (typeof record.languageModel === "function") {
      return (record.languageModel as (id: string) => unknown)(modelID)
    }
  }
  throw new Error(`Unable to instantiate model "${modelID}"`)
}

function isAuthMessage(error: unknown): boolean {
  if (!(error instanceof Error)) return false
  return error.message.includes(":AUTH_UNAVAILABLE]") || error.message.includes(":OAUTH_REFRESH_FAILED]")
}

function modelProviderHint(providerID: string, provider?: ProviderInfo): Error {
  return buildAuthUnavailableError(providerID, provider?.env[0] || "the provider's API key env var")
}

export function hashText(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex").slice(0, 16)
}

export function createSyntheticPartID(): string {
  return `prt_${randomUUID().replaceAll("-", "")}`
}

export function createTranslator(
  client: PluginClientLike,
  options: ResolvedTranslateOptions,
  deps: TranslatorDependencies = {},
) {
  // Bypasses proxy for Gitee AI (fallback) domains
  const bypassHosts = ["ai.gitee.com", "gitee.com"];
  for (const envKey of ["NO_PROXY", "no_proxy"]) {
    const current = process.env[envKey];
    if (current) {
      const existing = current.split(",").map((d) => d.trim()).filter(Boolean);
      for (const host of bypassHosts) {
        if (!existing.includes(host)) {
          existing.push(host);
        }
      }
      process.env[envKey] = existing.join(",");
    } else {
      process.env[envKey] = bypassHosts.join(",");
    }
  }

  const sleepImpl = deps.sleep ?? ((ms: number) => sleep(ms))
  const now = deps.now ?? (() => Date.now())
  const generateTextImpl = deps.generateTextImpl ?? generateText
  const credentialResolver = deps.credentialResolver ?? createCredentialResolver(client, options)
  const timeoutMs = deps.timeoutMs ?? DEFAULT_TRANSLATE_TIMEOUT_MS

  async function resolveBaseURL(providerID: string): Promise<string | undefined> {
    // Explicit baseURL in plugin options takes precedence
    if (options.baseURL) return options.baseURL

    // Try opencode.json first (custom providers like LongCat, SiliconFlow)
    const config = await resolveProviderFromConfig(providerID)
    if (config.baseURL) return config.baseURL

    // Fallback: try OpenCode's provider configuration (official providers)
    try {
      const providers = unwrapData(await client.provider.list({ throwOnError: true }))
      const providerInfo = providers.all.find((p) => p.id === providerID)
      if (providerInfo?.options?.baseURL) {
        return providerInfo.options.baseURL as string
      }
    } catch {
      // Ignore errors from provider list
    }

    // Final fallback: models.dev catalog (covers built-in openai-compatible providers
    // like nvidia whose endpoint is not exposed in provider.list() options)
    return resolveModelsDevBaseURL(providerID)
  }

  async function resolveApiKey(providerID: string): Promise<string | undefined> {
    // Explicit apiKey in plugin options takes precedence
    if (options.apiKey) return options.apiKey

    // Try opencode.json first (custom providers like LongCat, SiliconFlow)
    const config = await resolveProviderFromConfig(providerID)
    if (config.apiKey) return config.apiKey

    // Fallback: try OpenCode's provider configuration (official providers)
    try {
      const providers = unwrapData(await client.provider.list({ throwOnError: true }))
      const providerInfo = providers.all.find((p) => p.id === providerID)
      if (providerInfo?.key) {
        return providerInfo.key
      }
    } catch {
      // Ignore errors from provider list
    }

    // Fallback: check common env var patterns
    const envKey = `${providerID.toUpperCase().replace(/-/g, "_")}_API_KEY`
    if (process.env[envKey]) {
      return process.env[envKey]
    }
    return undefined
  }

  async function translateText(input: TranslateTextInput): Promise<{ text: string; modelUsed: string }> {
    if (!input.text) return { text: input.text, modelUsed: options.translatorModel }
    if (input.sourceLanguage === input.targetLanguage) return { text: input.text, modelUsed: options.translatorModel }

    const startedAt = now()

    async function attemptTranslation(modelString: string, { retry = true, deadline = Infinity } = {}): Promise<string> {
      const { providerID, modelID } = parseTranslatorModel(modelString)
const credentials = await credentialResolver.resolve(modelString)
const resolvedBaseURL = await resolveBaseURL(providerID)
      const resolvedApiKey = !credentials.apiKey ? await resolveApiKey(providerID) : undefined

      const baseFetch = credentials.fetch ?? fetch
      const robustFetch: FetchLike = async (input, init) => {
        let lastError: unknown
        for (let attempt = 1; attempt <= 3; attempt++) {
          try {
return await baseFetch(input, {
...init,
              // @ts-ignore
              ...(options.verbose ? {
              verbose: true } : {
            }),
            })
          } catch (error) {
            lastError = error
            const msg = String(error).toLowerCase()
            const isNetworkError =
              msg.includes("socket") ||
              msg.includes("fetch") ||
              msg.includes("network") ||
              msg.includes("econn") ||
              msg.includes("closed")
            if (!isNetworkError || attempt === 3) {
              throw error
            }
            if (options.verbose) {
              await client.app.log({
                body: {
                  service: PLUGIN_NAME,
                  level: "warn",
                  message: `Fetch failed (attempt ${attempt}/3): ${msg}. Retrying...`,
                },
              })
            }
            await sleepImpl(500 * Math.pow(2, attempt - 1))
          }
        }
        throw lastError
      }
const factory = await loadFactory(providerID)
const provider = instantiateProvider(factory, providerID, {
        ...credentials,
        fetch: robustFetch,
...(resolvedApiKey ? { apiKey: resolvedApiKey } : {}),
baseURL: resolvedBaseURL,
})
      const model = instantiateModel(provider, modelID)

      const modelProviderOptions =
        modelString === options.translatorModel
          ? (options.translatorProviderOptions ?? options.providerOptions)
          : (options.fallbackProviderOptions ?? options.providerOptions)

      const doTranslate = async () => {
        try {
          const result = (await withTimeout(
            generateTextImpl({
              model: model as never,
              system: buildSystemPrompt({
                sourceLanguage: input.sourceLanguage,
                targetLanguage: input.targetLanguage,
                text: input.text,
              }),
              temperature: 0,
              prompt: buildUserPrompt({
                sourceLanguage: input.sourceLanguage,
                targetLanguage: input.targetLanguage,
                text: input.text,
              }),
              maxRetries: 0,
              ...(modelProviderOptions ? { providerOptions: modelProviderOptions as never } : {}),
            }) as Promise<{ text: string }>,
            retry ? Math.min(timeoutMs, Math.max(0, deadline - now())) : Math.min(PRIMARY_TIMEOUT_MS, Math.max(0, deadline - now())),
            "Translator generateText",
          )) as { text: string }
          return unwrapEchoedTextEnvelope(result.text)
        } catch (error) {
          if (isAuthMessage(error)) throw error
          if (credentials.mode === "default" && credentialResolver.isMissingCredentialError(error)) {
            throw modelProviderHint(providerID, credentials.provider)
          }
          throw error
        }
      }

      if (retry) {
        const remaining = deadline - now()
        if (remaining <= 0) throw new Error("Translation budget exhausted before retry")
        return withRetry(doTranslate, sleepImpl)
      }
      return doTranslate()
    }

    const deadline = options.fallbackModel ? now() + TOTAL_BUDGET_MS : Infinity
    let translated: string
    let usedModel = options.translatorModel
    try {
      // Primary: fast-fail when fallback is available, so fallback gets remaining budget
      translated = await attemptTranslation(options.translatorModel, { retry: !options.fallbackModel, deadline })
    } catch (primaryError) {
      if (options.fallbackModel) {
        const remaining = deadline - now()
        if (remaining <= 500) {
          // Not enough time for a meaningful fallback attempt — return original text
          return { text: input.text, modelUsed: "passthrough" }
        }
        usedModel = options.fallbackModel
        try {
          // Fallback: single attempt within remaining budget (total deadline is the safety net)
          translated = await attemptTranslation(options.fallbackModel, { retry: false, deadline })
        } catch {
          // Both models failed within budget — return original text
          return { text: input.text, modelUsed: "passthrough" }
        }
      } else {
        throw primaryError
      }
    }

    if (options.verbose) {
      await client.app.log({
        body: {
          service: PLUGIN_NAME,
          level: "info",
          message: "translated",
          extra: {
            direction: input.direction,
            chars_in: input.text.length,
            chars_out: translated.length,
            ms: now() - startedAt,
            cached: false,
            model: usedModel,
          },
        },
      })
    }

    return { text: translated, modelUsed: usedModel }
  }

  async function translateTexts(
    input: TranslateTextsInput
  ): Promise<{ texts: string[]; modelUsed: string }> {
    if (input.texts.length === 0) return { texts: [], modelUsed: options.translatorModel }
    if (input.sourceLanguage === input.targetLanguage) return { texts: [...input.texts], modelUsed: options.translatorModel }

    const startedAt = now()

    async function attemptBatchTranslation(modelString: string, { retry = true, deadline = Infinity } = {}): Promise<string[]> {
      const { providerID, modelID } = parseTranslatorModel(modelString)
      const credentials = await credentialResolver.resolve(modelString)
      const resolvedBaseURL = await resolveBaseURL(providerID)
      const resolvedApiKey = !credentials.apiKey ? await resolveApiKey(providerID) : undefined

      const baseFetch = credentials.fetch ?? fetch
      const robustFetch: FetchLike = async (fetchInput, init) => {
        let lastError: unknown
        for (let attempt = 1; attempt <= 3; attempt++) {
          try {
            return await baseFetch(fetchInput, {
              ...init,
              // @ts-ignore
              ...(options.verbose ? { verbose: true } : {}),
            })
          } catch (error) {
            lastError = error
            const msg = String(error).toLowerCase()
            const isNetworkError =
              msg.includes("socket") ||
              msg.includes("fetch") ||
              msg.includes("network") ||
              msg.includes("econn") ||
              msg.includes("closed")
            if (!isNetworkError || attempt === 3) {
              throw error
            }
            if (options.verbose) {
              await client.app.log({
                body: {
                  service: PLUGIN_NAME,
                  level: "warn",
                  message: `Fetch failed (attempt ${attempt}/3): ${msg}. Retrying...`,
                },
              })
            }
            await sleepImpl(500 * Math.pow(2, attempt - 1))
          }
        }
        throw lastError
      }
      const factory = await loadFactory(providerID)
      const provider = instantiateProvider(factory, providerID, {
        ...credentials,
        fetch: robustFetch,
        ...(resolvedApiKey ? { apiKey: resolvedApiKey } : {}),
        baseURL: resolvedBaseURL,
      })
      const model = instantiateModel(provider, modelID)

      const modelProviderOptions =
        modelString === options.translatorModel
          ? (options.translatorProviderOptions ?? options.providerOptions)
          : (options.fallbackProviderOptions ?? options.providerOptions)

      const doTranslate = async () => {
        try {
          const result = (await withTimeout(
            generateTextImpl({
              model: model as never,
              system: buildBatchSystemPrompt({
                sourceLanguage: input.sourceLanguage,
                targetLanguage: input.targetLanguage,
                texts: input.texts,
              }),
              temperature: 0,
              prompt: buildBatchUserPrompt({
                texts: input.texts,
              }),
              maxRetries: 0,
              ...(modelProviderOptions ? { providerOptions: modelProviderOptions as never } : {}),
            }) as Promise<{ text: string }>,
            retry ? Math.min(timeoutMs, Math.max(0, deadline - now())) : Math.min(PRIMARY_TIMEOUT_MS, Math.max(0, deadline - now())),
            "Translator generateText (batch)",
          )) as { text: string }
          return parseBatchSegments(result.text, input.texts.length).map(unwrapEchoedTextEnvelope)
        } catch (error) {
          if (isAuthMessage(error)) throw error
          if (credentials.mode === "default" && credentialResolver.isMissingCredentialError(error)) {
            throw modelProviderHint(providerID, credentials.provider)
          }
          throw error
        }
      }

      if (retry) {
        const remaining = deadline - now()
        if (remaining <= 0) throw new Error("Translation budget exhausted before retry")
        return withRetry(doTranslate, sleepImpl)
      }
      return doTranslate()
    }
    const deadline = options.fallbackModel ? now() + TOTAL_BUDGET_MS : Infinity
    let translated: string[]
    let usedModel = options.translatorModel
    try {
      translated = await attemptBatchTranslation(options.translatorModel, { retry: !options.fallbackModel, deadline })
    } catch (primaryError) {
      if (options.fallbackModel) {
        const remaining = deadline - now()
        if (remaining <= 500) {
          return { texts: [...input.texts], modelUsed: "passthrough" }
        }
        usedModel = options.fallbackModel
        try {
          translated = await attemptBatchTranslation(options.fallbackModel, { retry: false, deadline })
        } catch {
          return { texts: [...input.texts], modelUsed: "passthrough" }
        }
      } else {
        throw primaryError
      }
    }

    if (options.verbose) {
      await client.app.log({
        body: {
          service: PLUGIN_NAME,
          level: "info",
          message: "translated",
          extra: {
            direction: input.direction,
            chars_in: input.texts.reduce((sum, t) => sum + t.length, 0),
            chars_out: translated.reduce((sum, t) => sum + t.length, 0),
            ms: now() - startedAt,
            cached: false,
            model: usedModel,
          },
        },
      })
    }

    return { texts: translated, modelUsed: usedModel }
  }

  return {
    translateText,
    translateTexts,
  }
}
