import { createOpenAICompatible } from "@ai-sdk/openai-compatible"
import { generateText } from "ai"
import { mkdir, readFile, writeFile } from "node:fs/promises"
import path from "node:path"
import type { PluginClientLike, ResolvedTranslateOptions } from "../src/constants.ts"
import { parseTranslatorModel } from "../src/constants.ts"
import { createTranslator } from "../src/translator.ts"

const CONFIG_PATH = "C:\\Users\\sxlon\\.config\\opencode\\opencode.json"
const RESULTS_DIR = "benchmark-results"

interface TestSentence {
  id: string
  source: string
  reference: string
}

const SENTENCES: TestSentence[] = [
  {
    id: "simple",
    source: "你好，请帮我总结一下这份文档。",
    reference: "Hello, please help me summarize this document.",
  },
  {
    id: "technical",
    source: "这个函数的时间复杂度是O(n log n)，因为使用了归并排序。",
    reference: "This function has a time complexity of O(n log n) because it uses merge sort.",
  },
  {
    id: "idiom",
    source: "授人以鱼不如授人以渔。",
    reference: "Give a man a fish and you feed him for a day; teach a man to fish and you feed him for a lifetime.",
  },
  {
    id: "code-mixed",
    source: "我需要用React和TypeScript重构这个组件，确保类型安全。",
    reference: "I need to refactor this component using React and TypeScript to ensure type safety.",
  },
  {
    id: "formal",
    source: "关于下周的项目进度会议，请准备一份详细的时间表。",
    reference: "Regarding next week's project progress meeting, please prepare a detailed schedule.",
  },
  {
    id: "long-technical",
    source: "请分析这段Python代码中的内存泄漏问题，并提出具体的优化建议，例如使用生成器或弱引用。",
    reference:
      "Please analyze the memory leak issue in this Python code and propose specific optimization suggestions, such as using generators or weak references.",
  },
]

interface CandidateModel {
  model: string
  providerOptions: Record<string, unknown>
}

const CANDIDATES: CandidateModel[] = [
  {
    model: "cloudflare-ob/@cf/google/gemma-4-26b-a4b-it",
    providerOptions: {
      reasoning_effort: null,
      chat_template_kwargs: { enable_thinking: false, thinking: false },
    },
  },
  {
    model: "gitee/internlm3-8b-instruct",
    providerOptions: {},
  },
  {
    model: "nvidia-nim-ob/nvidia/riva-translate-4b-instruct-v2",
    providerOptions: {
      reasoning_effort: "none",
      chat_template_kwargs: { enable_thinking: false, thinking: false },
    },
  },
  {
    model: "nvidia-nim-ob/google/diffusiongemma-26b-a4b-it",
    providerOptions: {
      reasoning_effort: "none",
      chat_template_kwargs: { enable_thinking: false, thinking: false },
    },
  },
  {
    model: "internlm/intern-s1-mini",
    providerOptions: {},
  },
]

interface RawResult {
  model: string
  sentenceId: string
  run: number
  success: boolean
  latencyMs: number
  translation?: string
  error?: string
}

interface QualityScore {
  accuracy: number
  fluency: number
  terminology: number
  overall: number
  justification: string
}

interface ScoredResult extends RawResult {
  scores?: QualityScore
  judgeError?: string
}

async function readConfig(): Promise<Record<string, unknown>> {
  const raw = await readFile(CONFIG_PATH, "utf-8")
  return JSON.parse(raw) as Record<string, unknown>
}

function fakeClient(): PluginClientLike {
  return {
    session: {
      get: async () => ({ id: "ses_benchmark", parentID: null }),
      messages: async () => [],
      message: async () => undefined as never,
    },
    provider: { list: async () => ({ all: [] }) },
    auth: { set: async () => undefined },
    app: { log: async () => undefined },
  }
}

function resolveEnvVar(value: string): string {
  const match = value.match(/^\{env:(.+)}$/)
  if (match) return process.env[match[1]] || ""
  return value
}

function resolveProviderConfig(config: Record<string, unknown>, providerID: string): { baseURL?: string; apiKey?: string } {
  const providers = config.provider as Record<string, { options?: { baseURL?: string; apiKey?: string } }> | undefined
  const entry = providers?.[providerID]
  return { baseURL: entry?.options?.baseURL, apiKey: entry?.options?.apiKey ? resolveEnvVar(entry.options.apiKey) : undefined }
}


async function translateOnce(
  model: string,
  source: string,
  providerOptions: Record<string, unknown>,
  config: Record<string, unknown>,
): Promise<{ success: true; latencyMs: number; translation: string } | { success: false; latencyMs: number; error: string }> {
  const { providerID } = parseTranslatorModel(model)
  const { apiKey, baseURL } = resolveProviderConfig(config, providerID)
  const translator = createTranslator(
    fakeClient(),
    {
      translatorModel: model,
      triggerKeywords: ["$en"],
      sourceLanguage: "zh",
      displayLanguage: "zh",
      disableKeywords: ["$dis"],
      translateResponses: false,
      verbose: false,
      translatorProviderOptions: providerOptions,
    } as ResolvedTranslateOptions,
    {
      timeoutMs: 15_000,
      credentialResolver: {
        resolve: async () => ({
          providerID,
          apiKey: apiKey ?? "",
          mode: "apiKey" as const,
          provider: {
            id: providerID,
            source: "api" as const,
            env: [],
            options: baseURL ? { baseURL } : undefined,
          },
        }),
        isMissingCredentialError: () => false,
        authUnavailable: () => new Error("auth unavailable"),
        envFallback: "API_KEY",
      },
    },
  )
  const start = performance.now()
  try {
    const result = await translator.translateText({
      text: source,
      sourceLanguage: "zh",
      targetLanguage: "en",
      direction: "inbound",
    })
    return { success: true, latencyMs: performance.now() - start, translation: result.text }
  } catch (error) {
    return { success: false, latencyMs: performance.now() - start, error: String(error) }
  }
}

async function benchmark(config: Record<string, unknown>): Promise<RawResult[]> {
  const results: RawResult[] = []
  for (const { model, providerOptions } of CANDIDATES) {
    console.log(`\n[Model] ${model}`)
    for (const sentence of SENTENCES) {
      process.stdout.write(`  ${sentence.id}: `)
      for (let run = 0; run < 1; run++) {
        const outcome = await translateOnce(model, sentence.source, providerOptions, config)
        if (outcome.success) {
          results.push({
            model,
            sentenceId: sentence.id,
            run,
            success: true,
            latencyMs: Math.round(outcome.latencyMs),
            translation: outcome.translation,
          })
          process.stdout.write(outcome.latencyMs > 10_000 ? "S(slow) " : "OK ")
        } else {
          results.push({
            model,
            sentenceId: sentence.id,
            run,
            success: false,
            latencyMs: Math.round(outcome.latencyMs),
            error: outcome.error,
          })
          process.stdout.write("FAIL ")
          // One immediate retry
          const retry = await translateOnce(model, sentence.source, providerOptions, config)
          if (retry.success) {
            results.push({
              model,
              sentenceId: sentence.id,
              run: run + 1,
              success: true,
              latencyMs: Math.round(retry.latencyMs),
              translation: retry.translation,
            })
            process.stdout.write("retry-OK ")
          } else {
            results.push({
              model,
              sentenceId: sentence.id,
              run: run + 1,
              success: false,
              latencyMs: Math.round(retry.latencyMs),
              error: retry.error,
            })
            process.stdout.write("retry-FAIL ")
          }
        }
      }
      process.stdout.write("\n")
      await mkdir(RESULTS_DIR, { recursive: true })
      await writeFile(
        path.join(RESULTS_DIR, "latest-raw.json"),
        JSON.stringify({ candidates: CANDIDATES.map((c) => c.model), sentences: SENTENCES, results }, null, 2),
      )
    }
  }
  return results
}

function buildJudgePrompt(source: string, reference: string, candidate: string): string {
  return [
    "You are evaluating Chinese-to-English translations. Score the candidate translation against the source Chinese and reference English translation.",
    "",
    "Source Chinese:",
    source,
    "",
    "Reference English:",
    reference,
    "",
    "Candidate English:",
    candidate,
    "",
    "Score the candidate on:",
    "- accuracy (1-5): meaning preservation",
    "- fluency (1-5): naturalness of English",
    "- terminology (1-5): handling of technical/proper terms",
    "- overall (1-5): overall quality",
    "",
    "Respond ONLY with a compact JSON object, no markdown fences:",
    '{"accuracy":number,"fluency":number,"terminology":number,"overall":number,"justification":"string"}',
  ].join("\n")
}

async function scoreWithJudge(
  _config: Record<string, unknown>,
  results: RawResult[],
): Promise<ScoredResult[]> {
  // User requested assistant to act as judge manually without external LLM judge.
  return results.map((r) => ({ ...r }))
}

function median(values: number[]): number {
  if (values.length === 0) return NaN
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

function mean(values: number[]): number {
  if (values.length === 0) return NaN
  return values.reduce((a, b) => a + b, 0) / values.length
}

function summarize(scored: ScoredResult[]) {
  const byModel = new Map<string, ScoredResult[]>()
  for (const r of scored) {
    if (!byModel.has(r.model)) byModel.set(r.model, [])
    byModel.get(r.model)!.push(r)
  }

  const summary: {
    model: string
    successRate: number
    medianLatencyMs: number
    meanAccuracy: number
    meanFluency: number
    meanTerminology: number
    meanOverall: number
    failures: number
  }[] = []

  for (const [model, rows] of byModel.entries()) {
    const successful = rows.filter((r) => r.success)
    const scoredRows = successful.filter((r) => r.scores)
    const latencies = successful.map((r) => r.latencyMs)
    summary.push({
      model,
      successRate: successful.length / rows.length,
      medianLatencyMs: Math.round(median(latencies)),
      meanAccuracy: Math.round(mean(scoredRows.map((r) => r.scores!.accuracy)) * 100) / 100,
      meanFluency: Math.round(mean(scoredRows.map((r) => r.scores!.fluency)) * 100) / 100,
      meanTerminology: Math.round(mean(scoredRows.map((r) => r.scores!.terminology)) * 100) / 100,
      meanOverall: Math.round(mean(scoredRows.map((r) => r.scores!.overall)) * 100) / 100,
      failures: rows.length - successful.length,
    })
  }

  summary.sort((a, b) => b.meanOverall - a.meanOverall || a.medianLatencyMs - b.medianLatencyMs)
  return summary
}

async function writeReports(
  timestamp: string,
  scored: ScoredResult[],
  summary: ReturnType<typeof summarize>,
) {
  await mkdir(RESULTS_DIR, { recursive: true })

  const rawPath = path.join(RESULTS_DIR, `${timestamp}-raw.json`)
  await writeFile(rawPath, JSON.stringify({ timestamp, candidates: CANDIDATES.map((c) => c.model), sentences: SENTENCES, results: scored }, null, 2))

  const mdPath = path.join(RESULTS_DIR, `${timestamp}-summary.md`)
  const lines: string[] = []
  lines.push("# Translation Benchmark Summary")
  lines.push("")
  lines.push(`Generated: ${timestamp}`)
  lines.push("")
  lines.push("## Models")
  lines.push("")
  lines.push("| Model | Success Rate | Median Latency (ms) | Accuracy | Fluency | Terminology | Overall | Failures |")
  lines.push("|-------|-------------:|--------------------:|---------:|--------:|------------:|--------:|---------:|")
  for (const row of summary) {
    lines.push(
      `| ${row.model} | ${(row.successRate * 100).toFixed(0)}% | ${row.medianLatencyMs} | ${row.meanAccuracy.toFixed(2)} | ${row.meanFluency.toFixed(2)} | ${row.meanTerminology.toFixed(2)} | ${row.meanOverall.toFixed(2)} | ${row.failures} |`,
    )
  }
  lines.push("")
  lines.push("## Per-Sentence Details")
  lines.push("")
  for (const sentence of SENTENCES) {
    lines.push(`### ${sentence.id}: ${sentence.source}`)
    lines.push("")
    lines.push(`Reference: ${sentence.reference}`)
    lines.push("")
    lines.push("| Model | Run | Latency (ms) | Translation | Overall | Justification |")
    lines.push("|-------|----:|-------------:|-------------|--------:|---------------|")
    for (const r of scored.filter((r) => r.sentenceId === sentence.id && r.success)) {
      const translation = (r.translation ?? "").replace(/\|/g, "\\|").replace(/\n/g, " ")
      const overall = r.scores?.overall?.toFixed(2) ?? "-"
      const justification = (r.scores?.justification ?? "-").replace(/\|/g, "\\|").replace(/\n/g, " ")
      lines.push(`| ${r.model} | ${r.run} | ${r.latencyMs} | ${translation} | ${overall} | ${justification} |`)
    }
    lines.push("")
  }

  await writeFile(mdPath, lines.join("\n"))
  return { rawPath, mdPath }
}

async function runBenchmark(config: Record<string, unknown>, timestamp: string, skipJudge: boolean) {
  console.log("=== Translation Benchmark ===")
  console.log(`Config: ${CONFIG_PATH}`)
  console.log(`Sentences: ${SENTENCES.length}`)
  console.log(`Candidates: ${CANDIDATES.length}`)
  console.log("")

  const raw = await benchmark(config)

  // Save raw results immediately so a later judge run can resume.
  await mkdir(RESULTS_DIR, { recursive: true })
  const rawPath = path.join(RESULTS_DIR, `${timestamp}-raw.json`)
  await writeFile(rawPath, JSON.stringify({ timestamp, candidates: CANDIDATES.map((c) => c.model), sentences: SENTENCES, results: raw }, null, 2))
  console.log(`\nRaw results saved: ${rawPath}`)

  if (skipJudge) {
    console.log("Skipping judge scoring (--skip-judge).")
    return
  }

  const scored = await scoreWithJudge(config, raw)
  const summary = summarize(scored)
  const { mdPath } = await writeReports(timestamp, scored, summary)

  console.log("\n=== Summary ===")
  for (const row of summary) {
    console.log(
      `${row.model}: overall=${row.meanOverall.toFixed(2)} latency=${row.medianLatencyMs}ms success=${(row.successRate * 100).toFixed(0)}%`,
    )
  }
  console.log(`\nSummary: ${mdPath}`)
}

async function runJudgeOnly(config: Record<string, unknown>, rawPath: string) {
  console.log(`=== Judge Only: ${rawPath} ===`)
  const raw = JSON.parse(await readFile(rawPath, "utf-8")) as { timestamp: string; results: RawResult[] }
  const scored = await scoreWithJudge(config, raw.results)
  const summary = summarize(scored)
  const { rawPath: outRawPath, mdPath } = await writeReports(raw.timestamp, scored, summary)

  console.log("\n=== Summary ===")
  for (const row of summary) {
    console.log(
      `${row.model}: overall=${row.meanOverall.toFixed(2)} latency=${row.medianLatencyMs}ms success=${(row.successRate * 100).toFixed(0)}%`,
    )
  }
  console.log(`\nScored raw results: ${outRawPath}`)
  console.log(`Summary: ${mdPath}`)
}

async function main() {
  const args = process.argv.slice(2)
  const skipJudge = args.includes("--skip-judge")
  const judgeOnlyIndex = args.indexOf("--judge-only")
  const judgeOnlyPath = judgeOnlyIndex >= 0 ? args[judgeOnlyIndex + 1] : undefined

  const config = await readConfig()

  if (judgeOnlyPath) {
    await runJudgeOnly(config, judgeOnlyPath)
  } else {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-")
    await runBenchmark(config, timestamp, skipJudge)
  }
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
