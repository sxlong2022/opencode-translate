import { randomBytes } from "node:crypto"

import type { Hooks, PluginInput, PluginOptions } from "@opencode-ai/plugin"
import {
  buildInboundTranslationError,
  isAsciiOnlyText,
  isTextPart,
  isTranslateStateRecord,
  isUserAuthoredTextPart,
  LLM_LANGUAGE,
  type MessageWithPartsLike,
  NONCE_PATTERN,
  normalizeReason,
  PLUGIN_NAME,
  type PluginClientLike,
  type ResolvedTranslateOptions,
  resolveOptions,
  SPEC_VERSION,
  type StoredTextMetadata,
  type TextPartLike,
  type TranslateState,
  unwrapData,
} from "./constants"
import { composeTranslatedAssistantText, composeTranslationFailureText, extractEnglishHistoryText } from "./formatting"
import { getDisplayLanguageLabel } from "./labels"
import {
  isQuestionArgs,
  type QuestionSnapshot,
  type QuestionToolOutput,
  restoreQuestionOutput,
  snapshotQuestions,
  translateQuestionArgs,
} from "./question-tool"
import { createSyntheticPartID, createTranslator, hashText } from "./translator"

const INACTIVE_ROOT_SESSION = Symbol("INACTIVE_ROOT_SESSION")
const INACTIVE_CHILD_SESSION = Symbol("INACTIVE_CHILD_SESSION")

type CachedSessionState = TranslateState | typeof INACTIVE_ROOT_SESSION | typeof INACTIVE_CHILD_SESSION

const sessionStateCache = new Map<string, CachedSessionState>()
const questionSnapshots = new Map<string, QuestionSnapshot>()
const QUESTION_TOOL_ID = "question"

export function __resetActivationCacheForTest() {
  sessionStateCache.clear()
  questionSnapshots.clear()
}

interface ResolvedSessionState {
  sessionActive: boolean
  canActivate: boolean
  state?: TranslateState
  storedMessages: MessageWithPartsLike[]
}

interface TriggerMatch {
  partArrayIndex: number
  eligibleIndex: number
  keyword: string
  offset: number
}

interface HookDependencies {
  translator?: {
    translateText(input: {
      text: string
      sourceLanguage: string
      targetLanguage: string
      direction: "inbound" | "outbound"
    }): Promise<{ text: string; modelUsed: string }>
    translateTexts?(input: {
      texts: readonly string[]
      sourceLanguage: string
      targetLanguage: string
      direction: "inbound" | "outbound"
    }): Promise<{ texts: string[]; modelUsed: string }>
  }
}

function logError(client: PluginClientLike, error: unknown) {
  return client.app.log({
    body: {
      service: PLUGIN_NAME,
      level: "error",
      message: normalizeReason(error),
    },
  })
}

function createState(options: ResolvedTranslateOptions): TranslateState {
  return {
    translate_enabled: true,
    translate_source_lang: options.sourceLanguage,
    translate_display_lang: options.displayLanguage,
    translate_llm_lang: LLM_LANGUAGE,
    translate_nonce: randomBytes(16).toString("hex"),
  }
}

function createActivationBannerText(options: ResolvedTranslateOptions): string {
  return `✓ Translation enabled · ${options.translatorModel}`
}

function asMetadata(part: TextPartLike): StoredTextMetadata {
  return (part.metadata ?? {}) as StoredTextMetadata
}

function extractStateFromMetadata(metadata: StoredTextMetadata | undefined): TranslateState | undefined {
  if (!isTranslateStateRecord(metadata)) return undefined
  return {
    translate_enabled: true,
    translate_source_lang: metadata.translate_source_lang,
    translate_display_lang: metadata.translate_display_lang,
    translate_llm_lang: LLM_LANGUAGE,
    translate_nonce: metadata.translate_nonce,
  }
}

export function extractStoredState(messages: MessageWithPartsLike[]): TranslateState | undefined {
  let fallback: TranslateState | undefined

  for (let messageIndex = messages.length - 1; messageIndex >= 0; messageIndex--) {
    const message = messages[messageIndex]
    const parts = message.parts ?? []

    for (let partIndex = parts.length - 1; partIndex >= 0; partIndex--) {
      const part = parts[partIndex]
      if (!isTextPart(part)) continue
      const metadata = asMetadata(part)
      const state = extractStateFromMetadata(metadata)

      if (metadata.translate_role === "activation_banner") {
        if (state) return state
        return undefined
      }
      if (
        state &&
        message.info.role === "user" &&
        part.synthetic !== true &&
        fallback === undefined
      ) {
        fallback = state
      }
    }
  }

  return fallback
}

function mergeTranslatedMetadata(state: TranslateState, part: TextPartLike, english: string): Record<string, unknown> {
  return {
    ...(part.metadata ?? {}),
    ...state,
    translate_source_hash: hashText(part.text ?? ""),
    translate_en: english,
  }
}

// OpenCode's flag semantics, observed from packages/opencode and packages/ui:
//   synthetic: true  -> hidden from the user UI, still sent to the LLM
//   ignored: true    -> hidden from the LLM, still shown in the user UI
// The translation preview, activation banner, and failure notices are
// user-facing status/diagnostic parts that must not leak into the LLM
// prompt, so they use synthetic:false + ignored:true.
function createSyntheticTextPart(
  sessionID: string,
  messageID: string,
  text: string,
  metadata: Record<string, unknown>,
): TextPartLike {
  return {
    id: createSyntheticPartID(),
    sessionID,
    messageID,
    type: "text",
    text,
    synthetic: false,
    ignored: true,
    metadata,
  }
}

// LLM-only text part: hidden from the TUI but the only LLM-visible
// representation of the user's source-language text. The original
// user-authored part is marked `ignored:true` so the LLM never sees it,
// and this synthetic English twin carries the actual prompt content.
function createLlmOnlyTextPart(
  sessionID: string,
  messageID: string,
  text: string,
  metadata: Record<string, unknown>,
): TextPartLike {
  return {
    id: createSyntheticPartID(),
    sessionID,
    messageID,
    type: "text",
    text,
    synthetic: true,
    ignored: false,
    metadata,
  }
}

function isTranslatedUserDisplayPart(part: TextPartLike): boolean {
  if (!isTextPart(part) || part.synthetic === true) return false
  return extractStateFromMetadata(asMetadata(part)) !== undefined
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

export function findTriggerMatch(parts: TextPartLike[], triggerKeywords: string[]): TriggerMatch | undefined {
  let eligibleIndex = 0
  for (let partArrayIndex = 0; partArrayIndex < parts.length; partArrayIndex += 1) {
    const part = parts[partArrayIndex]
    if (!isUserAuthoredTextPart(part)) continue

    let bestForPart: TriggerMatch | undefined
    for (let keywordIndex = 0; keywordIndex < triggerKeywords.length; keywordIndex += 1) {
      const keyword = triggerKeywords[keywordIndex]
      const pattern = new RegExp(`(^|[ \\t\\r\\n\\f\\v])${escapeRegex(keyword)}(?=$|[ \\t\\r\\n\\f\\v])`)
      const match = pattern.exec(part.text)
      if (!match) continue
      const offset = match.index + match[1].length
      if (!bestForPart || offset < bestForPart.offset) {
        bestForPart = {
          partArrayIndex,
          eligibleIndex,
          keyword,
          offset,
        }
      }
    }

    if (bestForPart) return bestForPart
    eligibleIndex += 1
  }

  return undefined
}

export function stripTriggerKeyword(text: string, keyword: string, offset: number): string {
  const lineStart = text.lastIndexOf("\n", offset - 1) + 1
  const nextNewline = text.indexOf("\n", offset)
  const lineEnd = nextNewline === -1 ? text.length : nextNewline
  const line = text.slice(lineStart, lineEnd)
  const localOffset = offset - lineStart

  let rewrittenLine: string
  if (localOffset === 0 && line.startsWith(`${keyword} `)) {
    rewrittenLine = line.slice(keyword.length + 1)
  } else if (
    localOffset + keyword.length === line.length &&
    localOffset > 0 &&
    line.slice(localOffset - 1, localOffset) === " "
  ) {
    rewrittenLine = line.slice(0, localOffset - 1)
  } else if (
    localOffset > 0 &&
    line.slice(localOffset - 1, localOffset) === " " &&
    line.slice(localOffset + keyword.length, localOffset + keyword.length + 1) === " "
  ) {
    rewrittenLine = `${line.slice(0, localOffset - 1)} ${line.slice(localOffset + keyword.length + 1)}`
  } else {
    rewrittenLine = `${line.slice(0, localOffset)}${line.slice(localOffset + keyword.length)}`
  }

  return `${text.slice(0, lineStart)}${rewrittenLine}${text.slice(lineEnd)}`
}

function cachedStateResult(cached: CachedSessionState): ResolvedSessionState {
  if (cached === INACTIVE_ROOT_SESSION) {
    return {
      sessionActive: false,
      canActivate: true,
      storedMessages: [],
    }
  }

  if (cached === INACTIVE_CHILD_SESSION) {
    return {
      sessionActive: false,
      canActivate: false,
      storedMessages: [],
    }
  }

  return {
    sessionActive: true,
    canActivate: false,
    state: cached,
    storedMessages: [],
  }
}

async function resolveSessionState(
  client: PluginClientLike,
  directory: string | undefined,
  sessionID: string,
): Promise<ResolvedSessionState> {
  const cached = sessionStateCache.get(sessionID)
  if (cached !== undefined) return cachedStateResult(cached)

  const session = unwrapData(
    await client.session.get({
      path: { id: sessionID },
      query: { ...(directory ? { directory } : {}) },
      throwOnError: true,
    }),
  )
  if (session.parentID != null) {
    sessionStateCache.set(sessionID, INACTIVE_CHILD_SESSION)
    return { sessionActive: false, canActivate: false, storedMessages: [] }
  }

  const storedMessages = unwrapData(
    await client.session.messages({
      path: { id: sessionID },
      query: { ...(directory ? { directory } : {}) },
      throwOnError: true,
    }),
  )
  const state = extractStoredState(storedMessages)
  sessionStateCache.set(sessionID, state ?? INACTIVE_ROOT_SESSION)

  return {
    sessionActive: Boolean(state),
    canActivate: !state,
    state: state ?? undefined,
    storedMessages,
  }
}

export function createHooks(ctx: PluginInput, rawOptions: PluginOptions = {}, deps: HookDependencies = {}): Hooks {
  if (process.env.OPENCODE_TRANSLATE_DISABLE === "1") {
    return {}
  }

  const client = ctx.client as unknown as PluginClientLike
  const options = resolveOptions(rawOptions)
  const translator = deps.translator ?? createTranslator(client, options)

  return {
    "chat.message": async (input, output) => {
      // Hooks must never throw. A thrown error propagates into OpenCode's
      // Effect runtime as a defect, kills the fiber, and stalls the session —
      // to the user this looks like infinite loading with no error message.
      // Instead, we log the failure and fall back to the untranslated text so
      // the chat keeps moving.
      try {
        const resolved = await resolveSessionState(client, ctx.directory, input.sessionID)
        let activeState = resolved.state
        const wasActiveBeforeTrigger = !!activeState
        let activatedThisTurn = false
        let disabledThisTurn = false

        // Check for disable keyword first (works even if translation is active)
        const disableMatch = findTriggerMatch(output.parts as TextPartLike[], options.disableKeywords)
        if (disableMatch) {
          const part = output.parts[disableMatch.partArrayIndex] as TextPartLike & { text: string }
          part.text = stripTriggerKeyword(part.text, disableMatch.keyword, disableMatch.offset)
          sessionStateCache.set(input.sessionID, INACTIVE_ROOT_SESSION)
          activeState = undefined
          disabledThisTurn = true
        }

        let match: TriggerMatch | undefined
        if (!disabledThisTurn && (activeState || resolved.canActivate)) {
          match = findTriggerMatch(output.parts as TextPartLike[], options.triggerKeywords)
          if (match) {
            const part = output.parts[match.partArrayIndex] as TextPartLike & { text: string }
            const originalText = part.text
            part.text = stripTriggerKeyword(part.text, match.keyword, match.offset)
            if (!activeState) {
              activeState = createState(options)
              if (!NONCE_PATTERN.test(activeState.translate_nonce)) {
                part.text = originalText
                await logError(client, new Error("Generated invalid translation nonce"))
                return
              }
            }
            activatedThisTurn = true
            sessionStateCache.set(input.sessionID, activeState)
          } else if (!activeState && resolved.canActivate) {
            sessionStateCache.set(input.sessionID, INACTIVE_ROOT_SESSION)
          }
        }


        if (!activeState && !disabledThisTurn) return

        // Track which parts originally contained a trigger keyword before stripping.
        // These parts should never be skipped by the ASCII-only bypass.
        const triggerKeywordPartIndices = new Set<number>()
        if (match) {
          triggerKeywordPartIndices.add(match.partArrayIndex)
        }

        // Step 1: Collect eligible parts that need translation
        const eligibleParts: Array<{ part: TextPartLike; currentEligibleIndex: number }> = []
        let partArrayIndex = -1
        let eligibleIndex = 0

        for (const part of output.parts as TextPartLike[]) {
          partArrayIndex += 1
          if (!activeState) continue
          if (!isUserAuthoredTextPart(part)) continue
          if (part.text.trim().length === 0) continue
          if (isAsciiOnlyText(part) && !triggerKeywordPartIndices.has(partArrayIndex)) continue

          eligibleParts.push({ part, currentEligibleIndex: eligibleIndex })
          eligibleIndex += 1
        }

        // Step 2: Translate all eligible parts concurrently
        const translationResults: Array<{
          part: TextPartLike
          success: boolean
          text?: string
          modelUsed?: string
          error?: unknown
        }> = await Promise.all(
          eligibleParts.map(async ({ part }) => {
            try {
              const result = await translator.translateText({
                text: part.text ?? "",
                sourceLanguage: activeState!.translate_source_lang,
                targetLanguage: LLM_LANGUAGE,
                direction: "inbound",
              })
              return { part, success: true, text: result.text, modelUsed: result.modelUsed }
            } catch (error) {
              return { part, success: false, error }
            }
          })
        )

        // Step 3: Reassemble nextParts with translation results in original order
        const nextParts: TextPartLike[] = []
        let fallbackUsedModel: string | undefined
        const translationErrors: { part: TextPartLike; error: unknown }[] = []

        for (const part of output.parts as TextPartLike[]) {
          nextParts.push(part)

          const eligibleInfo = eligibleParts.find((e) => e.part === part)
          if (!eligibleInfo) continue

          const result = translationResults.find((r) => r.part === part)!

          if (result.success) {
            const english = result.text!
            if (options.fallbackModel && result.modelUsed === options.fallbackModel) {
              fallbackUsedModel = result.modelUsed
            }

            const sourceHash = hashText(part.text ?? "")
            part.metadata = {
              ...(part.metadata ?? {}),
              ...mergeTranslatedMetadata(activeState!, part, english),
            }

            nextParts.push(
              createLlmOnlyTextPart(part.sessionID, part.messageID, english, {
                translate_role: "llm_only_translation",
                translate_nonce: activeState!.translate_nonce,
                translate_source_hash: sourceHash,
                translate_part_index: eligibleInfo.currentEligibleIndex,
              }),
            )
          } else {
            translationErrors.push({ part, error: result.error! })
            nextParts.push(
              createSyntheticTextPart(
                part.sessionID,
                part.messageID,
                `⚠️ Translation failed: ${normalizeReason(result.error!)}. Original text will be sent to the model.`,
                {
                  translate_role: "translation_failure",
                  translate_nonce: activeState!.translate_nonce,
                  translate_part_index: eligibleInfo.currentEligibleIndex,
                },
              ),
            )
          }
        }

        // Step 4: Log errors after all translations complete
        if (translationErrors.length > 0 && activeState) {
          await Promise.all(
            translationErrors.map(({ error }) =>
              logError(client, buildInboundTranslationError(activeState.translate_source_lang, normalizeReason(error)))
            )
          )
        }

        // If this was the FIRST activation this turn and translation failed for every
        // user-authored part, roll back activation so the next turn does a clean retry
        // instead of cementing a broken state. If the session was already active before
        // this turn, we keep the existing state so the user still sees the status banner.
        if (activatedThisTurn && !wasActiveBeforeTrigger && translationErrors.length > 0 && eligibleIndex === translationErrors.length) {
          sessionStateCache.set(input.sessionID, INACTIVE_ROOT_SESSION)
          return
        }

        if (activatedThisTurn) {
          nextParts.push(
            createSyntheticTextPart(input.sessionID, output.message.id, createActivationBannerText(options), {
              ...activeState,
              translate_role: "activation_banner",
              translate_spec_version: SPEC_VERSION,
              compaction_continue: true,
            }),
          )
        }

        // Show fallback banner if fallback model was used
        if (fallbackUsedModel) {
          nextParts.push(
            createSyntheticTextPart(input.sessionID, output.message.id, `⚠ Translation fallback: ${fallbackUsedModel}`, {
              ...activeState,
              translate_role: "activation_banner",
              translate_spec_version: SPEC_VERSION,
              compaction_continue: true,
            }),
          )
        }

        if (disabledThisTurn) {
          nextParts.push(
            createSyntheticTextPart(input.sessionID, output.message.id, "✗ Translation disabled", {
              translate_role: "activation_banner",
              translate_spec_version: SPEC_VERSION,
              compaction_continue: true,
            }),
          )
        }

        output.parts.splice(0, output.parts.length, ...(nextParts as typeof output.parts))
      } catch (error) {
        await logError(client, error)
      }
    },
    "experimental.chat.messages.transform": async (_input, output) => {
      try {
        const sessionID = output.messages[0]?.info.sessionID
        if (!sessionID) return

        const resolved = await resolveSessionState(client, ctx.directory, sessionID)
        const activeState = resolved.state
        if (!activeState) return

        for (const message of output.messages as MessageWithPartsLike[]) {
          if (message.info.role === "user") {
            for (const part of message.parts) {
              if (isTranslatedUserDisplayPart(part)) {
                part.ignored = true
              }
            }
            continue
          }

          if (message.info.role !== "assistant") continue
          for (const part of message.parts) {
            if (!isTextPart(part)) continue
            part.text = extractEnglishHistoryText(part.text, activeState.translate_nonce)
          }
        }
      } catch (error) {
        await logError(client, error)
      }
    },
    "experimental.text.complete": async (input, output) => {
      try {
        const resolved = await resolveSessionState(client, ctx.directory, input.sessionID)
        const activeState = resolved.state
        if (!activeState) return
        if (!options.translateResponses) return

        const message = unwrapData(
          await client.session.message({
            path: { id: input.sessionID, messageID: input.messageID },
            query: { ...(ctx.directory ? { directory: ctx.directory } : {}) },
            throwOnError: true,
          }),
        ) as MessageWithPartsLike & { info: Record<string, unknown> }

        if (message.info.role !== "assistant") return
        if (message.info.summary === true) return
        if (activeState.translate_display_lang === LLM_LANGUAGE || output.text.length === 0) return

        try {
          const translationResult = await translator.translateText({
            text: output.text,
            sourceLanguage: LLM_LANGUAGE,
            targetLanguage: activeState.translate_display_lang,
            direction: "outbound",
          })

          output.text = composeTranslatedAssistantText(
            output.text,
            getDisplayLanguageLabel(activeState.translate_display_lang),
            translationResult.text,
            activeState.translate_nonce,
          )
        } catch (error) {
          output.text = composeTranslationFailureText(output.text, activeState.translate_nonce)
          await logError(client, error)
        }
      } catch (error) {
        await logError(client, error)
      }
    },
    // Translate the built-in `question` tool so the TUI dialog renders in
    // the user's displayLanguage. The tool output string is restored back
    // to English in `tool.execute.after` so the main LLM context stays
    // English-only.
    "tool.execute.before": async (input, output) => {
      try {
        if (input.tool !== QUESTION_TOOL_ID) return
        const resolved = await resolveSessionState(client, ctx.directory, input.sessionID)
        const activeState = resolved.state
        if (!activeState) return
        if (activeState.translate_display_lang === LLM_LANGUAGE) return

        const args = output.args as unknown
        if (!isQuestionArgs(args)) return

        const original = snapshotQuestions(args)
        try {
          const batchTranslate = async (texts: readonly string[]) => {
            return translator.translateTexts
              ? translator.translateTexts({
                  texts,
                  sourceLanguage: LLM_LANGUAGE,
                  targetLanguage: activeState.translate_display_lang,
                  direction: "outbound",
                }).then((r: any) => r.texts)
              : Promise.all(
                  texts.map((text) =>
                    translator.translateText({
                      text,
                      sourceLanguage: LLM_LANGUAGE,
                      targetLanguage: activeState.translate_display_lang,
                      direction: "outbound",
                    }).then((r: any) => r.text)
                  )
                )
          }
          batchTranslate.isBatch = true
          await translateQuestionArgs(args, batchTranslate)
        } catch (error) {
          // Translation failed: restore the originals so the dialog at least
          // renders in English instead of a half-translated mess.
          args.questions.splice(0, args.questions.length, ...snapshotQuestions({ questions: original }))
          await logError(client, error)
          return
        }

        const translated = snapshotQuestions(args)
        questionSnapshots.set(input.callID, { original, translated })
      } catch (error) {
        await logError(client, error)
      }
    },
    "tool.execute.after": async (input, output) => {
      try {
        if (input.tool !== QUESTION_TOOL_ID) return
        const snapshot = questionSnapshots.get(input.callID)
        if (!snapshot) return
        questionSnapshots.delete(input.callID)
        const resolved = await resolveSessionState(client, ctx.directory, input.sessionID)
        const activeState = resolved.state
        const translateCustomAnswers = activeState && activeState.translate_source_lang !== LLM_LANGUAGE
          ? async (texts: readonly string[]) => {
              return translator.translateTexts
                ? translator.translateTexts({
                    texts,
                    sourceLanguage: activeState.translate_source_lang,
                    targetLanguage: LLM_LANGUAGE,
                    direction: "inbound",
                  }).then((r: any) => r.texts)
                : Promise.all(
                    texts.map((text) =>
                      translator.translateText({
                        text,
                        sourceLanguage: activeState.translate_source_lang,
                        targetLanguage: LLM_LANGUAGE,
                        direction: "inbound",
                      }).then((r: any) => r.text)
                    )
                  )
            }
          : undefined

        await restoreQuestionOutput(output as QuestionToolOutput, snapshot, {
          ...(translateCustomAnswers ? { translateCustomAnswers } : {}),
          onTranslationError: async (error) => {
            if (!activeState) {
              await logError(client, error)
              return
            }
            await logError(
              client,
              buildInboundTranslationError(activeState.translate_source_lang, normalizeReason(error)),
            )
          },
        })
      } catch (error) {
        await logError(client, error)
      }
    },
    "experimental.session.compacting": async (input, output) => {
      try {
        const state = sessionStateCache.get(input.sessionID)
        if (!state || typeof state === "symbol") return

        const modelLabel = options.translatorModel
        output.context.push(
          `## Translation State\n` +
          `- Translation is ENABLED for this session\n` +
          `- Source language: ${state.translate_source_lang}\n` +
          `- Display language: ${state.translate_display_lang}\n` +
          `- LLM language: ${state.translate_llm_lang}\n` +
          `- Translator model: ${modelLabel}\n` +
          `\n` +
          `After compaction, translation should remain active. The next user message will be translated automatically.\n` +
          `Do NOT include translation state in your summary — it is managed by the plugin.\n`,
        )
      } catch (error) {
        // Compaction hook failures must not block compaction
        await logError(client, error)
      }
    },
  }
}
