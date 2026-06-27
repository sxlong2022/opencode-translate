import { unwrapEchoedTextEnvelope } from "./prompts"

// Translation layer for OpenCode's built-in `question` tool.
//
// Flow:
//   1. Agent (main LLM, English-only) invokes the `question` tool with an
//      `args.questions[]` payload in English.
//   2. `tool.execute.before` hook translates each question's text, header,
//      and every option's label + description into `displayLanguage` so the
//      question prompt renders in the user's language.
//   3. OpenCode publishes `question.asked`; the TUI shows the translated
//      dialog and the user picks an option (or types a custom answer).
//   4. `tool.execute.after` hook reverses the substitution using the
//      snapshot we captured in step 2, so the tool output string delivered
//      back to the LLM stays in English.
//
// A per-callID snapshot is kept so mapping a user-selected translated label
// back to its original English label is deterministic.

type TextRecord = { question: string; header: string; options: OptionRecord[]; multiple?: boolean; custom?: boolean }
type OptionRecord = { label: string; description: string }

export interface QuestionArgs {
  questions: TextRecord[]
}

export interface QuestionSnapshot {
  original: TextRecord[]
  translated: TextRecord[]
}

export interface QuestionToolOutput {
  title?: string
  output?: string
  metadata?: { answers?: readonly (readonly string[])[] } | Record<string, unknown>
}

function cloneQuestion(q: TextRecord): TextRecord {
  return {
    question: q.question,
    header: q.header,
    options: q.options.map((option) => ({ label: option.label, description: option.description })),
    ...(q.multiple !== undefined ? { multiple: q.multiple } : {}),
    ...(q.custom !== undefined ? { custom: q.custom } : {}),
  }
}

export function snapshotQuestions(args: QuestionArgs): TextRecord[] {
  return args.questions.map(cloneQuestion)
}

export function isQuestionArgs(value: unknown): value is QuestionArgs {
  if (!value || typeof value !== "object") return false
  const questions = (value as Record<string, unknown>).questions
  if (!Array.isArray(questions)) return false
  for (const q of questions) {
    if (!q || typeof q !== "object") return false
    const record = q as Record<string, unknown>
    if (typeof record.question !== "string") return false
    if (typeof record.header !== "string") return false
    if (!Array.isArray(record.options)) return false
    for (const opt of record.options) {
      if (!opt || typeof opt !== "object") return false
      const optRecord = opt as Record<string, unknown>
      if (typeof optRecord.label !== "string") return false
      if (typeof optRecord.description !== "string") return false
    }
  }
  return true
}

export interface RestoreQuestionOutputOptions {
  translateCustomAnswers?: (texts: readonly string[]) => Promise<readonly string[]>
  translateCustomAnswer?: (text: string) => Promise<string>
  onTranslationError?: (error: unknown) => Promise<void> | void
}

interface TranslatableField {
  text: string
  set(value: string): void
}

interface CustomAnswerSlot {
  questionIndex: number
  answerIndex: number
  text: string
}

export async function translateQuestionArgs(
  args: QuestionArgs,
  translate: (((text: string) => Promise<string>) | ((texts: readonly string[]) => Promise<readonly string[]>)) & { isBatch?: boolean },
): Promise<void> {
  const translatedQuestions = snapshotQuestions(args)
  const fields: TranslatableField[] = []

  function addField(text: string, set: (value: string) => void) {
    if (text.length === 0) return
    fields.push({ text, set })
  }

  for (const q of translatedQuestions) {
    addField(q.question, (value) => {
      q.question = value
    })
    addField(q.header, (value) => {
      q.header = value
    })
    for (const option of q.options) {
      addField(option.label, (value) => {
        option.label = value
      })
      addField(option.description, (value) => {
        option.description = value
      })
    }
  }

  if (fields.length === 0) return

  if (translate.isBatch) {
    const batchTranslate = translate as unknown as (texts: readonly string[]) => Promise<readonly string[]>
    const translated = await batchTranslate(fields.map((field) => field.text))
    if (translated.length !== fields.length) {
      throw new Error(`Question translator returned ${translated.length} translations for ${fields.length} fields`)
    }
    for (const [index, field] of fields.entries()) {
      field.set(unwrapEchoedTextEnvelope(translated[index]))
    }
  } else {
    const singleTranslate = translate as (text: string) => Promise<string>
    const jobs = fields.map(async (field) => {
      const val = await singleTranslate(field.text)
      field.set(unwrapEchoedTextEnvelope(val))
    })
    await Promise.all(jobs)
  }

  args.questions.splice(0, args.questions.length, ...translatedQuestions)
}

// Given the user-selected labels (`answers`), find the matching translated
// option and return its original English label. If no match (e.g. a custom
// free-text answer), return the label verbatim so the LLM still sees what
// the user actually typed.
function restoreOptionLabel(
  selectedLabel: string,
  translatedOptions: readonly OptionRecord[],
  originalOptions: readonly OptionRecord[],
): string | undefined {
  const idx = translatedOptions.findIndex((option) => option.label === selectedLabel)
  if (idx < 0) return undefined
  return originalOptions[idx]?.label ?? selectedLabel
}

async function restoreQuestionAnswers(
  original: readonly TextRecord[],
  translated: readonly TextRecord[],
  answers: readonly (readonly string[])[],
  options: RestoreQuestionOutputOptions = {},
): Promise<string[][]> {
  const translateCustomAnswers = options.translateCustomAnswers
  const translateCustomAnswer = options.translateCustomAnswer
  const customSlots: CustomAnswerSlot[] = []
  const restored = original.map((q, questionIndex) => {
    const selected = answers[questionIndex] ?? []
    const translatedOptions = translated[questionIndex]?.options ?? []
    const originalOptions = q.options

    return selected.map((label, answerIndex) => {
      const restoredLabel = restoreOptionLabel(label, translatedOptions, originalOptions)
      if (restoredLabel !== undefined) return restoredLabel
      if (label.trim().length === 0) return label
      if (!translateCustomAnswers && !translateCustomAnswer) return label

      customSlots.push({ questionIndex, answerIndex, text: label })
      return label
    })
  })

  if (customSlots.length === 0) return restored

  if (translateCustomAnswers) {
    try {
      const translatedCustomAnswers = await translateCustomAnswers(customSlots.map((slot) => slot.text))
      if (translatedCustomAnswers.length !== customSlots.length) {
        throw new Error(
          `Question custom-answer translator returned ${translatedCustomAnswers.length} translations for ${customSlots.length} answers`,
        )
      }
      for (const [index, slot] of customSlots.entries()) {
        restored[slot.questionIndex][slot.answerIndex] = unwrapEchoedTextEnvelope(translatedCustomAnswers[index])
      }
    } catch (error) {
      await options.onTranslationError?.(error)
    }
  } else if (translateCustomAnswer) {
    const jobs = customSlots.map(async (slot) => {
      try {
        const val = await translateCustomAnswer(slot.text)
        restored[slot.questionIndex][slot.answerIndex] = unwrapEchoedTextEnvelope(val)
      } catch (error) {
        await options.onTranslationError?.(error)
      }
    })
    await Promise.all(jobs)
  }

  return restored
}

function formatRestoredOutput(original: readonly TextRecord[], answers: readonly (readonly string[])[]): string {
  const formatted = original
    .map((q, i) => {
      const selected = answers[i] ?? []
      const rendered = selected.length > 0 ? selected.join(", ") : "Unanswered"
      return `"${q.question}"="${rendered}"`
    })
    .join(", ")
  return `User has answered your questions: ${formatted}. You can now continue with the user's answers in mind.`
}

export async function restoreQuestionOutput(
  output: QuestionToolOutput,
  snapshot: QuestionSnapshot,
  options?: RestoreQuestionOutputOptions,
): Promise<void> {
  if (typeof output.output !== "string") return
  const answersRaw = (output.metadata as { answers?: readonly (readonly string[])[] } | undefined)?.answers
  const answers = Array.isArray(answersRaw) ? answersRaw : []
  const restored = await restoreQuestionAnswers(snapshot.original, snapshot.translated, answers, options)
  output.output = formatRestoredOutput(snapshot.original, restored)
}

export function buildRestoredOutput(
  original: readonly TextRecord[],
  translated: readonly TextRecord[],
  answers: readonly (readonly string[])[],
): string {
  const restored = original.map((q, questionIndex) => {
    const selected = answers[questionIndex] ?? []
    const translatedOptions = translated[questionIndex]?.options ?? []
    const originalOptions = q.options

    return selected.map((label) => {
      const restoredLabel = restoreOptionLabel(label, translatedOptions, originalOptions)
      return restoredLabel ?? label
    })
  })
  return formatRestoredOutput(original, restored)
}
