import { beforeEach, describe, expect, test } from "bun:test"
import { __resetActivationCacheForTest, createHooks } from "../src/activation"
import type { MessageWithPartsLike, PluginClientLike, TextPartLike } from "../src/constants"
import { __resetTranslatorCachesForTest, createTranslator, hashText } from "../src/translator"

function textPart(id: string, text: string, extra: Partial<TextPartLike> = {}): TextPartLike {
  return {
    id,
    sessionID: "ses_1",
    messageID: "msg_1",
    type: "text",
    text,
    ...extra,
  }
}

function activeStateMetadata(text: string) {
  return {
    translate_enabled: true,
    translate_source_lang: "ko",
    translate_display_lang: "ko",
    translate_llm_lang: "en",
    translate_nonce: "0123456789abcdef0123456789abcdef",
    translate_source_hash: hashText(text),
    translate_en: `EN:${text}`,
  }
}

function fakeClient(messages: MessageWithPartsLike[]): PluginClientLike {
  return {
    session: {
      get: async () => ({ id: "ses_1", parentID: null }),
      messages: async () => messages,
      message: async () => messages[0],
    },
    provider: {
      list: async () => ({ all: [] }),
    },
    auth: {
      set: async () => undefined,
    },
    app: {
      log: async () => undefined,
    },
  }
}

describe("translator", () => {
  beforeEach(() => {
    __resetActivationCacheForTest()
    __resetTranslatorCachesForTest()
  })

  test("retry succeeds after one transient failure", async () => {
    let calls = 0
    const translator = createTranslator(
      fakeClient([]),
      {
        translatorModel: "anthropic/claude-haiku-4-5",
        triggerKeywords: ["$en"],
        sourceLanguage: "ko",
        displayLanguage: "ko",
        verbose: false,
        disableKeywords: ["$dis"],
        translateResponses: false,
        baseURL: "https://api.anthropic.com",
      },
      {
        credentialResolver: {
          resolve: async () => ({ providerID: "anthropic", apiKey: "test-key", mode: "apiKey" as const }),
          isMissingCredentialError: () => false,
          authUnavailable: () => new Error("unused"),
          envFallback: "ANTHROPIC_API_KEY",
        },
        generateTextImpl: async () => {
          calls += 1
          if (calls === 1) {
            const error = new Error("HTTP 500") as Error & { status?: number }
            error.status = 500
            throw error
          }
          return { text: "hello" } as never
        },
        sleep: async () => undefined,
        now: (() => {
          let value = 0
          return () => (value += 10)
        })(),
      },
    )

    const translated = await translator.translateText({
      text: "안녕",
      sourceLanguage: "ko",
      targetLanguage: "en",
      direction: "inbound",
    })

    expect(translated).toEqual({ text: "hello", modelUsed: "anthropic/claude-haiku-4-5" })
    expect(calls).toBe(2)
  })

  test("falls back to the fallback model on primary model failure immediately without primary retries", async () => {
    let calls = 0
    const translator = createTranslator(
      fakeClient([]),
      {
        translatorModel: "anthropic/claude-haiku-4-5",
        fallbackModel: "openai/gpt-5.4",
        triggerKeywords: ["$en"],
        sourceLanguage: "ko",
        displayLanguage: "ko",
        verbose: false,
        disableKeywords: ["$dis"],
        translateResponses: false,
        baseURL: "https://api.anthropic.com",
      },
      {
        credentialResolver: {
          resolve: async (model) => ({
            providerID: model.split("/")[0],
            apiKey: "test-key",
            mode: "apiKey" as const,
          }),
          isMissingCredentialError: () => false,
          authUnavailable: () => new Error("unused"),
          envFallback: "API_KEY",
        },
        generateTextImpl: async (args: any) => {
          calls += 1
          if (calls === 1) {
            const error = new Error("HTTP 500") as Error & { status?: number }
            error.status = 500
            throw error
          }
          return { text: "fallback-hello" } as never
        },
        sleep: async () => undefined,
        now: (() => {
          let value = 0
          return () => (value += 10)
        })(),
      },
    )

    const translated = await translator.translateText({
      text: "안녕",
      sourceLanguage: "ko",
      targetLanguage: "en",
      direction: "inbound",
    })

    expect(translated).toEqual({ text: "fallback-hello", modelUsed: "openai/gpt-5.4" })
    expect(calls).toBe(2)
  })

  test("final failure in chat.message does not throw and falls back to the untranslated text", async () => {
    const hooks = createHooks(
      {
        client: fakeClient([]),
        directory: "/workspace",
      } as never,
      { sourceLanguage: "ko", displayLanguage: "ko" },
      {
        translator: {
          translateText: async () => {
            throw new Error("translator unavailable")
          },
        },
      },
    )

    const output = {
      message: { id: "msg_new" },
      parts: [textPart("p1", "$en 안녕")],
    }

    // Must NOT reject — a thrown error in chat.message stalls OpenCode's
    // session fiber, which appears to the user as infinite loading.
    await hooks["chat.message"]!({ sessionID: "ses_1" }, output as never)

    // Activation is rolled back so a later turn can retry cleanly.
    // Only the original (trigger-stripped) user part remains.
    expect((output.parts[0] as TextPartLike).text).toBe("안녕")
    expect((output.parts[0] as TextPartLike).metadata?.translate_en).toBeUndefined()
  })

  test("transform leaves user parts untouched on the LLM-only twin architecture", async () => {
    let calls = 0
    const messages = [
      {
        info: { id: "msg_user", sessionID: "ses_1", role: "user" },
        parts: [
          textPart("hist", "안녕", {
            metadata: activeStateMetadata("안녕"),
          }),
        ],
      },
    ]

    const hooks = createHooks(
      {
        client: fakeClient(messages),
        directory: "/workspace",
      } as never,
      { sourceLanguage: "ko", displayLanguage: "ko" },
      {
        translator: {
          translateText: async ({ text }) => {
            calls += 1
            return { text: `EN:${text}`, modelUsed: "test/model" }
          },
        },
      },
    )

    const output = {
      messages: [
        {
          info: { id: "msg_user", sessionID: "ses_1", role: "user" },
          parts: [
            textPart("hist", "안녕", {
              metadata: activeStateMetadata("안녕"),
            }),
          ],
        },
      ],
    }

    await hooks["experimental.chat.messages.transform"]!({} as never, output as never)

    // The translator must never run inside transform (no network in this
    // hook), and the user-side source-language text is left untouched.
    // The synthetic LLM-only English twin (added in `chat.message`) is
    // what actually feeds the model; transform is responsible only for
    // assistant-side trailer stripping now.
    expect(calls).toBe(0)
    expect((output.messages[0].parts[0] as TextPartLike).text).toBe("안녕")
  })

  test("hash mismatch in transform does not throw and does not call the translator", async () => {
    let calls = 0
    const history = [
      {
        info: { id: "msg_user", sessionID: "ses_1", role: "user" },
        parts: [
          textPart("hist", "원본", {
            metadata: activeStateMetadata("원본"),
          }),
        ],
      },
    ]

    const hooks = createHooks(
      {
        client: fakeClient(history),
        directory: "/workspace",
      } as never,
      { sourceLanguage: "ko", displayLanguage: "ko" },
      {
        translator: {
          translateText: async ({ text }) => {
            calls += 1
            return { text: `EN:${text}`, modelUsed: "test/model" }
          },
        },
      },
    )

    const output = {
      messages: [
        {
          info: { id: "msg_user", sessionID: "ses_1", role: "user" },
          parts: [
            textPart("hist", "편집됨", {
              metadata: activeStateMetadata("원본"),
            }),
          ],
        },
      ],
    }

    // Must NOT reject — throwing from a hook stalls OpenCode's session.
    // The edited text stays as-is so the original user message still
    // reaches the model.
    await hooks["experimental.chat.messages.transform"]!({} as never, output as never)

    expect(calls).toBe(0)
    expect((output.messages[0].parts[0] as TextPartLike).text).toBe("편집됨")
  })

  test("synthetic user parts are skipped during inbound translation", async () => {
    let calls = 0
    const hooks = createHooks(
      {
        client: fakeClient([]),
        directory: "/workspace",
      } as never,
      { sourceLanguage: "ko", displayLanguage: "ko" },
      {
        translator: {
          translateText: async ({ text }) => {
            calls += 1
            return { text: `EN:${text}`, modelUsed: "test/model" }
          },
        },
      },
    )

    const output = {
      message: { id: "msg_new" },
      parts: [
        textPart("p1", "$en 첫번째"),
        textPart("p2", "compaction marker", {
          synthetic: true,
          ignored: true,
          metadata: { compaction_continue: true },
        }),
        textPart("p3", "두번째"),
      ],
    }

    await hooks["chat.message"]!({ sessionID: "ses_1" }, output as never)

    // After two user-authored translations the layout is:
    //   [0] "첫번째"           (source, ignored:true)
    //   [1] "EN:첫번째"        (LLM-only synthetic twin)
    //   [2] compaction marker (untouched, was synthetic in input)
    //   [3] "두번째"           (source, ignored:true)
    //   [4] "EN:두번째"        (LLM-only synthetic twin)
    //   [5] activation banner
    expect(calls).toBe(2)
    expect((output.parts[2] as TextPartLike).text).toBe("compaction marker")
    expect((output.parts[2] as TextPartLike).synthetic).toBe(true)
  })

  test("missing credentials surface the exact auth-unavailable error", async () => {
    const translator = createTranslator(
      fakeClient([]),
      {
        translatorModel: "anthropic/claude-haiku-4-5",
        triggerKeywords: ["$en"],
        sourceLanguage: "ko",
        displayLanguage: "ko",
        verbose: false,
        disableKeywords: ["$dis"],
        translateResponses: false,
        baseURL: "https://api.anthropic.com",
      },
      {
        credentialResolver: {
          resolve: async () => ({
            providerID: "anthropic",
            provider: {
              id: "anthropic",
              source: "env",
              env: ["ANTHROPIC_API_KEY"],
            },
            mode: "default" as const,
          }),
          isMissingCredentialError: () => true,
          authUnavailable: () => new Error("unused"),
          envFallback: "ANTHROPIC_API_KEY",
        },
        generateTextImpl: async () => {
          throw new Error("Missing API key")
        },
        sleep: async () => undefined,
      },
    )

    await expect(
      translator.translateText({
        text: "안녕",
        sourceLanguage: "ko",
        targetLanguage: "en",
        direction: "inbound",
      }),
    ).rejects.toThrow(
      '[opencode-translate:AUTH_UNAVAILABLE] No credential found for provider "anthropic". Set ANTHROPIC_API_KEY in the environment, run "opencode auth login anthropic", or set options.apiKey in opencode.json.',
    )
  })

  test("model-specific providerOptions are correctly passed based on the model used", async () => {
    let capturedOptions: any = null
    const customClient: any = {
      ...fakeClient([]),
      provider: {
        list: async () => ({
          all: [
            {
              id: "cloudflare",
              source: "api",
              key: "test-key",
              env: ["CLOUDFLARE_API_KEY"],
              options: {
                baseURL: "https://api.cloudflare.com/client/v4",
              },
              models: {},
            },
          ],
        }),
      },
    }
    const translator = createTranslator(
      customClient,
      {
        translatorModel: "cloudflare/gemma-it",
        fallbackModel: "cloudflare/llama-instruct",
        triggerKeywords: ["$en"],
        sourceLanguage: "ko",
        displayLanguage: "ko",
        verbose: false,
        disableKeywords: ["$dis"],
        translateResponses: false,
        translatorProviderOptions: {
          cloudflare: { reasoning_effort: "low" },
        },
        fallbackProviderOptions: {
          cloudflare: { fallback_only: true },
        },
      },
      {
        credentialResolver: {
          resolve: async () => ({
            providerID: "cloudflare",
            mode: "apiKey" as const,
            apiKey: "test-key",
            baseURL: "https://api.cloudflare.com/client/v4",
          }),
          isMissingCredentialError: () => false,
          authUnavailable: () => new Error("unused"),
          envFallback: "CLOUDFLARE_API_KEY",
        },
generateTextImpl: async (args: any) => {
capturedOptions = args.providerOptions
          return { text: "Hello" } as never
},
        sleep: async () => undefined,
      },
    )

    // Call translation (primary model is used first)
    const result = await translator.translateText({
      text: "안녕",
      sourceLanguage: "ko",
      targetLanguage: "en",
      direction: "inbound",
    })

    expect(result.text).toBe("Hello")
    expect(result.modelUsed).toBe("cloudflare/gemma-it")
    expect(capturedOptions).toEqual({ cloudflare: { reasoning_effort: "low" } })

    // Now test fallback
    let fallbackCapturedOptions: any = null
    const fallbackTranslator = createTranslator(
      customClient,
      {
        translatorModel: "cloudflare/gemma-it",
        fallbackModel: "cloudflare/llama-instruct",
        triggerKeywords: ["$en"],
        sourceLanguage: "ko",
        displayLanguage: "ko",
        verbose: false,
        disableKeywords: ["$dis"],
        translateResponses: false,
        translatorProviderOptions: {
          cloudflare: { reasoning_effort: "low" },
        },
        fallbackProviderOptions: {
          cloudflare: { fallback_only: true },
        },
      },
      {
        credentialResolver: {
          resolve: async () => ({
            providerID: "cloudflare",
            mode: "apiKey" as const,
            apiKey: "test-key",
            baseURL: "https://api.cloudflare.com/client/v4",
          }),
          isMissingCredentialError: () => false,
          authUnavailable: () => new Error("unused"),
          envFallback: "CLOUDFLARE_API_KEY",
        },
generateTextImpl: async (args: any) => {
if (args.model && typeof args.model === "object" && args.model.modelId === "gemma-it") {
throw new Error("Primary model failed")
}
fallbackCapturedOptions = args.providerOptions
          return { text: "Hello from fallback" } as never
},
        sleep: async () => undefined,
      },
    )

    const fallbackResult = await fallbackTranslator.translateText({
      text: "안녕",
      sourceLanguage: "ko",
      targetLanguage: "en",
      direction: "inbound",
    })

    expect(fallbackResult.text).toBe("Hello from fallback")
    expect(fallbackResult.modelUsed).toBe("cloudflare/llama-instruct")
    expect(fallbackCapturedOptions).toEqual({ cloudflare: { fallback_only: true } })
  })

  // ---------------------------------------------------------------------------
  // Total budget tests
  // ---------------------------------------------------------------------------

  function makeCredentialResolver(providerID = "anthropic") {
    return {
      resolve: async (model: string) => ({
        providerID: model.split("/")[0],
        apiKey: "test-key",
        mode: "apiKey" as const,
      }),
      isMissingCredentialError: () => false,
      authUnavailable: () => new Error("unused"),
      envFallback: "API_KEY",
    }
  }

  test("primary timeout triggers fallback which succeeds within budget", async () => {
    let calls = 0
    // Use a fixed now() so deadline arithmetic is deterministic.
    // deadline = 0 + 15_000 = 15_000.
    // After primary throws, remaining = 15_000 - 1 = 14_999 > 500 → fallback runs.
    let timeNow = 0
    const translator = createTranslator(
      fakeClient([]),
      {
        translatorModel: "anthropic/claude-haiku-4-5",
        fallbackModel: "openai/gpt-4o-mini",
        triggerKeywords: ["$en"],
        sourceLanguage: "ko",
        displayLanguage: "ko",
        verbose: false,
        disableKeywords: ["$dis"],
        translateResponses: false,
      },
      {
        credentialResolver: makeCredentialResolver(),
        generateTextImpl: async () => {
          calls += 1
          if (calls === 1) {
            timeNow = 1 // advance by 1ms — well within budget
            throw new Error("Translator generateText timed out after 10000ms")
          }
          return { text: "fallback-result" } as never
        },
        sleep: async () => undefined,
        now: () => timeNow,
      },
    )

    const result = await translator.translateText({
      text: "안녕",
      sourceLanguage: "ko",
      targetLanguage: "en",
      direction: "inbound",
    })

    expect(result.text).toBe("fallback-result")
    expect(result.modelUsed).toBe("openai/gpt-4o-mini")
    expect(calls).toBe(2)
  })

  test("both models fail within budget — returns original text without throwing", async () => {
    const translator = createTranslator(
      fakeClient([]),
      {
        translatorModel: "anthropic/claude-haiku-4-5",
        fallbackModel: "openai/gpt-4o-mini",
        triggerKeywords: ["$en"],
        sourceLanguage: "ko",
        displayLanguage: "ko",
        verbose: false,
        disableKeywords: ["$dis"],
        translateResponses: false,
      },
      {
        credentialResolver: makeCredentialResolver(),
        generateTextImpl: async () => {
          throw new Error("network error")
        },
        sleep: async () => undefined,
        now: (() => {
          let t = 0
          return () => (t += 10)
        })(),
      },
    )

    const result = await translator.translateText({
      text: "안녕",
      sourceLanguage: "ko",
      targetLanguage: "en",
      direction: "inbound",
    })

    // Must NOT throw; returns original text with passthrough sentinel
    expect(result.text).toBe("안녕")
    expect(result.modelUsed).toBe("passthrough")
  })

  test("fallback is skipped when remaining budget is ≤ 500ms — returns original text", async () => {
    let calls = 0
    let timeValue = 0
    const translator = createTranslator(
      fakeClient([]),
      {
        translatorModel: "anthropic/claude-haiku-4-5",
        fallbackModel: "openai/gpt-4o-mini",
        triggerKeywords: ["$en"],
        sourceLanguage: "ko",
        displayLanguage: "ko",
        verbose: false,
        disableKeywords: ["$dis"],
        translateResponses: false,
      },
      {
        credentialResolver: makeCredentialResolver(),
        generateTextImpl: async () => {
          calls += 1
          throw new Error("primary failed")
        },
        sleep: async () => undefined,
        now: () => {
          // Start at 0; after primary fails jump past total budget so remaining <= 500ms
          return calls >= 1 ? 15_000 : timeValue++
        },
      },
    )

    const result = await translator.translateText({
      text: "안녕",
      sourceLanguage: "ko",
      targetLanguage: "en",
      direction: "inbound",
    })

    // Fallback should be skipped; original text returned
    expect(result.text).toBe("안녕")
    expect(result.modelUsed).toBe("passthrough")
    // Only primary was called (1 call), fallback never attempted
    expect(calls).toBe(1)
  })
})
