/**
 * 翻译质量对比实验
 *
 * 测试模型:
 *   1. cloudflare/@cf/meta/llama-3.1-8b-instruct (基线)
 *   2. cloudflare/@cf/qwen/qwen3-30b-a3b-fp8
 *   3. cloudflare/@cf/zai-org/glm-4.7-flash
 *   4. nvidia-nim/qwen/qwen3.5-122b-a10b
 *
 * 评测维度:
 *   - 翻译速度 (首字延迟 TTFT + 总耗时)
 *   - Token 吞吐量 (tokens/sec)
 *   - 术语准确率 (关键术语是否正确翻译)
 *   - 语义保真度 (是否偏离原意)
 *   - 学术规范性 (是否符合英文学术写作惯例)
 */

const API_KEY = 'sk-7xK9mP2vR8nL5wQ3fJ6hD4aE1bC0gY7t'
const CF_BASE = 'https://one-balance.sxlong2013.workers.dev/api/cloudflare/v1'
const NIM_BASE = 'https://one-balance.sxlong2013.workers.dev/api/nvidia/v1'

const ACCOUNT_ID = '8022ee5c02b431fe9eeca2ef82b8767a'

const MODELS = [
  { id: 'qwen3.5-122b',             full: 'qwen/qwen3.5-122b-a10b',         base: NIM_BASE, key: API_KEY, label: 'Qwen3.5 122B (Nvidia)' },
  { id: 'llama-3.1-8b',             full: '@cf/meta/llama-3.1-8b-instruct', base: CF_BASE,  key: API_KEY, label: 'Llama 3.1 8B (CF)' },
  { id: 'gemma-4-26b-thinking',     full: '@cf/google/gemma-4-26b-a4b-it',  base: CF_BASE,  key: API_KEY, label: 'Gemma 4 26B (CF Thinking)' },
  {
    id: 'gemma-4-26b-non-thinking',
    full: '@cf/google/gemma-4-26b-a4b-it',
    base: CF_BASE,
    key: API_KEY,
    label: 'Gemma 4 26B (CF Non-Thinking)',
    extraParams: {
      reasoning_effort: null,
      chat_template_kwargs: { enable_thinking: false, thinking: false }
    }
  }
]

// ─── 测试用例 ───────────────────────────────────────────────────────────────

const TEST_CASES = [
  {
    category: '术语精准度',
    source: '本文采用基于Transformer架构的多头注意力机制，通过交叉熵损失函数优化模型参数，并使用AdamW优化器进行梯度下降。',
    reference: 'This paper employs a multi-head attention mechanism based on the Transformer architecture, optimizing model parameters via the cross-entropy loss function and performing gradient descent using the AdamW optimizer.',
    keyTerms: ['Transformer', 'multi-head attention', 'cross-entropy', 'AdamW', 'gradient descent'],
  },
  {
    category: '口语化表达',
    source: '说实话，这个模型跑起来太慢了，而且效果也没有论文里说的那么好，感觉有点被高估了。',
    reference: 'Honestly, this model runs too slowly, and its performance is not as good as claimed in the paper. It feels somewhat overrated.',
    keyTerms: [],
  },
  {
    category: '长段落',
    source: '大语言模型的出现彻底改变了人工智能的研究范式。从GPT-3到GPT-4，从LLaMA到Qwen，这些模型在文本生成、代码编写、数学推理等多个领域展现出了惊人的能力。然而，随之而来的伦理问题、安全风险和环境影响也引发了广泛的讨论。如何在推动技术进步的同时确保AI的负责任发展，已成为学术界 and 工业界共同关注的核心议题。本文旨在系统性地回顾大语言模型的发展历程，分析其技术原理 and 应用场景，并探讨未来的研究方向和潜在挑战。',
    reference: 'The emergence of large language models has fundamentally transformed the research paradigm in artificial intelligence. From GPT-3 to GPT-4, from LLaMA to Qwen, these models have demonstrated remarkable capabilities across multiple domains including text generation, code writing, and mathematical reasoning. However, the accompanying ethical concerns, safety risks, and environmental impacts have also sparked widespread discussion. How to ensure responsible AI development while advancing technology has become a core issue of shared concern across academia and industry. This paper aims to systematically review the development history of large language models, analyze their technical principles and application scenarios, and explore future research directions and potential challenges.',
    keyTerms: ['large language models', 'GPT-3', 'GPT-4', 'LLaMA', 'Qwen', 'text generation', 'mathematical reasoning', 'responsible AI'],
  },
]

// ─── 工具函数 ───────────────────────────────────────────────────────────────

async function translate(model, text) {
  const url = `${model.base}/chat/completions`
  const headers = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${model.key}`,
  }
const body = JSON.stringify({
model: model.full,
messages: [
{
role: 'system',
content:
'You are a professional academic translator. Translate the following Chinese text into fluent, accurate, publication-ready English. Preserve all technical terms, proper nouns, numbers, and formatting exactly. Output ONLY the translation, nothing else.',
},
{ role: 'user', content: text },
],
temperature: 0.3,
max_tokens: 2048,
    stream: true,
    ...(model.extraParams || {
  }),
  })

  const start = performance.now()
  const resp = await fetch(url, { method: 'POST', headers, body, verbose: true })

  if (!resp.ok) {
    const errText = await resp.text().catch(() => '(no body)')
    throw new Error(`HTTP ${resp.status}: ${errText.slice(0, 200)}`)
  }

  // Stream-parse SSE to get text + count tokens
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let textOut = ''
  let tokenCount = 0
  let ttft = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data:')) continue
      const colonIdx = trimmed.indexOf(':')
      const dataStr = trimmed.substring(colonIdx + 1).trim()
      if (dataStr === '[DONE]') continue
      try {
        const data = JSON.parse(dataStr)
        const delta = data.choices?.[0]?.delta?.content
        if (typeof delta === 'string' && delta.length > 0) {
          if (ttft === null) ttft = performance.now() - start
          textOut += delta
          tokenCount++
        }
      } catch {
        // ignore parse errors
      }
    }
  }

  const totalMs = performance.now() - start
  return {
    text: textOut.trim(),
    ttft: ttft ?? totalMs,
    totalMs,
    tokenCount,
    tokensPerSec: tokenCount / (totalMs / 1000),
  }
}
async function translateWithRetry(model, text, maxRetries = 3) {
  let lastError = null
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      if (attempt > 1) {
        process.stdout.write(`(重试 #${attempt})... `)
        await new Promise((resolve) => setTimeout(resolve, 2000 * Math.pow(2, attempt - 2)))
      }
      return await translate(model, text)
    } catch (e) {
      lastError = e
    }
  }
  throw lastError
}

function scoreTerms(text, keyTerms) {
  if (keyTerms.length === 0) return null
  const lower = text.toLowerCase()
  const hits = keyTerms.filter((t) => lower.includes(t.toLowerCase()))
  return { hits: hits.length, total: keyTerms.length, pct: Math.round((hits.length / keyTerms.length) * 100) }
}

// ─── 主流程 ─────────────────────────────────────────────────────────────────

console.log('=' .repeat(60))
console.log('  翻译质量对比实验')
console.log('=' .repeat(60))
console.log(`  测试模型: ${MODELS.map((m) => m.label).join(', ')}`)
console.log(`  测试用例: ${TEST_CASES.length} 条`)
console.log('=' .repeat(60))

const results = []

for (const model of MODELS) {
  console.log(`>>> 启动模型测试: ${model.label}`)
  for (let i = 0; i < TEST_CASES.length; i++) {
    const tc = TEST_CASES[i]
    try {
      const r = await translateWithRetry(model, tc.source)
      const terms = scoreTerms(r.text, tc.keyTerms)
      const termStr = terms ? ` | 术语 ${terms.hits}/${terms.total}` : ''
      console.log(`  [${model.label}] [${i + 1}/${TEST_CASES.length}] ${tc.category} 完成: ${Math.round(r.totalMs)}ms, ${r.tokenCount} tok${termStr}`)
      results.push({ model: model.label, category: tc.category, ...r, terms, source: tc.source, reference: tc.reference, error: null })
    } catch (e) {
      console.log(`  [${model.label}] [${i + 1}/${TEST_CASES.length}] ${tc.category} 失败: ${e.message}`)
      results.push({ model: model.label, category: tc.category, text: null, ttft: 0, totalMs: 0, tokenCount: 0, tokensPerSec: 0, terms: null, source: tc.source, reference: tc.reference, error: e.message })
    }
  }
}

// ─── 生成报告 ───────────────────────────────────────────────────────────────

const sep = '=' .repeat(60)
let md = `# 翻译质量对比实验报告\n\n`
md += `**日期**: ${new Date().toLocaleDateString('zh-CN')}\n\n`
md += `**测试模型**:\n`
for (const m of MODELS) md += `- ${m.label}\n`
md += `\n**测试用例**: ${TEST_CASES.length} 条（涵盖术语精准度、复杂长句、研究方法、学术论证、口语化表达、中英混合、短文本、长段落）\n\n`
md += `**实验参数**: temperature=0.3, max_tokens=2048, stream=true\n\n`

// --- 模型总览表 ---
md += `## 模型总览\n\n`
md += `| 模型 | 平均耗时 | 平均TTFT | 平均吞吐 | 术语准确率 |\n`
md += `|------|----------|----------|----------|------------|\n`

for (const model of MODELS) {
  const mr = results.filter((r) => r.model === model.label && !r.error)
  if (mr.length === 0) {
    md += `| ${model.label} | ERROR | - | - | - |\n`
    continue
  }
  const avgMs = Math.round(mr.reduce((s, r) => s + r.totalMs, 0) / mr.length)
  const avgTtft = Math.round(mr.reduce((s, r) => s + r.ttft, 0) / mr.length)
  const avgTps = (mr.reduce((s, r) => s + r.tokensPerSec, 0) / mr.length).toFixed(1)
  const allTerms = mr.filter((r) => r.terms)
  const termPct = allTerms.length > 0
    ? Math.round(allTerms.reduce((s, r) => s + r.terms.pct, 0) / allTerms.length)
    : '-'
  md += `| ${model.label} | ${avgMs}ms | ${avgTtft}ms | ${avgTps} tok/s | ${typeof termPct === 'number' ? termPct + '%' : termPct} |\n`
}

// --- 逐条对比表 ---
md += `\n## 逐条翻译对比\n\n`

for (let i = 0; i < TEST_CASES.length; i++) {
  const tc = TEST_CASES[i]
  md += `### ${i + 1}. ${tc.category}\n\n`
  md += `**原文**: ${tc.source}\n\n`
  md += `**参考译文**: ${tc.reference}\n\n`
  md += `| 模型 | 耗时 | TTFT | Tokens | 术语 | 译文 |\n`
  md += `|------|------|------|--------|------|------|\n`

  for (const model of MODELS) {
    const r = results.find((r2) => r2.model === model.label && r2.category === tc.category)
    if (!r || r.error) {
      md += `| ${model.label} | ERROR | - | - | - | ${r?.error || 'unknown'} |\n`
      continue
    }
    const termStr = r.terms ? `${r.terms.hits}/${r.terms.total} (${r.terms.pct}%)` : '-'
    const translation = r.text.replace(/\n/g, ' ').replace(/\|/g, '\\|')
    md += `| ${model.label} | ${Math.round(r.totalMs)}ms | ${Math.round(r.ttft)}ms | ${r.tokenCount} | ${termStr} | ${translation} |\n`
  }
  md += `\n`
}

// --- 人工评分模板 ---
md += `## 人工评分表\n\n`
md += `请为每条翻译打分（1-5分）:\n`
md += `- 5分: 准确、流畅、符合学术规范\n`
md += `- 4分: 基本准确，个别表达可优化\n`
md += `- 3分: 大意正确，但有明显不自然之处\n`
md += `- 2分: 部分偏离原意\n`
md += `- 1分: 严重偏离原意或无法理解\n\n`
md += `| 测试用例 | ${MODELS.map((m) => m.label).join(' | ')} |\n`
md += `|----------|${MODELS.map(() => '------').join('|')}|\n`
for (let i = 0; i < TEST_CASES.length; i++) {
  md += `| ${TEST_CASES[i].category} ${i + 1} | ${MODELS.map(() => '').join(' | ')} |\n`
}
md += `\n`

// --- 结论建议 ---
md += `## 结论与建议\n\n`
md += `根据实验结果，推荐用于翻译任务的模型:\n\n`
md += `1. **速度优先**: 查看"平均耗时"和"平均TTFT"列\n`
md += `2. **质量优先**: 查看"术语准确率"列 + 人工评分\n`
md += `3. **综合推荐**: 在速度和质量之间取得最佳平衡的模型\n\n`
md += `---\n\n`
md += `*本报告由翻译对比实验脚本自动生成*\n`

const reportPath = 'D:/Personal_project/opencode-translate/translation-experiment-report.md'
import { writeFileSync } from 'fs'
writeFileSync(reportPath, md, 'utf-8')

console.log(`\n${sep}`)
console.log('  实验完成!')
console.log(`${sep}`)
console.log(`  报告已保存: ${reportPath}`)
console.log(`  请打开报告查看详细对比结果，并进行人工评分。`)
console.log(`${sep}`)
