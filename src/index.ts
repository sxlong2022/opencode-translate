import type { Plugin, PluginInput, PluginOptions, PluginModule } from "@opencode-ai/plugin"
import { createHooks } from "./activation"
import { writeFileSync } from "node:fs"

const debugLog = (msg: string) => {
  try { writeFileSync("C:\\Users\\sxlon\\AppData\\Local\\Temp\\opencode-translate-debug.log", `${new Date().toISOString()} ${msg}\n`, { flag: "a" }); } catch { }
}

debugLog("MODULE_LOADED")

const OpencodeTranslate: Plugin = async (ctx: PluginInput, options?: PluginOptions) => {
  debugLog(`FUNCTION_CALLED ctx.directory=${ctx.directory}`)
  try {
    const hooks = createHooks(ctx, options ?? {})
    debugLog(`HOOKS_REGISTERED keys=${Object.keys(hooks).join(",")}`)
    return hooks
  } catch (e: any) {
    debugLog(`INIT_ERROR ${e?.message ?? String(e)}`)
    throw e
  }
}

const pluginModule: PluginModule = {
  id: "opencode-translate",
  server: OpencodeTranslate,
}

export default pluginModule
