import type { Plugin, PluginInput, PluginOptions } from "@opencode-ai/plugin"
import { createHooks } from "./activation"

const plugin: Plugin = async (ctx: PluginInput, options?: PluginOptions) =>
  createHooks(ctx, options ?? {})

export default plugin
export { plugin }
