import { ApiProvider } from "@shared/api"

/**
 * Determines if reasoning content should be rendered as markdown based on model
 */
export function shouldUseMarkdownForReasoning(modelId: string): boolean {
	// DeepSeek R1 models output structured reasoning that benefits from markdown
	return modelId.toLowerCase().includes("deepseek-r1")
}
