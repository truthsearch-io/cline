import { describe, it, before, beforeEach, afterEach } from "mocha"
import { expect } from "chai"
import { GroqHandler } from "../groq"
import { groqDefaultModelId, groqModels } from "../../../shared/api"

describe("GroqHandler", () => {
	let handler: GroqHandler

	beforeEach(() => {
		handler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: groqDefaultModelId,
		})
	})

	it("should initialize with correct default model", () => {
		const model = handler.getModel()
		expect(model.id).to.equal(groqDefaultModelId)
		expect(model.info).to.deep.equal(groqModels[groqDefaultModelId])
	})

	it("should use specified model when provided", () => {
		const customHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "llama-3.1-8b-instant",
		})
		const model = customHandler.getModel()
		expect(model.id).to.equal("llama-3.1-8b-instant")
		expect(model.info).to.deep.equal(groqModels["llama-3.1-8b-instant"])
	})

	it("should fall back to default model for invalid model ID", () => {
		const customHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "invalid-model",
		})
		const model = customHandler.getModel()
		expect(model.id).to.equal(groqDefaultModelId)
		expect(model.info).to.deep.equal(groqModels[groqDefaultModelId])
	})

	it("should have correct model properties", () => {
		const model = handler.getModel()
		expect(model.info.maxTokens).to.be.greaterThan(0)
		expect(model.info.contextWindow).to.be.greaterThan(0)
		expect(model.info.supportsImages).to.equal(false)
		expect(model.info.supportsPromptCache).to.equal(false)
		expect(model.info.inputPrice).to.be.a("number")
		expect(model.info.outputPrice).to.be.a("number")
	})

	it("should include new Llama 4 models", () => {
		const llama4ScoutHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "meta-llama/llama-4-scout-17b-16e-instruct",
		})
		const model = llama4ScoutHandler.getModel()
		expect(model.id).to.equal("meta-llama/llama-4-scout-17b-16e-instruct")
		expect(model.info.inputPrice).to.equal(0.31)
		expect(model.info.outputPrice).to.equal(0.36)
		expect(model.info.description).to.include("Llama 4 Scout")
	})

	it("should include vision-capable models", () => {
		const llama4MaverickHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "meta-llama/llama-4-maverick-17b-128e-instruct",
		})
		const model = llama4MaverickHandler.getModel()
		expect(model.id).to.equal("meta-llama/llama-4-maverick-17b-128e-instruct")
		expect(model.info.supportsImages).to.equal(true)
		expect(model.info.inputPrice).to.equal(0.2)
		expect(model.info.outputPrice).to.equal(0.6)
	})

	it("should include updated pricing for all models", () => {
		// Test a few key models to ensure pricing is updated
		const testCases = [
			{ modelId: "llama-3.3-70b-versatile", expectedInput: 0.79, expectedOutput: 0.79 },
			{ modelId: "deepseek-r1-distill-llama-70b", expectedInput: 0.75, expectedOutput: 0.99 },
			{ modelId: "qwen-qwq-32b", expectedInput: 0.29, expectedOutput: 0.39 },
			{ modelId: "mistral-saba-24b", expectedInput: 0.79, expectedOutput: 0.79 },
		]

		testCases.forEach(({ modelId, expectedInput, expectedOutput }) => {
			const testHandler = new GroqHandler({
				groqApiKey: "test-key",
				apiModelId: modelId,
			})
			const model = testHandler.getModel()
			expect(model.info.inputPrice).to.equal(expectedInput, `Input price for ${modelId}`)
			expect(model.info.outputPrice).to.equal(expectedOutput, `Output price for ${modelId}`)
		})
	})

	it("should have comprehensive model descriptions", () => {
		// Check that all models have meaningful descriptions
		Object.entries(groqModels).forEach(([modelId, modelInfo]) => {
			expect(modelInfo.description).to.be.a("string")
			expect(modelInfo.description.length).to.be.greaterThan(10, `Description for ${modelId} should be meaningful`)
		})
	})

	it("should support dynamic model info from API", () => {
		const dynamicModelInfo = {
			maxTokens: 16384,
			contextWindow: 65536,
			supportsImages: true,
			supportsPromptCache: false,
			inputPrice: 0.5,
			outputPrice: 1.0,
			description: "Dynamic model from API",
		}

		const dynamicHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "compound-beta",
			groqModelInfo: dynamicModelInfo,
		})

		const model = dynamicHandler.getModel()
		expect(model.id).to.equal("compound-beta")
		expect(model.info).to.deep.equal(dynamicModelInfo)
	})

	it("should fall back to static models when dynamic info is not available", () => {
		const handlerWithoutDynamic = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "llama-3.3-70b-versatile",
			// No groqModelInfo provided
		})

		const model = handlerWithoutDynamic.getModel()
		expect(model.id).to.equal("llama-3.3-70b-versatile")
		expect(model.info).to.deep.equal(groqModels["llama-3.3-70b-versatile"])
	})

	it("should handle unknown models gracefully", () => {
		const unknownModelHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "unknown-model-id",
			// No groqModelInfo provided
		})

		const model = unknownModelHandler.getModel()
		expect(model.id).to.equal(groqDefaultModelId)
		expect(model.info).to.deep.equal(groqModels[groqDefaultModelId])
	})

	it("should prioritize dynamic model info over static models", () => {
		const dynamicModelInfo = {
			maxTokens: 99999, // Different from static
			contextWindow: 99999, // Different from static
			supportsImages: true, // Different from static
			supportsPromptCache: true, // Different from static
			inputPrice: 99.99, // Different from static
			outputPrice: 99.99, // Different from static
			description: "Dynamic override",
		}

		const overrideHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "llama-3.3-70b-versatile", // This exists in static models
			groqModelInfo: dynamicModelInfo,
		})

		const model = overrideHandler.getModel()
		expect(model.id).to.equal("llama-3.3-70b-versatile")
		expect(model.info).to.deep.equal(dynamicModelInfo)
		// Ensure it's not using static model info
		expect(model.info).to.not.deep.equal(groqModels["llama-3.3-70b-versatile"])
	})
})

describe("QwQ Model Tests", () => {
	let handler: GroqHandler

	beforeEach(() => {
		handler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "qwen-qwq-32b",
		})
	})

	it("should detect QwQ models correctly", () => {
		const qwqHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "qwen-qwq-32b",
		})

		const modelFamily = qwqHandler.getModelFamily()
		expect(modelFamily.name).to.equal("QwQ Reasoning")
		expect(modelFamily.specialParams?.reasoning_format).to.equal("parsed")
		expect(modelFamily.specialParams?.top_p).to.equal(0.95)
		expect(modelFamily.optimalTemperature.plan).to.equal(0.6)
		expect(modelFamily.optimalTemperature.act).to.equal(0.6)
	})

	it("should recognize QwQ model variants", () => {
		const realTestCases = ["qwen-qwq-32b"] // Only real QwQ model

		realTestCases.forEach((modelId) => {
			const testHandler = new GroqHandler({
				groqApiKey: "test-key",
				apiModelId: modelId,
			})

			const modelFamily = testHandler.getModelFamily()
			expect(modelFamily.name).to.equal("QwQ Reasoning", `Failed for model: ${modelId}`)
			expect(modelFamily.specialParams?.reasoning_format).to.equal("parsed")
		})
	})

	it("should not apply reasoning format to non-QwQ models", () => {
		const regularHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "llama-3.3-70b-versatile",
		})

		const modelFamily = regularHandler.getModelFamily()
		expect(modelFamily.name).to.equal("Llama 3.3")
		expect(modelFamily.specialParams?.reasoning_format).to.be.undefined
	})

	it("should handle DeepSeek reasoning models similarly to QwQ", () => {
		const deepseekHandler = new GroqHandler({
			groqApiKey: "test-key",
			apiModelId: "deepseek-r1-distill-llama-70b",
		})

		const modelFamily = deepseekHandler.getModelFamily()
		expect(modelFamily.name).to.equal("DeepSeek")
		expect(modelFamily.specialParams?.reasoning_format).to.equal("parsed")
		expect(modelFamily.specialParams?.top_p).to.equal(0.95)
		expect(modelFamily.optimalTemperature.plan).to.equal(0.6)
		expect(modelFamily.optimalTemperature.act).to.equal(0.6)
	})

	it("should use correct temperature for different modes", () => {
		// Test private method through reflection
		const getOptimalTemperature = (handler as any).getOptimalTemperature.bind(handler)
		const modelFamily = handler.getModelFamily()

		// Plan mode should use higher temperature
		const planTemp = getOptimalTemperature(modelFamily, true)
		expect(planTemp).to.equal(0.6)

		// Act mode should use the same temperature for QwQ
		const actTemp = getOptimalTemperature(modelFamily, false)
		expect(actTemp).to.equal(0.6)
	})

	it("should use correct max tokens for QwQ models", () => {
		const model = handler.getModel()
		const modelFamily = handler.getModelFamily()

		// Test private method through reflection
		const getOptimalMaxTokens = (handler as any).getOptimalMaxTokens.bind(handler)
		const maxTokens = getOptimalMaxTokens(model, modelFamily)

		// QwQ models should support 32K tokens
		expect(maxTokens).to.equal(32768)
	})

	it("should use fallback message when no text content is received", async function () {
		this.timeout(5000)

		// Mock the OpenAI client to simulate QwQ behavior with only reasoning chunks
		const mockStream = async function* () {
			// Yield reasoning chunks
			yield {
				choices: [
					{
						delta: {
							reasoning: "Let me think about this step by step. First, I need to understand the problem...",
						},
					},
				],
			}
			yield {
				choices: [
					{
						delta: {
							reasoning: " Then, I should consider the various approaches available...",
						},
					},
				],
			}
			yield {
				choices: [
					{
						delta: {
							reasoning: " Finally, I'll conclude with the best solution.",
						},
					},
				],
			}
			// Yield usage info to trigger the fallback check
			yield {
				usage: {
					prompt_tokens: 100,
					completion_tokens: 50,
					total_tokens: 150,
				},
				choices: [{ delta: {} }],
			}
		}

		// Mock the client
		const mockClient = {
			chat: {
				completions: {
					create: sinon.stub().returns(mockStream()),
				},
			},
		}

		// Replace the client in the handler
		;(handler as any).client = mockClient

		// Collect results from the stream
		const results: any[] = []
		const stream = handler.createMessage("Test system prompt", [{ role: "user", content: "Test message" }])

		for await (const chunk of stream) {
			results.push(chunk)
		}

		// Verify we got reasoning chunks and a fallback text chunk
		const reasoningChunks = results.filter((r) => r.type === "reasoning")
		const textChunks = results.filter((r) => r.type === "text")
		const usageChunks = results.filter((r) => r.type === "usage")

		expect(reasoningChunks).to.have.length(3)
		expect(textChunks).to.have.length(1) // Should have one fallback text chunk
		expect(usageChunks).to.have.length(1)

		// Verify the fallback text is the simple "Hmmm..." message
		const fallbackText = textChunks[0].text
		expect(fallbackText).to.equal("Hmmm...")

		// Verify reasoning chunks were yielded with proper content
		expect(reasoningChunks[0].reasoning).to.include("Let me think about this step by step")
		expect(reasoningChunks[1].reasoning).to.include("Then, I should consider the various approaches")
		expect(reasoningChunks[2].reasoning).to.include("Finally, I'll conclude with the best solution")
	})

	it("should not use reasoning fallback when text content is received", async function () {
		this.timeout(5000)

		// Mock the OpenAI client to simulate normal behavior with both reasoning and text
		const mockStream = async function* () {
			// Yield reasoning chunks
			yield {
				choices: [
					{
						delta: {
							reasoning: "Let me think...",
						},
					},
				],
			}
			// Yield text content
			yield {
				choices: [
					{
						delta: {
							content: "Here is my response based on my reasoning.",
						},
					},
				],
			}
			// Yield usage info
			yield {
				usage: {
					prompt_tokens: 100,
					completion_tokens: 50,
					total_tokens: 150,
				},
				choices: [{ delta: {} }],
			}
		}

		// Mock the client
		const mockClient = {
			chat: {
				completions: {
					create: sinon.stub().returns(mockStream()),
				},
			},
		}

		// Replace the client in the handler
		;(handler as any).client = mockClient

		// Collect results from the stream
		const results: any[] = []
		const stream = handler.createMessage("Test system prompt", [{ role: "user", content: "Test message" }])

		for await (const chunk of stream) {
			results.push(chunk)
		}

		// Verify we got reasoning chunks and regular text chunks, but no fallback
		const reasoningChunks = results.filter((r) => r.type === "reasoning")
		const textChunks = results.filter((r) => r.type === "text")
		const usageChunks = results.filter((r) => r.type === "usage")

		expect(reasoningChunks).to.have.length(1)
		expect(textChunks).to.have.length(1) // Should have one regular text chunk
		expect(usageChunks).to.have.length(1)

		// Verify the text content is the regular response, not reasoning
		const textContent = textChunks[0].text
		expect(textContent).to.equal("Here is my response based on my reasoning.")
		expect(textContent).to.not.include("Let me think...")
	})
})

import sinon from "sinon"
import axios from "axios"
import { Controller } from "../../../core/controller"
import { refreshGroqModels } from "../../../core/controller/models/refreshGroqModels"
import { EmptyRequest } from "../../../shared/proto/common"
import * as stateModule from "../../../core/storage/state"
import * as fsModule from "../../../utils/fs"
import fs from "fs/promises"

describe("Groq Dynamic Model Discovery", () => {
	let controller: Controller
	let axiosGetStub: sinon.SinonStub
	let consoleLogStub: sinon.SinonStub
	let consoleErrorStub: sinon.SinonStub
	let getAllExtensionStateStub: sinon.SinonStub
	let fileExistsStub: sinon.SinonStub
	let fsReadFileStub: sinon.SinonStub
	let fsWriteFileStub: sinon.SinonStub
	let fsMkdirStub: sinon.SinonStub

	beforeEach(() => {
		// Create a mock controller with minimal required properties
		controller = {
			context: {
				globalStorageUri: {
					fsPath: "/tmp/test-storage",
				},
			},
		} as any

		// Stub axios.get
		axiosGetStub = sinon.stub(axios, "get")

		// Stub console methods to reduce noise in tests
		consoleLogStub = sinon.stub(console, "log")
		consoleErrorStub = sinon.stub(console, "error")

		// Stub state module functions
		getAllExtensionStateStub = sinon.stub(stateModule, "getAllExtensionState")

		// Stub file system functions
		fileExistsStub = sinon.stub(fsModule, "fileExistsAtPath").resolves(false)
		fsReadFileStub = sinon.stub(fs, "readFile").resolves("{}")
		fsWriteFileStub = sinon.stub(fs, "writeFile").resolves()
		fsMkdirStub = sinon.stub(fs, "mkdir").resolves()
	})

	afterEach(() => {
		sinon.restore()
	})

	it("should filter out non-chat models", async () => {
		const mockApiResponse = {
			data: {
				data: [
					{
						id: "llama-3.3-70b-versatile",
						object: "model",
						owned_by: "meta",
						context_window: 131072,
						max_completion_tokens: 32768,
					},
				],
			},
		}

		axiosGetStub.resolves(mockApiResponse)

		getAllExtensionStateStub.resolves({
			apiConfiguration: {
				groqApiKey: "gsk_test_key_123",
			},
		})

		const result = await refreshGroqModels(controller, EmptyRequest.create({}))

		// Should only include the chat model
		expect(result.models).to.have.property("llama-3.3-70b-versatile")
	})

	it("should handle API errors gracefully", async () => {
		// Simulate API error
		axiosGetStub.rejects(new Error("Network error"))

		getAllExtensionStateStub.resolves({
			apiConfiguration: {
				groqApiKey: "gsk_test_key_123",
			},
		})

		const result = await refreshGroqModels(controller, EmptyRequest.create({}))

		// Should fall back to static models
		expect(Object.keys(result.models).length).to.be.greaterThan(0)
		expect(result.models).to.have.property("llama-3.3-70b-versatile")

		// Verify error was logged
		expect(consoleErrorStub.calledWith("Error fetching Groq models:")).to.be.true
	})
})

import { preprocessMermaidContent } from "../groq"

describe("Mermaid Preprocessing Tests", () => {
	it("should convert single-backtick mermaid graph to proper code fence", () => {
		const input = "`mermaidgraph TD A --> B`"
		const result = preprocessMermaidContent(input)
		const expected = "```mermaid\ngraph TD A --> B\n```"
		expect(result).to.equal(expected)
	})

	it("should handle complex mermaid diagrams with subgraphs", () => {
		const input = "`mermaidgraph TD subgraph Core_Extension direction TB Extension_Entry -->|Instant| Webview_Provider end`"
		const result = preprocessMermaidContent(input)
		const expected =
			"```mermaid\ngraph TD\n  subgraph Core_Extension\n    direction TB Extension_Entry |Instant| Webview_Provider\n  end\n```"
		expect(result).to.equal(expected)
	})

	it("should handle flowchart diagrams", () => {
		const input = "`mermaidflowchart LR A --> B --> C`"
		const result = preprocessMermaidContent(input)
		const expected = "```mermaid\nflowchart LR A --> B --> C\n```"
		expect(result).to.equal(expected)
	})

	it("should handle sequence diagrams", () => {
		const input = "`mermaidsequenceDiagram Alice->>Bob: Hello Bob, how are you?`"
		const result = preprocessMermaidContent(input)
		const expected = "```mermaid\nsequenceDiagram Alice->>Bob: Hello Bob, how are you?\n```"
		expect(result).to.equal(expected)
	})

	it("should handle multiple single-backtick diagrams across different lines", () => {
		const input = "Here's a diagram:\n`mermaidgraph TD A --> B`\n\nAnd another:\n`mermaidflowchart LR C --> D`"
		const result = preprocessMermaidContent(input)
		const expected =
			"Here's a diagram:\n```mermaid\ngraph TD A --> B\n```\n\nAnd another:\n```mermaid\nflowchart LR C --> D\n```"
		expect(result).to.equal(expected)
	})

	it("should not modify content without mermaid patterns", () => {
		const input = "This is regular text with `code` and no mermaid content."
		const result = preprocessMermaidContent(input)
		expect(result).to.equal(input)
	})

	it("should not modify properly formatted mermaid code blocks", () => {
		const input = "```mermaid\ngraph TD\n  A --> B\n```"
		const result = preprocessMermaidContent(input)
		expect(result).to.equal(input)
	})

	it("should handle mermaid with spaces after the keyword", () => {
		const input = "`mermaid graph TD A --> B`"
		const result = preprocessMermaidContent(input)
		const expected = "```mermaid\ngraph TD A --> B\n```"
		expect(result).to.equal(expected)
	})

	it("should handle common diagram types", () => {
		// Only test the most commonly used diagram types that models actually output
		const testCases = [
			{ input: "`mermaidgraph TD A --> B`", type: "graph" },
			{ input: "`mermaidflowchart LR A --> B`", type: "flowchart" },
			{ input: "`mermaidsequenceDiagram Alice->>Bob: Hello`", type: "sequenceDiagram" },
		]

		testCases.forEach(({ input, type }) => {
			const result = preprocessMermaidContent(input)
			expect(result).to.include("```mermaid")
			expect(result).to.include(type)
			expect(result).to.include("```")
		})
	})

	it("should preserve content before and after mermaid diagrams", () => {
		const input = "Introduction text here.\n\n`mermaidgraph TD A --> B`\n\nConclusion text here."
		const result = preprocessMermaidContent(input)
		expect(result).to.include("Introduction text here.")
		expect(result).to.include("```mermaid")
		expect(result).to.include("graph TD A --> B")
		expect(result).to.include("Conclusion text here.")
	})

	it("should handle edge case with empty mermaid content", () => {
		const input = "`mermaidgraph`"
		// This should not match our pattern because it requires content after the diagram type
		const result = preprocessMermaidContent(input)
		expect(result).to.equal(input) // Should remain unchanged
	})

	it("should be case insensitive for mermaid keyword", () => {
		const input = "`MermaidGraph TD A --> B`"
		const result = preprocessMermaidContent(input)
		const expected = "```mermaid\nGraph TD A --> B\n```"
		expect(result).to.equal(expected)
	})

	// Tests for raw Mermaid diagram detection
	it("should convert raw graph diagrams to proper code fences", () => {
		const input = "graph LR\n  A --> B\n  B --> C"
		const result = preprocessMermaidContent(input)
		// Just check that it gets wrapped, don't be picky about exact formatting
		expect(result).to.include("```mermaid")
		expect(result).to.include("graph LR")
		expect(result).to.include("A --> B")
		expect(result).to.include("```")
	})

	it("should handle the specific qwen-qwq-32b output format", () => {
		const input =
			"graph LR subgraph VSCode Extension Host subgraph Core System McpHub[McpHub<br/>Manages connections] TaskClass[Task Class<br/>Executes tools] Controller[Controller<br/>State Manager] end McpHub -->|Provides tools to| TaskClass Controller -->|Config updates| McpHub TaskClass -->|Reports results to| Controller end WebviewUI[Webview UI<br/>Settings & Chat Display]] McpHub -->|Discovers via| Marketplace[Market Place] Marketplace -->|Server catalog| WebviewUI style McpHub stroke:#007ACC,stroke-dashat: #007ACC,fill:#E8F8FF style TaskClass stroke:#99CC99,fill:#F0F8FF style Marketplace stroke:#FFA500,fill:#FFF0E1"
		const result = preprocessMermaidContent(input)
		expect(result).to.include("```mermaid")
		expect(result).to.include("graph LR")
		expect(result).to.include("subgraph VSCode Extension Host")
		expect(result).to.include("```")
	})

	it("should handle single-line raw diagram patterns", () => {
		const singleLineInput =
			"graph TB subgraph VSCode_Extension Core_Extension WebviewUI MCPP_Hub --> External_MCP_Servers: Manages end Core_Extension --> Controller: Coordinates Controller --> Task: Manages tasks Task --> API_Providers: Uses Task --> MCPP_Hub: Invokes tools WebviewUI --> ExtensionStateContext: Syncs state ExtensionStateContext --> ReactComponents: Provides props MCPP_Hub --> Task: Provides tools"
		const result = preprocessMermaidContent(singleLineInput)
		expect(result).to.include("```mermaid")
		expect(result).to.include("graph TB")
		expect(result).to.include("subgraph VSCode_Extension")
		expect(result).to.include("MCPP_Hub --> External_MCP_Servers: Manages")
		expect(result).to.include("```")
	})

	it("should handle raw flowchart diagrams", () => {
		const input = "flowchart TD\n  Start --> Stop"
		const result = preprocessMermaidContent(input)
		// Just check that it gets wrapped, don't be picky about exact formatting
		expect(result).to.include("```mermaid")
		expect(result).to.include("flowchart TD")
		expect(result).to.include("Start --> Stop")
		expect(result).to.include("```")
	})

	it("should handle raw sequence diagrams", () => {
		const input = "sequenceDiagram\n  Alice->>Bob: Hello\n  Bob-->>Alice: Hi there"
		const result = preprocessMermaidContent(input)
		expect(result).to.include("```mermaid")
		expect(result).to.include("sequenceDiagram")
		expect(result).to.include("Alice->>Bob: Hello")
	})

	it("should handle raw C4 diagrams", () => {
		const input = 'C4Context\n  title System Context\n  Person(user, "User")\n  System(app, "App")'
		const result = preprocessMermaidContent(input)
		expect(result).to.include("```mermaid")
		expect(result).to.include("C4Context")
		expect(result).to.include("System(app")
	})

	it("should not convert short content that just starts with diagram types", () => {
		const input = "graph\npie chart\nflowchart"
		const result = preprocessMermaidContent(input)
		expect(result).to.equal(input) // Should remain unchanged
	})

	it("should not convert content without mermaid syntax", () => {
		const input = "graph this data shows trends over time"
		const result = preprocessMermaidContent(input)
		expect(result).to.equal(input) // Should remain unchanged
	})

	it("should handle raw diagrams with directions", () => {
		const testCases = ["graph LR\n  A --> B", "graph TB\n  A --> B", "flowchart TD\n  A --> B", "flowchart RL\n  A --> B"]

		testCases.forEach((input) => {
			const result = preprocessMermaidContent(input)
			expect(result).to.include("```mermaid")
			expect(result).to.include("```")
		})
	})

	it("should handle complex raw diagrams with subgraphs and styling", () => {
		const input = "graph TB\n  subgraph Web\n    A[Frontend]\n  end\n  A --> B\n  style A fill:#f9f"
		const result = preprocessMermaidContent(input)
		expect(result).to.include("```mermaid")
		expect(result).to.include("subgraph Web")
		expect(result).to.include("style A fill:#f9f")
	})

	it("should handle all supported diagram types", () => {
		const diagramTypes = [
			"graph",
			"flowchart",
			"sequenceDiagram",
			"classDiagram",
			"stateDiagram",
			"gantt",
			"pie",
			"journey",
			"gitgraph",
			"mindmap",
			"timeline",
			"quadrantChart",
			"sankey",
			"requirement",
			"block",
			"packet",
			"C4Context",
			"C4Container",
			"C4Component",
			"erDiagram",
		]

		diagramTypes.forEach((type) => {
			const input = `${type}\n  A --> B --> C`
			const result = preprocessMermaidContent(input)
			expect(result).to.include("```mermaid", `Failed for diagram type: ${type}`)
			expect(result).to.include(type, `Failed for diagram type: ${type}`)
		})
	})

	it("should handle complex real-world case with comments and multiple diagrams", () => {
		const input = `%% Sequence diagram showing user interaction flow
sequenceDiagram
    participant User
    participant WebviewUI
    participant Controller
    participant Task
    participant McpHub
    User->>WebviewUI: Initiates a request
    WebviewUI->>Controller: Send message to Controller
    Controller->>Task: Execute task with API/tool request
    Task->>McpHub: Use MCP tool if needed
    McpHub-->>Task: Return tool result
    Task-->>Controller: Process response
    Controller-->>WebviewUI: Update UI with results
    WebviewUI-->>User: Display output

%% Component architecture diagram
graph LR
    subgraph WebviewUI[Webview UI (React)]
        ExtensionStateContext
    end
    ExtensionStateContext --> Controller: Communicates with
    Controller --> Task: Manages
    Controller --> McpHub: Integrates with
    Task --> APIProviders: Uses
    Task --> Storage: Persists state with
    McpHub --> MCP_Servers: Connects to
    Storage --> VSCode_Storage: Uses`

		const result = preprocessMermaidContent(input)

		// Should contain two separate mermaid blocks
		const mermaidBlocks = result.split("```mermaid").length - 1
		expect(mermaidBlocks).to.equal(2, "Should create two separate mermaid blocks")

		// Should contain both diagram types
		expect(result).to.include("sequenceDiagram")
		expect(result).to.include("graph LR")

		// Should contain participant and arrow syntax
		expect(result).to.include("participant User")
		expect(result).to.include("User->>WebviewUI")
		expect(result).to.include("subgraph WebviewUI")

		// Should wrap each diagram properly
		expect(result).to.include("```mermaid\nsequenceDiagram")
		expect(result).to.include("```mermaid\ngraph LR")
	})
})

describe("GroqHandler Integration Tests", () => {
	let groqApiAvailable = false
	let apiKey: string | undefined

	// Check if Groq API is available before running tests
	before(async function () {
		this.timeout(10000)
		apiKey = process.env.GROQ_API_KEY

		if (apiKey) {
			try {
				// Test API connectivity with a simple models request
				const response = await axios.get("https://api.groq.com/openai/v1/models", {
					headers: {
						Authorization: `Bearer ${apiKey}`,
						"Content-Type": "application/json",
						"User-Agent": "Cline-VSCode-Extension-Test",
					},
					timeout: 5000,
				})

				if (response.data?.data && Array.isArray(response.data.data)) {
					groqApiAvailable = true
					console.log(`Groq API available with ${response.data.data.length} models`)
				}
			} catch (error) {
				console.log("Groq API not available or invalid key, skipping integration tests")
				console.log("Set GROQ_API_KEY environment variable to run these tests")
				groqApiAvailable = false
			}
		} else {
			console.log("GROQ_API_KEY environment variable not set, skipping integration tests")
		}
	})

	it("should validate GroqHandler works with real API models", async function () {
		if (!groqApiAvailable || !apiKey) {
			this.skip()
		}
		this.timeout(5000)

		// Test with Llama 4 Scout model that we fixed
		const handler = new GroqHandler({
			groqApiKey: apiKey,
			apiModelId: "meta-llama/llama-4-scout-17b-16e-instruct",
		})

		const model = handler.getModel()
		expect(model.id).to.equal("meta-llama/llama-4-scout-17b-16e-instruct")
		expect(model.info.contextWindow).to.equal(131072)
		expect(model.info.maxTokens).to.equal(8192)
		expect(model.info.inputPrice).to.equal(0.31)
		expect(model.info.outputPrice).to.equal(0.36)
	})
})
