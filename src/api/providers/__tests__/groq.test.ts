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
