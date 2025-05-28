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
						id: "whisper-large-v3",
						object: "model",
						owned_by: "openai",
					},
					{
						id: "llama-guard-3-8b",
						object: "model",
						owned_by: "meta",
					},
					{
						id: "tts-1",
						object: "model",
						owned_by: "openai",
					},
					{
						id: "text-embedding-ada-002",
						object: "model",
						owned_by: "openai",
					},
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

		// Should not include filtered models
		expect(result.models).to.not.have.property("whisper-large-v3")
		expect(result.models).to.not.have.property("llama-guard-3-8b")
		expect(result.models).to.not.have.property("tts-1")
		expect(result.models).to.not.have.property("text-embedding-ada-002")
	})

	it("should detect image support for vision models", async () => {
		const mockApiResponse = {
			data: {
				data: [
					{
						id: "llama-4-maverick-vision",
						object: "model",
						owned_by: "meta",
						context_window: 131072,
						max_completion_tokens: 8192,
					},
					{
						id: "gpt-4-vision-preview",
						object: "model",
						owned_by: "openai",
						context_window: 128000,
						max_completion_tokens: 4096,
					},
					{
						id: "regular-text-model",
						object: "model",
						owned_by: "test",
						context_window: 8192,
						max_completion_tokens: 4096,
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

		// Vision models should support images
		expect(result.models["llama-4-maverick-vision"].supportsImages).to.be.true
		expect(result.models["gpt-4-vision-preview"].supportsImages).to.be.true

		// Regular model should not support images
		expect(result.models["regular-text-model"].supportsImages).to.be.false
	})

	it("should handle API errors gracefully and fall back to static models", async () => {
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

		// Should include known static models
		expect(result.models).to.have.property("llama-3.3-70b-versatile")
		expect(result.models).to.have.property("llama-3.1-8b-instant")

		// Verify error was logged
		expect(consoleErrorStub.calledWith("Error fetching Groq models:")).to.be.true
	})

	it("should estimate pricing for unknown model types", async () => {
		const mockApiResponse = {
			data: {
				data: [
					{
						id: "new-70b-model",
						object: "model",
						owned_by: "test",
						context_window: 32768,
						max_completion_tokens: 8192,
					},
					{
						id: "new-8b-model",
						object: "model",
						owned_by: "test",
						context_window: 16384,
						max_completion_tokens: 4096,
					},
					{
						id: "unknown-size-model",
						object: "model",
						owned_by: "test",
						context_window: 8192,
						max_completion_tokens: 2048,
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

		// 70B model should have higher pricing
		expect(result.models["new-70b-model"].inputPrice).to.equal(0.59)
		expect(result.models["new-70b-model"].outputPrice).to.equal(0.79)

		// 8B model should have lower pricing
		expect(result.models["new-8b-model"].inputPrice).to.equal(0.05)
		expect(result.models["new-8b-model"].outputPrice).to.equal(0.08)

		// Unknown size should use default pricing
		expect(result.models["unknown-size-model"].inputPrice).to.equal(0.2)
		expect(result.models["unknown-size-model"].outputPrice).to.equal(0.2)
	})

	it("should handle missing API key", async () => {
		getAllExtensionStateStub.resolves({
			apiConfiguration: {
				// No groqApiKey provided
			},
		})

		const result = await refreshGroqModels(controller, EmptyRequest.create({}))

		// Should fall back to static models when no API key
		expect(Object.keys(result.models).length).to.be.greaterThan(0)
		expect(result.models).to.have.property("llama-3.3-70b-versatile")

		// Should log error about missing API key
		expect(consoleErrorStub.calledWith("Error fetching Groq models:")).to.be.true
	})

	it("should filter out inactive models", async () => {
		const mockApiResponse = {
			data: {
				data: [
					{
						id: "active-model",
						object: "model",
						owned_by: "test",
						active: true,
						context_window: 8192,
						max_completion_tokens: 4096,
					},
					{
						id: "inactive-model",
						object: "model",
						owned_by: "test",
						active: false,
						context_window: 8192,
						max_completion_tokens: 4096,
					},
					{
						id: "model-without-active-property",
						object: "model",
						owned_by: "test",
						// No active property - should be included
						context_window: 8192,
						max_completion_tokens: 4096,
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

		// Should include active model
		expect(result.models).to.have.property("active-model")

		// Should include model without active property (defaults to active)
		expect(result.models).to.have.property("model-without-active-property")

		// Should NOT include inactive model
		expect(result.models).to.not.have.property("inactive-model")
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

	it("should fetch models from Groq API", async function () {
		if (!groqApiAvailable || !apiKey) {
			this.skip()
		}
		this.timeout(10000)

		const response = await axios.get("https://api.groq.com/openai/v1/models", {
			headers: {
				Authorization: `Bearer ${apiKey}`,
				"Content-Type": "application/json",
				"User-Agent": "Cline-VSCode-Extension-Test",
			},
			timeout: 5000,
		})

		expect(response.status).to.equal(200)
		expect(response.data).to.have.property("data")
		expect(response.data.data).to.be.an("array")
		expect(response.data.data.length).to.be.greaterThan(0)

		// Check that we have the expected models
		const modelIds = response.data.data.map((model: any) => model.id)

		// Verify Llama 4 models have proper prefixes
		expect(modelIds).to.include("meta-llama/llama-4-scout-17b-16e-instruct")
		expect(modelIds).to.include("meta-llama/llama-4-maverick-17b-128e-instruct")
		expect(modelIds).to.include("meta-llama/llama-guard-4-12b")

		// Verify other expected models
		expect(modelIds).to.include("llama-3.1-8b-instant")
		expect(modelIds).to.include("llama-3.3-70b-versatile")
		expect(modelIds).to.include("deepseek-r1-distill-llama-70b")
	})

	it("should validate model specifications match API response", async function () {
		if (!groqApiAvailable || !apiKey) {
			this.skip()
		}
		this.timeout(10000)

		const response = await axios.get("https://api.groq.com/openai/v1/models", {
			headers: {
				Authorization: `Bearer ${apiKey}`,
				"Content-Type": "application/json",
				"User-Agent": "Cline-VSCode-Extension-Test",
			},
			timeout: 5000,
		})

		const apiModels = response.data.data

		// Test specific models that we fixed
		const testCases = [
			{
				modelId: "meta-llama/llama-4-scout-17b-16e-instruct",
				expectedContextWindow: 131072,
				expectedMaxTokens: 8192,
			},
			{
				modelId: "meta-llama/llama-4-maverick-17b-128e-instruct",
				expectedContextWindow: 131072,
				expectedMaxTokens: 8192,
			},
			{
				modelId: "meta-llama/llama-guard-4-12b",
				expectedContextWindow: 131072,
				expectedMaxTokens: 1024,
			},
			{
				modelId: "llama-3.1-8b-instant",
				expectedContextWindow: 131072,
				expectedMaxTokens: 131072,
			},
		]

		for (const testCase of testCases) {
			const apiModel = apiModels.find((model: any) => model.id === testCase.modelId)
			expect(apiModel, `Model ${testCase.modelId} should exist in API response`).to.exist

			expect(apiModel.context_window).to.equal(testCase.expectedContextWindow, `Context window for ${testCase.modelId}`)
			expect(apiModel.max_completion_tokens).to.equal(testCase.expectedMaxTokens, `Max tokens for ${testCase.modelId}`)

			// Verify our static model info matches
			const staticModel = groqModels[testCase.modelId as keyof typeof groqModels]
			if (staticModel) {
				expect(staticModel.contextWindow).to.equal(
					testCase.expectedContextWindow,
					`Static context window for ${testCase.modelId}`,
				)
				expect(staticModel.maxTokens).to.equal(testCase.expectedMaxTokens, `Static max tokens for ${testCase.modelId}`)
			}
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

	it("should handle API errors gracefully", async function () {
		if (!groqApiAvailable) {
			this.skip()
		}
		this.timeout(10000)

		// Test with invalid API key
		try {
			await axios.get("https://api.groq.com/openai/v1/models", {
				headers: {
					Authorization: "Bearer invalid-key",
					"Content-Type": "application/json",
					"User-Agent": "Cline-VSCode-Extension-Test",
				},
				timeout: 5000,
			})
			// Should not reach here
			expect.fail("Expected API call to fail with invalid key")
		} catch (error) {
			if (axios.isAxiosError(error)) {
				expect(error.response?.status).to.equal(401)
			} else {
				throw error
			}
		}
	})

	it("should validate all static models exist in API", async function () {
		if (!groqApiAvailable || !apiKey) {
			this.skip()
		}
		this.timeout(10000)

		const response = await axios.get("https://api.groq.com/openai/v1/models", {
			headers: {
				Authorization: `Bearer ${apiKey}`,
				"Content-Type": "application/json",
				"User-Agent": "Cline-VSCode-Extension-Test",
			},
			timeout: 5000,
		})

		const apiModelIds = response.data.data.map((model: any) => model.id)
		const staticModelIds = Object.keys(groqModels)

		// Check that all our static models exist in the API
		// (Note: API might have more models than our static list)
		for (const staticModelId of staticModelIds) {
			// Skip deprecated models that might not be in API anymore
			if (staticModelId.includes("llama3-") && staticModelId.includes("-8192")) {
				continue // These are legacy model IDs
			}

			expect(apiModelIds).to.include(staticModelId, `Static model ${staticModelId} should exist in API response`)
		}
	})

	it("should validate vision model capabilities", async function () {
		if (!groqApiAvailable || !apiKey) {
			this.skip()
		}
		this.timeout(5000)

		// Test Llama 4 Maverick which should support vision
		const handler = new GroqHandler({
			groqApiKey: apiKey,
			apiModelId: "meta-llama/llama-4-maverick-17b-128e-instruct",
		})

		const model = handler.getModel()
		expect(model.id).to.equal("meta-llama/llama-4-maverick-17b-128e-instruct")
		expect(model.info.supportsImages).to.equal(true, "Llama 4 Maverick should support images")
		expect(model.info.contextWindow).to.equal(131072)
		expect(model.info.inputPrice).to.equal(0.2)
		expect(model.info.outputPrice).to.equal(0.6)
	})
})
