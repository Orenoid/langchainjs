import { describe, expect, it } from "vitest";
import { AIMessage, AIMessageChunk } from "../../ai.js";
import { ContentBlock } from "../../content/index.js";

describe("deepseekTranslator", () => {
  describe("Non-streaming Messages", () => {
    it("should translate reasoning_content to reasoning block", () => {
      const message = new AIMessage({
        content: "J'aime programmer",
        additional_kwargs: {
          reasoning_content: "The user wants a French translation...",
        },
        response_metadata: { model_provider: "deepseek" },
      });

      const expected: Array<ContentBlock.Standard> = [
        {
          type: "reasoning",
          reasoning: "The user wants a French translation...",
        },
        { type: "text", text: "J'aime programmer" },
      ];

      expect(message.contentBlocks).toEqual(expected);
    });

    it("should handle messages without reasoning_content", () => {
      const message = new AIMessage({
        content: "Hello world",
        response_metadata: { model_provider: "deepseek" },
      });

      const expected: Array<ContentBlock.Standard> = [
        { type: "text", text: "Hello world" },
      ];

      expect(message.contentBlocks).toEqual(expected);
    });

    it("should handle messages with reasoning and tool calls", () => {
      const message = new AIMessage({
        content: "I'll check the weather",
        additional_kwargs: {
          reasoning_content: "User wants weather information...",
        },
        tool_calls: [
          {
            id: "call_123",
            name: "get_weather",
            args: { location: "SF" },
          },
        ],
        response_metadata: { model_provider: "deepseek" },
      });

      const expected: Array<ContentBlock.Standard> = [
        {
          type: "reasoning",
          reasoning: "User wants weather information...",
        },
        { type: "text", text: "I'll check the weather" },
        {
          type: "tool_call",
          id: "call_123",
          name: "get_weather",
          args: { location: "SF" },
        },
      ];

      expect(message.contentBlocks).toEqual(expected);
    });

    it("should handle empty reasoning_content", () => {
      const message = new AIMessage({
        content: "Answer",
        additional_kwargs: {
          reasoning_content: "",
        },
        response_metadata: { model_provider: "deepseek" },
      });

      expect(message.contentBlocks).toEqual([
        { type: "text", text: "Answer" },
      ]);
    });

    it("should handle multiple tool calls", () => {
      const message = new AIMessage({
        content: "I'll check both",
        additional_kwargs: {
          reasoning_content: "Need to check weather and population...",
        },
        tool_calls: [
          {
            id: "call_1",
            name: "get_weather",
            args: { location: "NYC" },
          },
          {
            id: "call_2",
            name: "get_population",
            args: { location: "NYC" },
          },
        ],
        response_metadata: { model_provider: "deepseek" },
      });

      const expected: Array<ContentBlock.Standard> = [
        {
          type: "reasoning",
          reasoning: "Need to check weather and population...",
        },
        { type: "text", text: "I'll check both" },
        {
          type: "tool_call",
          id: "call_1",
          name: "get_weather",
          args: { location: "NYC" },
        },
        {
          type: "tool_call",
          id: "call_2",
          name: "get_population",
          args: { location: "NYC" },
        },
      ];

      expect(message.contentBlocks).toEqual(expected);
    });
  });

  describe("Streaming Message Chunks", () => {
    it("should translate reasoning in chunks", () => {
      const chunk1 = new AIMessageChunk({
        content: "",
        additional_kwargs: {
          reasoning_content: "Thinking...",
        },
        response_metadata: { model_provider: "deepseek" },
      });

      expect(chunk1.contentBlocks).toEqual([
        { type: "reasoning", reasoning: "Thinking..." },
        { type: "text", text: "" },
      ]);
    });

    it("should accumulate reasoning across chunks", () => {
      const chunk1 = new AIMessageChunk({
        content: "",
        additional_kwargs: { reasoning_content: "Part 1" },
        response_metadata: { model_provider: "deepseek" },
      });

      const chunk2 = new AIMessageChunk({
        content: "Answer",
        additional_kwargs: { reasoning_content: " Part 2" },
        response_metadata: { model_provider: "deepseek" },
      });

      const final = chunk1.concat(chunk2);

      expect(final.contentBlocks).toEqual([
        { type: "reasoning", reasoning: "Part 1 Part 2" },
        { type: "text", text: "Answer" },
      ]);
    });

    it("should handle chunks without reasoning", () => {
      const chunk = new AIMessageChunk({
        content: "Hello",
        response_metadata: { model_provider: "deepseek" },
      });

      expect(chunk.contentBlocks).toEqual([{ type: "text", text: "Hello" }]);
    });
  });

  describe("Backward Compatibility", () => {
    it("should maintain reasoning_content in additional_kwargs", () => {
      const message = new AIMessage({
        content: "Answer",
        additional_kwargs: {
          reasoning_content: "Thinking...",
        },
        response_metadata: { model_provider: "deepseek" },
      });

      // Verify additional_kwargs still works
      expect(message.additional_kwargs.reasoning_content).toBe("Thinking...");

      // Verify contentBlocks also works
      expect(message.contentBlocks[0].type).toBe("reasoning");
      if (message.contentBlocks[0].type === "reasoning") {
        expect(message.contentBlocks[0].reasoning).toBe("Thinking...");
      }
    });

    it("should not break when additional_kwargs is missing", () => {
      const message = new AIMessage({
        content: "Answer",
        response_metadata: { model_provider: "deepseek" },
      });

      expect(message.contentBlocks).toEqual([
        { type: "text", text: "Answer" },
      ]);
    });

    it("should not break when reasoning_content is not a string", () => {
      const message = new AIMessage({
        content: "Answer",
        additional_kwargs: {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          reasoning_content: null as any,
        },
        response_metadata: { model_provider: "deepseek" },
      });

      expect(message.contentBlocks).toEqual([
        { type: "text", text: "Answer" },
      ]);
    });
  });

  describe("Edge Cases", () => {
    it("should handle empty content with reasoning", () => {
      const message = new AIMessage({
        content: "",
        additional_kwargs: {
          reasoning_content: "Just thinking, no answer yet...",
        },
        response_metadata: { model_provider: "deepseek" },
      });

      expect(message.contentBlocks).toEqual([
        { type: "reasoning", reasoning: "Just thinking, no answer yet..." },
        { type: "text", text: "" },
      ]);
    });

    it("should handle array content", () => {
      const message = new AIMessage({
        content: [
          { type: "text", text: "First part" },
          { type: "text", text: "Second part" },
        ],
        additional_kwargs: {
          reasoning_content: "Analyzing...",
        },
        response_metadata: { model_provider: "deepseek" },
      });

      expect(message.contentBlocks).toEqual([
        { type: "reasoning", reasoning: "Analyzing..." },
        { type: "text", text: "First part" },
        { type: "text", text: "Second part" },
      ]);
    });
  });
});
