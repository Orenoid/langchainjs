/* eslint-disable @typescript-eslint/no-explicit-any */
import { test, expect } from "vitest";
import { ChatDeepSeek } from "../chat_models.js";

test("Can send deepseek-reasoner requests", async () => {
  const llm = new ChatDeepSeek({
    model: "deepseek-reasoner",
  });
  const input = `Translate "I love programming" into French.`;
  // Models also accept a list of chat messages or a formatted prompt
  const result = await llm.invoke(input);
  expect(
    (result.additional_kwargs.reasoning_content as any).length
  ).toBeGreaterThan(10);
});

test("Reasoning content available via contentBlocks", async () => {
  const llm = new ChatDeepSeek({
    model: "deepseek-reasoner",
  });

  const result = await llm.invoke(
    "Translate 'I love programming' into French."
  );

  // Check contentBlocks for standard access
  const reasoningBlocks = result.contentBlocks.filter(
    (block) => block.type === "reasoning"
  );
  expect(reasoningBlocks.length).toBe(1);
  if (reasoningBlocks[0].type === "reasoning") {
    expect(reasoningBlocks[0].reasoning.length).toBeGreaterThan(10);
  }

  // Check backward compatibility
  expect(result.additional_kwargs.reasoning_content).toBeTruthy();
  expect(typeof result.additional_kwargs.reasoning_content).toBe("string");

  // Verify they match
  if (reasoningBlocks[0].type === "reasoning") {
    expect(reasoningBlocks[0].reasoning).toBe(
      result.additional_kwargs.reasoning_content
    );
  }

  // Verify model_provider is set
  expect(result.response_metadata?.model_provider).toBe("deepseek");
});

test("Streaming chunks include reasoning blocks", async () => {
  const llm = new ChatDeepSeek({
    model: "deepseek-reasoner",
  });

  const chunks: any[] = [];
  for await (const chunk of await llm.stream("What is 2+2?")) {
    chunks.push(chunk);
  }

  // At least one chunk should have reasoning
  const chunksWithReasoning = chunks.filter((chunk) =>
    chunk.contentBlocks.some((b: any) => b.type === "reasoning")
  );
  expect(chunksWithReasoning.length).toBeGreaterThan(0);

  // Verify backward compatibility
  const chunksWithReasoningKwargs = chunks.filter(
    (chunk) => chunk.additional_kwargs.reasoning_content
  );
  expect(chunksWithReasoningKwargs.length).toBeGreaterThan(0);

  // Verify model_provider is set in chunks
  const chunksWithProvider = chunks.filter(
    (chunk) => chunk.response_metadata?.model_provider === "deepseek"
  );
  expect(chunksWithProvider.length).toBeGreaterThan(0);
});

test("Non-reasoning model (deepseek-chat) should not have reasoning blocks", async () => {
  const llm = new ChatDeepSeek({
    model: "deepseek-chat",
  });

  const result = await llm.invoke("Hello!");

  const reasoningBlocks = result.contentBlocks.filter(
    (block) => block.type === "reasoning"
  );
  expect(reasoningBlocks.length).toBe(0);
  expect(result.additional_kwargs.reasoning_content).toBeUndefined();

  // Should still have model_provider set
  expect(result.response_metadata?.model_provider).toBe("deepseek");
});
