import type { ContentBlock } from "../content/index.js";
import type { AIMessageChunk, AIMessage } from "../ai.js";
import type { StandardContentBlockTranslator } from "./index.js";
import { _isObject, _isString } from "./utils.js";

/**
 * Converts a DeepSeek message to v1 standard content blocks.
 *
 * This function processes AI messages from DeepSeek's API format and converts
 * them to the standardized v1 content block format. It handles reasoning content
 * from the deepseek-reasoner model, as well as standard text and tool calls.
 *
 * @param message - The AI message containing DeepSeek-formatted content
 * @returns Array of content blocks in v1 standard format
 *
 * @example
 * ```typescript
 * const message = new AIMessage({
 *   content: "J'aime programmer",
 *   additional_kwargs: {
 *     reasoning_content: "The user asked for a French translation..."
 *   },
 *   response_metadata: { model_provider: "deepseek" }
 * });
 *
 * const standardBlocks = convertToV1FromDeepSeek(message);
 * // Returns:
 * // [
 * //   { type: "reasoning", reasoning: "The user asked for a French translation..." },
 * //   { type: "text", text: "J'aime programmer" }
 * // ]
 * ```
 */
export function convertToV1FromDeepSeek(
  message: AIMessage | AIMessageChunk
): Array<ContentBlock.Standard> {
  const blocks: Array<ContentBlock.Standard> = [];

  // Extract reasoning content from additional_kwargs if present
  // Only deepseek-reasoner model provides this field
  if (
    _isObject(message.additional_kwargs) &&
    _isString(message.additional_kwargs.reasoning_content) &&
    message.additional_kwargs.reasoning_content.length > 0
  ) {
    blocks.push({
      type: "reasoning",
      reasoning: message.additional_kwargs.reasoning_content,
    });
  }

  // Add text content
  if (typeof message.content === "string") {
    blocks.push({
      type: "text",
      text: message.content,
    });
  } else if (Array.isArray(message.content)) {
    // Handle structured content (already in content block format)
    for (const block of message.content) {
      blocks.push(block as ContentBlock.Standard);
    }
  }

  // Add tool calls
  for (const toolCall of message.tool_calls ?? []) {
    blocks.push({
      type: "tool_call",
      id: toolCall.id,
      name: toolCall.name,
      args: toolCall.args,
    });
  }

  return blocks;
}

export const ChatDeepSeekTranslator: StandardContentBlockTranslator = {
  translateContent: convertToV1FromDeepSeek,
  translateContentChunk: convertToV1FromDeepSeek,
};
