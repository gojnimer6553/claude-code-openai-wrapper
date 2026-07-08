from typing import List, Optional, Dict, Any
from src.models import Message
import json
import re


class MessageAdapter:
    """Converts between OpenAI message format and Claude Code prompts."""

    # Max prompt size in characters. Keep small to avoid slow CLI responses.
    # ~30K chars ≈ ~7.5K tokens — enough context without multi-minute waits.
    MAX_PROMPT_CHARS = 30_000

    @staticmethod
    def build_tools_system_prompt(tools: List[Dict[str, Any]]) -> str:
        """
        Build a system prompt section that describes external tool definitions
        to Claude in a structured way it can reason about.

        The format mirrors the XML tool-use notation Claude is trained on,
        so it reliably emits <tool_use> / ToolUseBlock responses.
        """
        if not tools:
            return ""

        lines = [
            "You have access to the following external tools. "
            "When you need to call a tool, respond ONLY with a JSON object wrapped in "
            "<tool_call> tags — no other text before or after the tag on that turn. "
            "After receiving a <tool_result>, you MUST use the data inside it to answer "
            "the user's question directly and completely — do not say the tool was called "
            "or describe what happened, just present the actual result to the user.\n",
            "<tools>",
        ]
        for t in tools:
            if t.get("type") != "function":
                continue
            func = t.get("function", {})
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
            lines.append(f"  <tool_description>")
            lines.append(f"    <tool_name>{name}</tool_name>")
            lines.append(f"    <description>{description}</description>")
            lines.append(f"    <parameters>{json.dumps(parameters, indent=2)}</parameters>")
            lines.append(f"  </tool_description>")
        lines.append("</tools>")
        lines.append(
            "\nTo call a tool respond with ONLY:\n"
            "<tool_call>\n"
            '{"name": "<tool_name>", "arguments": {<args>}}\n'
            "</tool_call>"
        )
        return "\n".join(lines)

    @staticmethod
    def messages_to_prompt(messages: List[Message]) -> tuple[str, Optional[str]]:
        """
        Convert OpenAI messages to Claude Code prompt format.
        Returns (prompt, system_prompt)

        Handles tool_calls on assistant messages and tool-result user messages
        so that multi-turn external tool use is preserved correctly.

        Truncates older conversation history if the prompt would exceed
        the OS command-line argument size limit (ARG_MAX).
        """
        system_prompt = None
        conversation_parts = []

        for message in messages:
            if message.role == "system":
                # Use the last system message as the system prompt
                system_prompt = message.content
            elif message.role == "user":
                content = message.content or ""
                # tool-result messages arrive as role="user" after normalize_content
                # already prefixed them with "[Tool result for call ...]".
                conversation_parts.append(f"Human: {content}")
            elif message.role == "assistant":
                # Serialize any tool_calls so Claude sees them in history
                if message.tool_calls:
                    tc_parts = []
                    for tc in message.tool_calls:
                        if isinstance(tc, dict):
                            func = tc.get("function", {})
                            tc_name = func.get("name", "")
                            tc_args = func.get("arguments", "{}")
                            tc_id = tc.get("id", "")
                        else:
                            # Pydantic model
                            func = getattr(tc, "function", None)
                            tc_name = getattr(func, "name", "") if func else ""
                            tc_args = getattr(func, "arguments", "{}") if func else "{}"
                            tc_id = getattr(tc, "id", "")
                        tc_parts.append(
                            f"<tool_call>\n"
                            f'{{"id": "{tc_id}", "name": "{tc_name}", '
                            f'"arguments": {tc_args}}}\n'
                            f"</tool_call>"
                        )
                    text_part = (message.content or "").strip()
                    tool_call_text = "\n".join(tc_parts)
                    combined = f"{text_part}\n{tool_call_text}".strip() if text_part else tool_call_text
                    conversation_parts.append(f"Assistant: {combined}")
                else:
                    conversation_parts.append(f"Assistant: {message.content}")

        # If the last message wasn't from the user, add a continuation prompt
        if messages and messages[-1].role != "user":
            conversation_parts.append("Human: Please continue.")

        # Truncate from the front (oldest messages) if prompt is too large
        prompt = "\n\n".join(conversation_parts)
        if len(prompt) > MessageAdapter.MAX_PROMPT_CHARS and len(conversation_parts) > 1:
            # Always keep the last message; drop oldest until it fits
            while len(conversation_parts) > 1:
                conversation_parts.pop(0)
                candidate = "[Earlier conversation truncated for length]\n\n" + "\n\n".join(
                    conversation_parts
                )
                if len(candidate) <= MessageAdapter.MAX_PROMPT_CHARS:
                    prompt = candidate
                    break
            else:
                # Even a single message is too long — hard-truncate it
                prompt = conversation_parts[0][: MessageAdapter.MAX_PROMPT_CHARS]

        return prompt, system_prompt

    @staticmethod
    def extract_tool_calls_from_text(
        text: str,
        external_tool_names: Optional[List[str]] = None,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Parse any <tool_call>...</tool_call> blocks that Claude emitted as plain
        text (i.e. not as native SDK ToolUseBlock objects).

        Returns:
            (remaining_text, list_of_tool_call_dicts)

        Each tool_call_dict has keys: id, name, arguments (JSON string).
        The matching <tool_call> tags are removed from the returned text.
        """
        import uuid as _uuid

        tool_calls: List[Dict[str, Any]] = []
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

        def _replace(m: re.Match) -> str:
            raw = m.group(1).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Not valid JSON — leave the text intact and don't emit a tool call
                return m.group(0)

            name = parsed.get("name", "")
            if not name:
                return m.group(0)

            # If a whitelist is provided, only relay calls for known external tools
            if external_tool_names and name not in external_tool_names:
                return m.group(0)

            tc_id = parsed.get("id") or f"call_{_uuid.uuid4().hex[:24]}"
            arguments = parsed.get("arguments", {})
            tc_args = json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)

            tool_calls.append({"id": tc_id, "name": name, "arguments": tc_args})
            return ""  # remove the tag from text

        remaining = pattern.sub(_replace, text)
        remaining = remaining.strip()
        return remaining, tool_calls

    @staticmethod
    def filter_passthrough_response(content: str) -> str:
        """
        Strip internal tool-protocol tags from Claude's final text response
        in passthrough mode so they never reach the end user.

        Removes:
        - <tool_call>...</tool_call>  (Claude emitting a new call mid-response)
        - <tool_result>...</tool_result>  (echoed back from context window)
        """
        if not content:
            return content
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        content = re.sub(r"<tool_result>.*?</tool_result>", "", content, flags=re.DOTALL)
        content = re.sub(r"\n{3,}", "\n\n", content)  # collapse excess blank lines
        return content.strip()

    @staticmethod
    def filter_content(content: str) -> str:
        """
        Filter content for unsupported features and tool usage.
        Remove thinking blocks, tool calls, and image references.
        """
        if not content:
            return content

        # Remove thinking blocks (common when tools are disabled but Claude tries to think)
        thinking_pattern = r"<thinking>.*?</thinking>"
        content = re.sub(thinking_pattern, "", content, flags=re.DOTALL)

        # Extract content from attempt_completion blocks (these contain the actual user response)
        attempt_completion_pattern = r"<attempt_completion>(.*?)</attempt_completion>"
        attempt_matches = re.findall(attempt_completion_pattern, content, flags=re.DOTALL)
        if attempt_matches:
            # Use the content from the attempt_completion block
            extracted_content = attempt_matches[0].strip()

            # If there's a <result> tag inside, extract from that
            result_pattern = r"<result>(.*?)</result>"
            result_matches = re.findall(result_pattern, extracted_content, flags=re.DOTALL)
            if result_matches:
                extracted_content = result_matches[0].strip()

            if extracted_content:
                content = extracted_content
        else:
            # Remove other tool usage blocks (when tools are disabled but Claude tries to use them)
            tool_patterns = [
                r"<read_file>.*?</read_file>",
                r"<write_file>.*?</write_file>",
                r"<bash>.*?</bash>",
                r"<search_files>.*?</search_files>",
                r"<str_replace_editor>.*?</str_replace_editor>",
                r"<args>.*?</args>",
                r"<ask_followup_question>.*?</ask_followup_question>",
                r"<attempt_completion>.*?</attempt_completion>",
                r"<question>.*?</question>",
                r"<follow_up>.*?</follow_up>",
                r"<suggest>.*?</suggest>",
            ]

            for pattern in tool_patterns:
                content = re.sub(pattern, "", content, flags=re.DOTALL)

        # Pattern to match image references or base64 data
        image_pattern = r"\[Image:.*?\]|data:image/.*?;base64,.*?(?=\s|$)"

        def replace_image(match):
            return "[Image: Content not supported by Claude Code]"

        content = re.sub(image_pattern, replace_image, content)

        # Clean up extra whitespace and newlines
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)  # Multiple newlines to double
        content = content.strip()

        # If content is now empty or only whitespace, provide a fallback
        if not content or content.isspace():
            return "I understand you're testing the system. How can I help you today?"

        # Avoid false-positive billing error detection by downstream platforms.
        # Some platforms rewrite responses containing "billing" + "credits"/"plans"
        # to a billing error message. Replace with safe synonyms.
        content = re.sub(r'\bbilling\b', 'invoicing', content, flags=re.IGNORECASE)
        content = re.sub(r'\bBilling\b', 'Invoicing', content)
        content = re.sub(r'\binsufficient credits\b', 'insufficient balance', content, flags=re.IGNORECASE)
        content = re.sub(r'\bpayment required\b', 'payment needed', content, flags=re.IGNORECASE)

        return content

    @staticmethod
    def format_claude_response(
        content: str, model: str, finish_reason: str = "stop"
    ) -> Dict[str, Any]:
        """Format Claude response for OpenAI compatibility."""
        return {
            "role": "assistant",
            "content": content,
            "finish_reason": finish_reason,
            "model": model,
        }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough estimation of token count.
        OpenAI's rule of thumb: ~4 characters per token for English text.
        """
        return len(text) // 4
