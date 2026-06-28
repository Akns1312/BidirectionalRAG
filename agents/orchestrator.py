from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config.settings import settings
from agents.tools import retrieve_tool, store_tool, summarize_tool
from agents.prompts import ORCHESTRATOR_SYSTEM
from ingestion.prompt_ingestor import PromptEnvelope
from ingestion.session_manager import add_turn


# ── LLM instance ──────────────────────────────────────────────────────────────
# ChatOllama supports bind_tools() which is required by create_react_agent.
# OllamaLLM does not support tool calling — ChatOllama is the correct class.
_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.LLM_MODEL,
    temperature=0.2,
)

# ── Tools list ────────────────────────────────────────────────────────────────
_tools = [retrieve_tool, store_tool, summarize_tool]

# ── ReAct agent ───────────────────────────────────────────────────────────────
# prompt= is the correct parameter name in langgraph 1.2.4
# (state_modifier= and messages_modifier= are removed in this version)
_agent = create_react_agent(
    model=_llm,
    tools=_tools,
    prompt=ORCHESTRATOR_SYSTEM,
)


# ── Main run function ─────────────────────────────────────────────────────────

def run(envelope: PromptEnvelope) -> str:
    """
    Run the agent on a PromptEnvelope and return the final response string.

    Flow:
        1. Convert history to LangChain message objects
        2. Inject session_id into system message so agent passes it to store_tool
        3. Append current user message
        4. Invoke the agent loop
        5. Extract final response from agent output
        6. Record both user message and response in session history
        7. Return response string
    """

    messages = []

    # ── Step 1: Convert history to message objects ────────────────────────────
    for turn in envelope.history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    # ── Step 2: Inject session_id into system message ─────────────────────────
    # The agent needs session_id to pass it to store_tool as an argument.
    # We append it to the system prompt so the agent always knows the current
    # session and can include it when calling store_tool(text, session_id).
    system_with_session = (
        ORCHESTRATOR_SYSTEM
        + f"\n\nCurrent session_id: {envelope.session_id}"
        + "\nAlways pass this session_id when calling store_tool."
    )
    messages.insert(0, SystemMessage(content=system_with_session))

    # ── Step 3: Append current user message ───────────────────────────────────
    messages.append(HumanMessage(content=envelope.text))

    # ── Step 4: Invoke the agent ──────────────────────────────────────────────
    # No config needed — session_id is passed via the system message instead.
    try:
        result = _agent.invoke({"messages": messages})
    except Exception as e:
        error_msg = f"Agent encountered an error: {e}"
        print(f"[orchestrator] ERROR: {e}")
        return error_msg

    # ── Step 5: Extract final response ────────────────────────────────────────
    # LangGraph returns all messages including tool calls and observations.
    # The last message is always the final AI response to the user.
    all_messages = result.get("messages", [])

    if not all_messages:
        return "I was unable to generate a response. Please try again."

    final_message = all_messages[-1]

    if hasattr(final_message, "content"):
        response = final_message.content.strip()
    else:
        response = str(final_message).strip()

    if not response:
        return "I received an empty response. Please try again."

    # ── Step 6: Record in session history ─────────────────────────────────────
    # Record AFTER the agent finishes — never before.
    # If the agent crashed mid-run, we don't want a dangling user turn
    # in history with no corresponding assistant response.
    add_turn(envelope.session_id, "user", envelope.text)
    add_turn(envelope.session_id, "assistant", response)

    # ── Step 7: Return ────────────────────────────────────────────────────────
    return response


# ── Debug helper ──────────────────────────────────────────────────────────────

def run_with_trace(envelope: PromptEnvelope) -> dict:
    """
    Same as run() but returns a full trace including all tool calls made.
    Use this during development to understand what the agent did.

    Returns:
        {
            "response"  : "final answer string",
            "tool_calls": ["retrieve_tool(query=...)", "store_tool(text=...)"],
            "turn_count": 4,
        }
    """

    messages = []

    for turn in envelope.history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    system_with_session = (
        ORCHESTRATOR_SYSTEM
        + f"\n\nCurrent session_id: {envelope.session_id}"
        + "\nAlways pass this session_id when calling store_tool."
    )
    messages.insert(0, SystemMessage(content=system_with_session))
    messages.append(HumanMessage(content=envelope.text))

    try:
        result = _agent.invoke({"messages": messages})
    except Exception as e:
        return {"response": f"Error: {e}", "tool_calls": [], "turn_count": 0}

    all_messages = result.get("messages", [])

    # ── Extract tool calls from intermediate messages ─────────────────────────
    tool_calls = []
    for msg in all_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "unknown_tool")
                args = tc.get("args", {})
                args_str = ", ".join(
                    f"{k}={str(v)[:50]}{'...' if len(str(v)) > 50 else ''}"
                    for k, v in args.items()
                )
                tool_calls.append(f"{name}({args_str})")

    final_response = all_messages[-1].content.strip() if all_messages else ""

    add_turn(envelope.session_id, "user", envelope.text)
    add_turn(envelope.session_id, "assistant", final_response)

    return {
        "response"  : final_response,
        "tool_calls": tool_calls,
        "turn_count": envelope.turn,
    }