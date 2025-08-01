import json
import os
import uuid
import platform
import asyncio
from langgraph.graph.message import AnyMessage, add_messages
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from typing import Annotated, Any, Optional, Literal, Callable, List, Dict
from prompts import primary_prompt, plan_prompt, sim_prompt, num_prompt, policy_prompt, faq_prompt
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableConfig, Runnable
from langgraph.constants import END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from tools import plan_tools, sim_tools, num_tools, policy_tools, faq_tools
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
from gtts import gTTS
import tempfile

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)


# Define state with explicit message history
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    dialog_state: Literal[
        "Primary_Assistant", "Plan_Assistant", "Sim_Assistant", "Num_Assistant", "Policy_Assistant", "FAQ_Assistant", "Store_Assistant", None]


# Tool error handling
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# Create tool node with fallback
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Event printing utility with conversation history context
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: {current_state}")
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


# Dialog stack update
def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


# Assistant class with conversational context
class Assistant:
    def __init__(self, runnable: Runnable, max_history_length: int = 10):
        self.runnable = runnable
        self.max_history_length = max_history_length

    def __call__(self, state: State, config: RunnableConfig):
        # Limit the message history for LLM context
        messages = state["messages"][-self.max_history_length:] if len(state["messages"]) > self.max_history_length else \
        state["messages"]
        state = {**state, "messages": messages}

        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                    not result.content
                    or (isinstance(result.content, list)
                        and not result.content[0].get("text"))
            ):
                messages = state["messages"] + [HumanMessage(content="Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# Entry node for specialized assistants
def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        # Summarize prior conversation for context
        prior_conversation = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"][-5:]])
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Previous conversation summary:\n{prior_conversation}\n"
                            f"The user's intent is unsatisfied. Use the provided tools to complete the tasks. "
                            f"You are tasked with either fetching data, running browser use, or validating output. "
                            f"If the user changes their mind or needs other tasks, call CompleteOrEscalate. "
                            f"Do not mention your role - act as the proxy.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


# CompleteOrEscalate Tool
class CompleteOrEscalate(BaseModel):
    cancel: bool = True
    reason: str


class ToPlanAssistant(BaseModel):
    request: str = Field(description="Request for finding best plans.")
    expected_output: str = Field(description="Expected output for the task.")


class ToSimAssistant(BaseModel):
    request: str = Field(description="Request for sim related help.")
    expected_output: str = Field(description="Expected output for the task.")


class ToNumAssistant(BaseModel):
    request: str = Field(description="Request for finding Airtel related numbers from PDF or internet.")
    expected_output: str = Field(description="Expected output for the task.")


class ToPolicyAssistant(BaseModel):
    request: str = Field(description="Request for policy related information.")
    expected_output: str = Field(description="Expected output for the task.")


class ToFAQAssistant(BaseModel):
    request: str = Field(description="Request for frequently asked questions or their answers.")
    expected_output: str = Field(description="Expected output for the task.")


# Initialize Runnables with conversational context
primary_runnable = primary_prompt | llm.bind_tools([
    ToPlanAssistant,
    ToSimAssistant,
    ToNumAssistant,
    ToPolicyAssistant,
    ToFAQAssistant,
    CompleteOrEscalate
])

plan_runnable = plan_prompt | llm.bind_tools(plan_tools + [CompleteOrEscalate])
sim_runnable = sim_prompt | llm.bind_tools(sim_tools + [CompleteOrEscalate])
num_runnable = num_prompt | llm.bind_tools(num_tools + [CompleteOrEscalate])
policy_runnable = policy_prompt | llm.bind_tools(policy_tools + [CompleteOrEscalate])
faq_runnable = faq_prompt | llm.bind_tools(faq_tools + [CompleteOrEscalate])

# Build the graph
builder = StateGraph(State)

# Define nodes with conversational Assistants
builder.add_node("Primary_Assistant", Assistant(primary_runnable, max_history_length=10))
builder.add_node("Plan_Assistant", Assistant(plan_runnable, max_history_length=10))
builder.add_node("plan_tools", create_tool_node_with_fallback(plan_tools))
builder.add_node("Sim_Assistant", Assistant(sim_runnable, max_history_length=10))
builder.add_node("sim_tools", create_tool_node_with_fallback(sim_tools))
builder.add_node("Num_Assistant", Assistant(num_runnable, max_history_length=10))
builder.add_node("num_tools", create_tool_node_with_fallback(num_tools))
builder.add_node("Policy_Assistant", Assistant(policy_runnable, max_history_length=10))
builder.add_node("policy_tools", create_tool_node_with_fallback(policy_tools))
builder.add_node("FAQ_Assistant", Assistant(faq_runnable, max_history_length=10))
builder.add_node("faq_tools", create_tool_node_with_fallback(faq_tools))


# Routing functions
def route_plan(state: State):
    route = tools_condition(state)
    print(f"Plan Agent Routing To: {route}")
    last_message = state["messages"][-1]
    did_cancel = False
    if hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel or route == END:
        return "Primary_Assistant"
    return "plan_tools"


def route_policy(state: State):
    route = tools_condition(state)
    print(f"Policy Agent Routing To: {route}")
    last_message = state["messages"][-1]
    did_cancel = False
    if hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel or route == END:
        return "Primary_Assistant"
    return "policy_tools"


def route_faq(state: State):
    route = tools_condition(state)
    print(f"FAQ Agent Routing To: {route}")
    last_message = state["messages"][-1]
    did_cancel = False
    if hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel or route == END:
        return "Primary_Assistant"
    return "faq_tools"


def route_sim(state: State):
    route = tools_condition(state)
    print(f"Sim Agent Routing To: {route}")
    last_message = state["messages"][-1]
    did_cancel = False
    if hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel or route == END:
        return "Primary_Assistant"
    return "sim_tools"


def route_num(state: State):
    route = tools_condition(state)
    print(f"Num Agent Routing To: {route}")
    last_message = state["messages"][-1]
    did_cancel = False
    if hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel or route == END:
        return "Primary_Assistant"
    return "num_tools"

def route_primary_assistant(state: State):
    last_message = state["messages"][-1]
    user_input = last_message.content if isinstance(last_message.content, str) else ""

    if user_input.lower() in ["quit", "q", "end"]:
        print("User requested to end the flow.")
        return END

    route = tools_condition(state)
    print(f"Primary Agent Routing To: {route}")
    if route == END:
        return END
    tool_calls = last_message.tool_calls
    if tool_calls:
        call_name = tool_calls[0]["name"]
        if call_name == ToPlanAssistant.__name__:
            return "Plan_Assistant"
        elif call_name == ToSimAssistant.__name__:
            return "Sim_Assistant"
        elif call_name == ToNumAssistant.__name__:
            return "Num_Assistant"
        elif call_name == ToPolicyAssistant.__name__:
            return "Policy_Assistant"
        elif call_name == ToFAQAssistant.__name__:
            return "FAQ_Assistant"
    return END


# Whisper and TTS functions
whisper_model = whisper.load_model("base", device="cpu")


def listen_to_user_whisper(duration=5, language='en') -> str:
    print("🎤 Listening... Speak now.")
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    scipy.io.wavfile.write(temp_wav.name, fs, recording)
    try:
        result = whisper_model.transcribe(temp_wav.name, language=language)
        text = result['text'].strip()
        print(f"🗣️ You said: {text}")
        return text
    except Exception as e:
        print(f"❌ Whisper STT Error: {e}")
        return ""


def speak_output_gtts(text: str):
    print(f"\n🔊 Playing response...")
    try:
        tts = gTTS(text=text, lang='en-IN', slow=False)
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_mp3.name)
        os.system(f"mpg123 {temp_mp3.name}")
    except Exception as e:
        print(f"❌ TTS Error: {e}")


# Define edges
builder.add_edge(START, "Primary_Assistant")
builder.add_conditional_edges("Primary_Assistant", route_primary_assistant, {
    "Plan_Assistant": "Plan_Assistant",
    "Sim_Assistant": "Sim_Assistant",
    "Num_Assistant": "Num_Assistant",
    "Policy_Assistant": "Policy_Assistant",
    "FAQ_Assistant": "FAQ_Assistant",
    END: END
})
builder.add_conditional_edges("Plan_Assistant", route_plan, {
    "plan_tools": "plan_tools",
    "Primary_Assistant": "Primary_Assistant"
})
builder.add_edge("plan_tools", "Plan_Assistant")
builder.add_conditional_edges("Sim_Assistant", route_sim, {
    "sim_tools": "sim_tools",
    "Primary_Assistant": "Primary_Assistant"
})
builder.add_edge("sim_tools", "Sim_Assistant")
builder.add_conditional_edges("Num_Assistant", route_num, {
    "num_tools": "num_tools",
    "Primary_Assistant": "Primary_Assistant"
})
builder.add_edge("num_tools", "Num_Assistant")
builder.add_conditional_edges("Policy_Assistant", route_policy, {
    "policy_tools": "policy_tools",
    "Primary_Assistant": "Primary_Assistant"
})
builder.add_edge("policy_tools", "Policy_Assistant")
builder.add_conditional_edges("FAQ_Assistant", route_faq, {
    "faq_tools": "faq_tools",
    "Primary_Assistant": "Primary_Assistant"
})
builder.add_edge("faq_tools", "FAQ_Assistant")

# Compile the graph with memory
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Generate and save workflow visualization
try:
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    image_path = "workflow_graph.png"
    with open(image_path, "wb") as f:
        f.write(graph_image)
    print(f"Graph saved as {image_path}")

    system = platform.system()
    if system == "Darwin":
        os.system(f"open {image_path}")
    elif system == "Windows":
        os.startfile(image_path)
    else:
        os.system(f"xdg-open {image_path}")
except Exception as e:
    print(f"Failed to generate or open graph visualization: {e}")


def run_chatbot(user_input: str, thread_id: str = None, language: str = "en") -> str:
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    _printed = set()

    print(f"\n🤖 Processing input: {user_input} (Thread ID: {thread_id}, Language: {language})")

    # Validate language
    if language not in ["hi", "en"]:
        print(f"⚠️ Invalid language code: {language}. Defaulting to English (en).")
        language = "en"

    # Load previous conversation if thread_id exists
    try:
        checkpoint = memory.get(config)
        if checkpoint and "messages" in checkpoint:
            print("\n📜 Previous conversation loaded (last 3 messages):")
            for msg in checkpoint["messages"][-3:]:
                print(f"{msg.type}: {msg.content}")
    except Exception as e:
        print(f"⚠️ Could not load previous conversation: {e}")

    # Process the input through LangGraph
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values"
    )

    response_parts = []
    for event in events:
        _print_event(event, _printed)
        if "messages" in event:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, (AIMessage, ToolMessage)) and hasattr(last_msg, "content") and isinstance(
                    last_msg.content, str):
                response_parts.append(last_msg.content)

    # Combine response parts, filtering out error messages
    full_response = "\n".join([part for part in response_parts if part and "Please fix your mistakes" not in part])
    return full_response if full_response else "Sorry, I couldn't generate a complete response. Please try again."


if __name__ == "__main__":
    thread_id = input("Enter thread ID to resume a conversation (or press Enter for a new session): ").strip() or None
    mode = input("Choose input mode - [1] Type or [2] Speak: ").strip()
    user_input_lang = input(
        "🌐 Type language code - 'hi' for Hindi or 'en' for English (default: en): ").strip().lower() or "en"

    while True:
        if mode == "1":
            user_input = input("\n🧑 You (type): ")
        elif mode == "2":
            user_input = listen_to_user_whisper(language=user_input_lang)
        else:
            print("❌ Invalid mode selected. Defaulting to typing.")
            user_input = input("\n🧑 You (type): ")

        if user_input.lower() in ["quit", "q", "end"]:
            print("👋 Exiting chatbot.")
            break
        elif user_input.lower() == "show history":
            try:
                checkpoint = memory.get({"configurable": {"thread_id": thread_id}})
                if checkpoint and "messages" in checkpoint:
                    print("\n📜 Full conversation history:")
                    for msg in checkpoint["messages"]:
                        print(f"{msg.type}: {msg.content}")
                else:
                    print("No conversation history available.")
                continue
            except Exception as e:
                print(f"⚠️ Could not load conversation history: {e}")
                continue

        response = run_chatbot(user_input, thread_id, user_input_lang)
        print(f"\n🤖 Assistant: {response}")

        listen_option = input("\n🔈 Do you want to hear the assistant's reply? (y/n): ").strip().lower()
        if listen_option == "y":
            speak_output_gtts(response)
