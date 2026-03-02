"""
Educational Video Analyzer - 2-Hour Agent Project (YouTube transcript tools).
Summary with bullet points and key quotes; chapter timestamps; Q&A; compare videos.
"""
from __future__ import annotations

import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
try:
    from load_secrets import load_secrets
    load_secrets()
except Exception:
    pass

import re
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
try:
    from langchain.agents import create_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent as create_agent

# ---------------------------------------------------------------------------
# Environment: load .env if available, require OPENAI_API_KEY
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "OPENAI_API_KEY is not set. Set it in your environment or create a .env file with OPENAI_API_KEY=your-key"
    )


# ---------------------------------------------------------------------------
# Helper: format seconds as M:SS or H:MM:SS (for chapter timestamps)
# Used when we return timestamped transcript so the LLM sees human-readable times.
# ---------------------------------------------------------------------------
def format_time(seconds: float) -> str:
    """Convert seconds (float) to string M:SS or H:MM:SS."""
    s = int(round(seconds))
    if s < 3600:
        m, s = divmod(s, 60)
        return f"{m}:{s:02d}"
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Tool 1: get_youtube_transcript(video_id) -> plain text (no timestamps)
# The agent uses this when only the text is needed (summary, Q&A, comparison).
# ---------------------------------------------------------------------------
@tool
def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video by video ID. Returns plain text only (no timestamps)."""
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(entry["text"] for entry in transcript)
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    return " ".join(snippet.text for snippet in transcript)


# ---------------------------------------------------------------------------
# Tool 2: get_youtube_transcript_timestamped(video_id) -> text with [M:SS] prefixes
# The agent uses this when we need timestamps, e.g. to infer chapters.
# Each line is "[M:SS] sentence or phrase" so the LLM can suggest chapter boundaries.
# ---------------------------------------------------------------------------
def _fetch_transcript_snippets(video_id: str) -> list[dict]:
    """
    Internal: fetch raw transcript as list of dicts with keys 'text' and 'start'.
    Unified so we can support both old API (get_transcript) and new API (fetch).
    """
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        entries = YouTubeTranscriptApi.get_transcript(video_id)
        return [{"text": e["text"], "start": e.get("start", 0)} for e in entries]
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id)
    # New API: snippet has .text and .start
    return [{"text": getattr(s, "text", str(s)), "start": getattr(s, "start", 0)} for s in fetched]


@tool
def get_youtube_transcript_timestamped(video_id: str) -> str:
    """Fetch the transcript with timestamps. Returns one line per segment: [M:SS] or [H:MM:SS] followed by text. Use this when you need to identify chapter start times."""
    snippets = _fetch_transcript_snippets(video_id)
    lines = [f"[{format_time(s['start'])}] {s['text']}" for s in snippets]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# URL/ID parsing: extract 11-char video ID from URL or accept raw ID
# ---------------------------------------------------------------------------
def extract_video_id(value: str) -> str:
    """Extract YouTube video ID from URL or return as-is if it looks like an ID."""
    value = value.strip()
    m = re.search(r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})", value)
    if m:
        return m.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", value):
        return value
    raise ValueError("Could not find a YouTube video ID. Use a URL or an 11-character video ID.")


def extract_video_ids(line: str) -> list[str]:
    """Parse a line that may contain multiple URLs or IDs (comma/space separated). Returns list of video IDs."""
    ids = []
    for part in re.split(r"[\s,]+", line.strip()):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(extract_video_id(part))
        except ValueError:
            continue
    return ids


# ---------------------------------------------------------------------------
# Agent: we give it both tools so it can fetch plain or timestamped transcript
# depending on the task (chapters need timestamped; summary/Q&A/comparison use plain).
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
agent = create_agent(llm, [get_youtube_transcript, get_youtube_transcript_timestamped])


# ---------------------------------------------------------------------------
# FEATURE 1: Summary with bullet points and key quotes
# No new tools; we just change the prompt to ask for bullets and direct quotes.
# ---------------------------------------------------------------------------
EDUCATIONAL_PROMPT_FORMATTED = """Use the transcript from the video to produce the following. Format clearly with the section headers below.

## Summary
Summarize the video in **bullet points** (5ΓÇô7 bullets). Each bullet should capture one main idea.

## Key Quotes
List 3ΓÇô5 short **direct quotes** from the transcript that capture important or memorable ideas. Put each quote in quotation marks and keep them brief (1ΓÇô2 sentences each).

## Key Concepts
List the main concepts or takeaways as bullet points.

## Quiz Questions
Write 3ΓÇô5 multiple-choice or short-answer quiz questions to test understanding. For each question provide the correct answer."""


# ---------------------------------------------------------------------------
# FEATURE 2: Extract chapter timestamps
# We ask the agent to use the *timestamped* transcript and output suggested
# chapters as "timestamp ΓÇô chapter title". The timestamped tool returns lines
# like "[0:00] ..." so the model can pick meaningful boundaries.
# ---------------------------------------------------------------------------
CHAPTERS_PROMPT = """Use the timestamped transcript above. Identify logical chapters (section boundaries) and list them in this format, one per line:

timestamp ΓÇô Chapter title

Example:
0:00 ΓÇô Introduction
1:30 ΓÇô Main method
5:00 ΓÇô Results and discussion

Use the timestamps that appear in the transcript. Give each chapter a short, descriptive title."""


# ---------------------------------------------------------------------------
# FEATURE 3: Answer specific question ΓÇô we just pass the user's question
# in the prompt and ask the agent to use the transcript to answer.
# ---------------------------------------------------------------------------
def build_question_prompt(video_id: str, question: str) -> str:
    return (
        f"First, get the transcript for video {video_id}. "
        f"Then, using only the transcript, answer this question:\n\n{question}"
    )


# ---------------------------------------------------------------------------
# FEATURE 4: Compare multiple videos
# We ask the agent to fetch transcript for each video (by calling the tool
# for each video_id we provide), then compare them. We pass a list of
# video IDs in the prompt so the agent knows to fetch each one.
# ---------------------------------------------------------------------------
COMPARE_PROMPT_TEMPLATE = """You must get the transcript for each of these video IDs by calling get_youtube_transcript once for each:
{video_ids_list}

Call the tool for each video, then compare and contrast the transcripts. In your response include:

## Common themes
What topics or ideas appear in both (or all) videos?

## Differences
How do the videos differ in focus, depth, or perspective?

## Per-video overview
For each video ID, give a one-paragraph summary of what it covers.

Video IDs to fetch: {video_ids}"""


def build_compare_prompt(video_ids: list[str]) -> str:
    list_part = "\n".join(f"- {vid}" for vid in video_ids)
    return COMPARE_PROMPT_TEMPLATE.format(
        video_ids_list=list_part,
        video_ids=", ".join(video_ids),
    )


# ---------------------------------------------------------------------------
# Shared: invoke agent and return full message history + last assistant content
# We need the message list so we can continue the conversation (multi-turn Q&A).
# ---------------------------------------------------------------------------
def _handle_rate_limit() -> None:
    """Raise SystemExit with billing message on quota errors."""
    raise SystemExit(
        "OpenAI quota exceeded. Add payment method or wait for quota reset: "
        "https://platform.openai.com/account/billing"
    )


def run_agent(messages: list) -> tuple[list, str]:
    """
    Invoke the agent with the given message list (can be initial or conversation history).
    Returns (full_messages_after_invoke, last_assistant_content).
    Use full_messages when calling run_agent again for follow-up questions.
    """
    try:
        result = agent.invoke({"messages": messages})
    except openai.RateLimitError as e:
        if "quota" in str(e).lower() or "insufficient_quota" in str(e).lower():
            _handle_rate_limit()
        raise
    out_messages = result.get("messages", [])
    last = out_messages[-1] if out_messages else None
    content = last.content if (last and hasattr(last, "content")) else str(result)
    return (out_messages, content)


def run_agent_and_print(user_message: str) -> None:
    """One-shot: one user message, print the final answer. Used for compare (fresh context)."""
    _, content = run_agent([("user", user_message)])
    print(content)


# ---------------------------------------------------------------------------
# Main: new workflow
# 1. User enters one video URL ΓåÆ we generate transcript + summary/quiz etc. and show it.
# 2. Loop: user can (1) ask questions about the video, (2) generate chapters,
#    (3) compare with other videos, or (4) quit. We keep conversation state so
#    follow-up questions use the same transcript/summary context.
# ---------------------------------------------------------------------------
def main() -> None:
    # ----- Step 1: Get video and run full analysis (transcript + summary, quotes, concepts, quiz) -----
    print("Educational Video Analyzer")
    print("Enter a YouTube URL or video ID to analyze.")
    video_input = input("Video URL or ID: ").strip() or "dQw4w9WgXcQ"
    try:
        video_id = extract_video_id(video_input)
    except ValueError as e:
        raise SystemExit(e) from e

    initial_message = (
        f"First, get the transcript for video {video_id}. "
        f"Then, based on that transcript, do the following:\n\n{EDUCATIONAL_PROMPT_FORMATTED}"
    )
    print("\nGenerating transcript, summary, key quotes, concepts, and quiz questions...\n")
    messages, content = run_agent([("user", initial_message)])
    print(content)

    # ----- Step 2: Follow-up loop ΓÇô ask a question, chapters, compare, or quit -----
    while True:
        print("\n" + "ΓöÇ" * 60)
        print("What would you like to do next?")
        print("  1. Ask a question about this video")
        print("  2. Generate chapter timestamps for this video")
        print("  3. Compare this video with other video(s)")
        print("  4. Quit")
        choice = input("Choose 1ΓÇô4: ").strip()

        if choice == "4" or choice.lower() == "q":
            print("Goodbye.")
            break

        if choice == "1":
            # ----- Follow-up: answer a question using existing conversation -----
            # We append the user's question to the message history and re-invoke.
            # The agent already has the transcript (from tool result) in context.
            question = input("Your question: ").strip()
            if not question:
                continue
            # Append new user message; agent expects list of message objects or (role, content).
            # LangGraph state uses message objects, so we append HumanMessage for consistency.
            new_messages = list(messages) + [HumanMessage(content=question)]
            messages, content = run_agent(new_messages)
            print("\n" + content)

        elif choice == "2":
            # ----- Generate chapters for the same video -----
            # We add a new user message asking for chapters via the timestamped transcript.
            # The agent will call get_youtube_transcript_timestamped(video_id) and format chapters.
            chapters_message = (
                f"For the same video we analyzed (video ID: {video_id}), "
                f"get the timestamped transcript using get_youtube_transcript_timestamped and "
                f"then do the following:\n\n{CHAPTERS_PROMPT}"
            )
            new_messages = list(messages) + [HumanMessage(content=chapters_message)]
            messages, content = run_agent(new_messages)
            print("\n" + content)

        elif choice == "3":
            # ----- Compare this video with others -----
            # User enters additional URL(s). We build a compare prompt for [video_id] + others
            # and run a fresh invoke (no need to attach to previous conversation).
            other_input = input("Enter other video URL(s) to compare (comma or space separated): ").strip()
            other_ids = extract_video_ids(other_input)
            if not other_ids:
                print("No valid URLs or IDs entered. Skipping.")
                continue
            all_ids = [video_id] + other_ids
            user_message = build_compare_prompt(all_ids)
            print("\nComparing videos...\n")
            run_agent_and_print(user_message)
            # After compare we don't update `messages` ΓÇô the next loop turn still has
            # the original analysis context for more questions or chapters.

        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
