"""Web server for the AxiomCart HTML chat UI.

Project layout expected:
    web.py
    static/index.html
    src/main.py

Run from the project root with:
    uv run uvicorn web:app --reload
"""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.graph import axiomcart_graph


BASE_DIR = Path(__file__).resolve().parent
UI_FILE = BASE_DIR / "static" / "index.html"

app = FastAPI(title="AxiomCart Assistant")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
pending_questions: set[str] = set()
pending_questions_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


def get_thread_id(value: str | None) -> str:
    """Keep thread IDs in the UUID-hex format used by AxiomCartAssistant."""
    if value and re.fullmatch(r"[0-9a-fA-F]{32}", value):
        return value.lower()
    return uuid.uuid4().hex


@app.get("/")
def index() -> FileResponse:
    if not UI_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="UI not found. Save the HTML file as static/index.html.",
        )
    return FileResponse(UI_FILE)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")

    thread_id = get_thread_id(request.thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Resume an agent's HITL clarification when this conversation is waiting
        # for the customer's order ID/email; otherwise start a normal turn.
        with pending_questions_lock:
            is_resume = thread_id in pending_questions
        if is_resume:
            result = axiomcart_graph.invoke(Command(resume=message), config)
        else:
            result = axiomcart_graph.invoke(
                {"messages": [HumanMessage(content=message)], "user_query": message},
                config,
            )

        interrupts = result.get("__interrupt__", [])
        with pending_questions_lock:
            if interrupts:
                pending_questions.add(thread_id)
            else:
                pending_questions.discard(thread_id)

        answer = str(interrupts[0].value) if interrupts else result.get("final_answer", "")
        if not answer:
            answer = "Sorry, I wasn't able to process that. Could you try rephrasing?"
    except Exception as exc:
        # Keep the detailed exception in the server console, not in the response.
        print(f"AxiomCart chat request failed: {exc}")
        raise HTTPException(status_code=500, detail="The assistant could not process that message.") from exc

    return ChatResponse(answer=answer, thread_id=thread_id)
