"""FastAPI backend for MyPalClara assistant."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from db import init_db, SessionLocal
from models import Project, Session, Message
from memory_manager import MemoryManager, load_initial_profile
from llm_backends import make_llm

load_dotenv()

USER_ID = os.getenv("USER_ID", "demo-user")
DEFAULT_PROJECT = os.getenv("DEFAULT_PROJECT", "Default Project")

app = FastAPI(title="MyPalClara API")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors before returning 422."""
    body = await request.body()
    print(f"[api] Validation error!")
    print(f"[api] URL: {request.url}")
    print(f"[api] Body: {body}")
    print(f"[api] Errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# CORS for frontend - allow all origins in Docker for LAN access
# You can restrict this by setting CORS_ORIGINS env var (comma-separated)
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    cors_origins = [o.strip() for o in cors_origins_env.split(",")]
    allow_creds = True
else:
    # Default: allow all origins for easier LAN access
    # Note: credentials must be False when using wildcard origins
    cors_origins = ["*"]
    allow_creds = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)

mm: MemoryManager = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ContextRequest(BaseModel):
    message: str
    thread_id: str = None  # Optional: use specific thread
    project: str = DEFAULT_PROJECT


class StoreRequest(BaseModel):
    user_message: str
    assistant_message: str
    thread_id: str = None  # Optional: use specific thread
    project: str = DEFAULT_PROJECT


class ThreadRenameRequest(BaseModel):
    title: str


class MessageAppendRequest(BaseModel):
    role: str
    content: str
    id: str = None
    createdAt: str = None


class ChatRequest(BaseModel):
    message: str
    thread_id: str = None
    project: str = DEFAULT_PROJECT


@app.on_event("startup")
def startup():
    """Initialize database and memory manager on startup."""
    global mm
    print("[api] Starting up...")
    init_db()
    print("[api] Database initialized")
    llm = make_llm()
    print("[api] LLM created")
    mm = MemoryManager(llm_callable=llm)
    print("[api] MemoryManager initialized")
    load_initial_profile(USER_ID)
    print("[api] Initial profile loaded")
    print("[api] Ready to accept requests on http://localhost:8000")


def ensure_project(name: str) -> str:
    """Ensure project exists and return its ID."""
    db = SessionLocal()
    try:
        proj = db.query(Project).filter_by(owner_id=USER_ID, name=name).first()
        if not proj:
            proj = Project(owner_id=USER_ID, name=name)
            db.add(proj)
            db.commit()
            db.refresh(proj)
        return proj.id
    finally:
        db.close()


@app.post("/api/context")
def get_context(request: ContextRequest):
    """Get enriched context for a message (system prompt + memories)."""
    print(f"[api] /api/context called with message: {request.message[:50]}...")
    project_id = ensure_project(request.project or DEFAULT_PROJECT)

    db = SessionLocal()
    try:
        # Use specific thread if provided, otherwise create/get default
        if request.thread_id:
            sess = db.query(Session).filter_by(id=request.thread_id).first()
            if not sess:
                raise HTTPException(status_code=404, detail="Thread not found")
        else:
            sess = mm._get_or_create_session(db, USER_ID, project_id)

        recent_msgs = mm._get_recent_messages(db, sess.id)

        # Get mem0 memories
        user_mems, proj_mems = mm._fetch_mem0_context(
            USER_ID, project_id, request.message
        )

        # Build the full prompt
        prompt_messages = mm._build_prompt(
            user_mems,
            proj_mems,
            sess.context_snapshot,
            sess.session_summary,
            recent_msgs,
            request.message,
        )

        return {
            "messages": prompt_messages,
            "session_id": sess.id,
        }
    finally:
        db.close()


@app.post("/api/store")
def store_messages(request: StoreRequest):
    """Store user and assistant messages after completion."""
    project_id = ensure_project(request.project or DEFAULT_PROJECT)

    db = SessionLocal()
    try:
        # Use specific thread if provided
        if request.thread_id:
            sess = db.query(Session).filter_by(id=request.thread_id).first()
            if not sess:
                raise HTTPException(status_code=404, detail="Thread not found")
        else:
            sess = mm._get_or_create_session(db, USER_ID, project_id)

        recent_msgs = mm._get_recent_messages(db, sess.id)

        # Store messages
        mm._store_message(db, sess.id, USER_ID, "user", request.user_message)
        mm._store_message(db, sess.id, USER_ID, "assistant", request.assistant_message)
        sess.last_activity_at = datetime.now(timezone.utc).replace(tzinfo=None)

        # Auto-generate title from first user message if not set
        if not sess.title and request.user_message:
            title = request.user_message[:50]
            if len(request.user_message) > 50:
                title += "..."
            sess.title = title

        db.commit()

        # Add to mem0
        mm._add_to_mem0(
            USER_ID, project_id, recent_msgs,
            request.user_message, request.assistant_message
        )

        return {"status": "ok", "thread_id": sess.id}
    finally:
        db.close()


@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Chat endpoint for web UI.
    Gets context, calls LLM, stores messages, returns response.
    """
    print(f"[api] /api/chat: {request.message[:50]}...")
    project_id = ensure_project(request.project or DEFAULT_PROJECT)

    db = SessionLocal()
    try:
        # Get or use specific thread
        if request.thread_id:
            sess = db.query(Session).filter_by(id=request.thread_id).first()
            if not sess:
                raise HTTPException(status_code=404, detail="Thread not found")
        else:
            sess = mm._get_or_create_session(db, USER_ID, project_id)

        recent_msgs = mm._get_recent_messages(db, sess.id)

        # Get mem0 memories
        user_mems, proj_mems = mm._fetch_mem0_context(
            USER_ID, project_id, request.message
        )

        # Build the full prompt
        prompt_messages = mm._build_prompt(
            user_mems,
            proj_mems,
            sess.context_snapshot,
            sess.session_summary,
            recent_msgs,
            request.message,
        )

        # Call LLM
        llm = make_llm()
        response = llm(prompt_messages)

        # Store messages
        mm._store_message(db, sess.id, USER_ID, "user", request.message)
        mm._store_message(db, sess.id, USER_ID, "assistant", response)
        sess.last_activity_at = datetime.now(timezone.utc).replace(tzinfo=None)

        # Auto-generate title from first user message if not set
        if not sess.title and request.message:
            title = request.message[:50]
            if len(request.message) > 50:
                title += "..."
            sess.title = title

        db.commit()

        # Add to mem0
        mm._add_to_mem0(
            USER_ID, project_id, recent_msgs,
            request.message, response
        )

        print(f"[api] /api/chat response: {len(response)} chars")
        return {
            "content": response,
            "thread_id": sess.id,
        }
    finally:
        db.close()


@app.get("/api/projects")
def get_projects():
    """Get list of projects for the user."""
    db = SessionLocal()
    try:
        projects = db.query(Project).filter_by(owner_id=USER_ID).all()
        return {"projects": [p.name for p in projects] or [DEFAULT_PROJECT]}
    finally:
        db.close()


# ============== Thread Management API ==============

@app.get("/api/threads")
def list_threads():
    """List all threads for the user."""
    project_id = ensure_project(DEFAULT_PROJECT)
    db = SessionLocal()
    try:
        sessions = (
            db.query(Session)
            .filter_by(user_id=USER_ID, project_id=project_id)
            .order_by(Session.last_activity_at.desc())
            .all()
        )
        threads = []
        for sess in sessions:
            # Determine status: pinned > regular > archived
            if sess.archived == "pinned":
                status = "regular"  # Show as regular but will have isPinned flag
            elif sess.archived == "true":
                status = "archived"
            else:
                status = "regular"

            threads.append({
                "remoteId": sess.id,
                "status": status,
                "title": sess.title,
                "isPinned": sess.archived == "pinned",
            })

        # Sort: pinned first, then by activity
        threads.sort(key=lambda t: (not t.get("isPinned", False), 0))
        return {"threads": threads}
    finally:
        db.close()


@app.post("/api/threads")
def create_thread():
    """Create a new thread and return its ID."""
    project_id = ensure_project(DEFAULT_PROJECT)
    db = SessionLocal()
    try:
        sess = Session(
            project_id=project_id,
            user_id=USER_ID,
            title=None,
            archived="false",
        )
        db.add(sess)
        db.commit()
        db.refresh(sess)
        print(f"[api] Created new thread: {sess.id}")
        return {"remoteId": sess.id}
    finally:
        db.close()


@app.get("/api/threads/{thread_id}/messages")
def get_thread_messages(thread_id: str):
    """Get all messages for a thread."""
    db = SessionLocal()
    try:
        sess = db.query(Session).filter_by(id=thread_id).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Thread not found")

        messages = (
            db.query(Message)
            .filter_by(session_id=thread_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        return {
            "messages": [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": [{"type": "text", "text": msg.content}],
                    "createdAt": msg.created_at.isoformat(),
                }
                for msg in messages
            ]
        }
    finally:
        db.close()


@app.post("/api/threads/{thread_id}/messages")
def append_message(thread_id: str, request: MessageAppendRequest):
    """Append a message to a thread."""
    db = SessionLocal()
    try:
        sess = db.query(Session).filter_by(id=thread_id).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Thread not found")

        msg = Message(
            session_id=thread_id,
            user_id=USER_ID,
            role=request.role,
            content=request.content,
        )
        db.add(msg)
        sess.last_activity_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        print(f"[api] Appended {request.role} message to thread {thread_id}")
        return {"status": "ok", "id": str(msg.id)}
    finally:
        db.close()


@app.put("/api/threads/{thread_id}")
def rename_thread(thread_id: str, request: ThreadRenameRequest):
    """Rename a thread."""
    db = SessionLocal()
    try:
        sess = db.query(Session).filter_by(id=thread_id).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Thread not found")

        sess.title = request.title
        db.commit()
        print(f"[api] Renamed thread {thread_id} to: {request.title}")
        return {"status": "ok"}
    finally:
        db.close()


@app.post("/api/threads/{thread_id}/generate-title")
def generate_thread_title(thread_id: str):
    """Generate a title for the thread using LLM based on messages."""
    db = SessionLocal()
    try:
        sess = db.query(Session).filter_by(id=thread_id).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Get the first few messages
        messages = (
            db.query(Message)
            .filter_by(session_id=thread_id)
            .order_by(Message.created_at.asc())
            .limit(4)
            .all()
        )

        if not messages:
            return {"title": "New Chat"}

        # Build context for title generation
        context = "\n".join([
            f"{msg.role}: {msg.content[:200]}"
            for msg in messages
            if msg.content
        ])

        # Use LLM to generate a short title
        title_prompt = [
            {
                "role": "system",
                "content": "Generate a very short title (3-6 words) that summarizes this conversation. Return ONLY the title, nothing else. No quotes, no punctuation at the end."
            },
            {
                "role": "user",
                "content": f"Conversation:\n{context}"
            }
        ]

        try:
            llm = make_llm()
            title = llm(title_prompt).strip()
            # Clean up and truncate
            title = title.strip('"\'').strip()
            if len(title) > 50:
                title = title[:47] + "..."
        except Exception as e:
            print(f"[api] Error generating title with LLM: {e}")
            # Fallback to first user message
            first_user = next((m for m in messages if m.role == "user"), None)
            if first_user and first_user.content:
                title = first_user.content[:50]
                if len(first_user.content) > 50:
                    title += "..."
            else:
                title = "New Chat"

        # Save the title
        sess.title = title
        db.commit()
        print(f"[api] Generated title for thread {thread_id}: {title}")

        return {"title": title}
    finally:
        db.close()


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str):
    """Archive (soft delete) a thread."""
    db = SessionLocal()
    try:
        sess = db.query(Session).filter_by(id=thread_id).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Prevent archiving pinned threads
        if sess.archived == "pinned":
            raise HTTPException(status_code=400, detail="Cannot archive pinned thread")

        sess.archived = "true"
        db.commit()
        print(f"[api] Archived thread {thread_id}")
        return {"status": "ok"}
    finally:
        db.close()


@app.get("/api/threads/{thread_id}")
def get_thread(thread_id: str):
    """Get thread metadata."""
    db = SessionLocal()
    try:
        sess = db.query(Session).filter_by(id=thread_id).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Thread not found")

        if sess.archived == "pinned":
            status = "regular"
        elif sess.archived == "true":
            status = "archived"
        else:
            status = "regular"

        return {
            "remoteId": sess.id,
            "status": status,
            "title": sess.title,
            "isPinned": sess.archived == "pinned",
        }
    finally:
        db.close()


@app.post("/api/threads/{thread_id}/unarchive")
def unarchive_thread(thread_id: str):
    """Unarchive a thread."""
    db = SessionLocal()
    try:
        sess = db.query(Session).filter_by(id=thread_id).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Thread not found")

        sess.archived = "false"
        db.commit()
        print(f"[api] Unarchived thread {thread_id}")
        return {"status": "ok"}
    finally:
        db.close()


@app.get("/health")
def health():
    """Health check endpoint."""
    print("[api] Health check called")
    return {"status": "ok"}


@app.post("/api/test")
def test_post(request: ContextRequest):
    """Test endpoint to verify POST requests work."""
    print(f"[api] Test endpoint called with: {request}")
    return {"received": request.model_dump()}


# ============== Memory Management API ==============

from mem0_config import MEM0


class MemoryUpdateRequest(BaseModel):
    text: str


@app.get("/api/memories")
def list_memories(project_id: str = None):
    """List all memories for the user, optionally filtered by project."""
    if MEM0 is None:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        result = MEM0.get_all(user_id=USER_ID)
        memories = result.get("results", []) if isinstance(result, dict) else result

        # Filter by project if specified
        if project_id:
            memories = [
                m for m in memories
                if m.get("metadata", {}).get("project_id") == project_id
            ]

        print(f"[api] Listed {len(memories)} memories")
        return {"memories": memories}
    except Exception as e:
        print(f"[api] Error listing memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memories/{memory_id}")
def get_memory(memory_id: str):
    """Get a specific memory by ID."""
    if MEM0 is None:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        result = MEM0.get(memory_id)
        if not result:
            raise HTTPException(status_code=404, detail="Memory not found")
        print(f"[api] Retrieved memory {memory_id}")
        return {"memory": result}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[api] Error getting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/memories/{memory_id}")
def update_memory(memory_id: str, request: MemoryUpdateRequest):
    """Update a memory's text."""
    if MEM0 is None:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        result = MEM0.update(memory_id, request.text)
        print(f"[api] Updated memory {memory_id}")
        return {"status": "ok", "result": result}
    except Exception as e:
        print(f"[api] Error updating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str):
    """Delete a specific memory."""
    if MEM0 is None:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        MEM0.delete(memory_id)
        print(f"[api] Deleted memory {memory_id}")
        return {"status": "ok"}
    except Exception as e:
        print(f"[api] Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memories")
def delete_all_memories(project_id: str = None):
    """Delete all memories for the user, optionally filtered by project."""
    if MEM0 is None:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        if project_id:
            # Get all memories and delete those matching project
            result = MEM0.get_all(user_id=USER_ID)
            memories = result.get("results", []) if isinstance(result, dict) else result
            deleted = 0
            for m in memories:
                if m.get("metadata", {}).get("project_id") == project_id:
                    MEM0.delete(m["id"])
                    deleted += 1
            print(f"[api] Deleted {deleted} memories for project {project_id}")
            return {"status": "ok", "deleted": deleted}
        else:
            MEM0.delete_all(user_id=USER_ID)
            print(f"[api] Deleted all memories for user {USER_ID}")
            return {"status": "ok"}
    except Exception as e:
        print(f"[api] Error deleting memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memories/search")
def search_memories(request: ContextRequest):
    """Search memories by query."""
    if MEM0 is None:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        result = MEM0.search(
            request.message,
            user_id=USER_ID,
        )
        memories = result.get("results", []) if isinstance(result, dict) else result
        print(f"[api] Search found {len(memories)} memories")
        return {"memories": memories}
    except Exception as e:
        print(f"[api] Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
