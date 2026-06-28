"""DocuMind (Chainlit) — Chat with your PDF, powered by Amazon Bedrock.

Run:  chainlit run chainlit_app.py -w
Reuses rag.py and bedrock_client.py unchanged.
"""
import asyncio

import chainlit as cl
from chainlit.input_widget import Select, Slider

from bedrock_client import chat_stream, CHAT_MODEL_ID
from rag import load_pdf, chunk_pages, VectorStore, build_prompt

# --- Models verified available on this Bedrock key (friendly name -> model id) ---
MODELS = {
    "Moonshot Kimi K2.5  ·  best": "moonshotai.kimi-k2.5",
    "DeepSeek V3.2": "deepseek.v3.2",
    "Mistral Large 3 (675B)": "mistral.mistral-large-3-675b-instruct",
    "Z.AI GLM-5": "zai.glm-5",
    "Qwen3 Next 80B": "qwen.qwen3-next-80b-a3b",
    "OpenAI gpt-oss-120b": "openai.gpt-oss-120b-1:0",
    "Amazon Nova Pro": "us.amazon.nova-pro-v1:0",
    "Amazon Nova 2 Lite  ·  fast": "us.amazon.nova-2-lite-v1:0",
}
# Real per-model max output-token limits (queried from Bedrock Converse).
MODEL_MAX_TOKENS = {
    "moonshotai.kimi-k2.5": 262144,
    "deepseek.v3.2": 163840,
    "mistral.mistral-large-3-675b-instruct": 262144,
    "zai.glm-5": 202752,
    "qwen.qwen3-next-80b-a3b": 262144,
    "openai.gpt-oss-120b-1:0": 128000,
    "us.amazon.nova-pro-v1:0": 10000,
    "us.amazon.nova-2-lite-v1:0": 65535,
}
_LABELS = list(MODELS)
_DEFAULT_IDX = next((i for i, k in enumerate(_LABELS) if MODELS[k] == CHAT_MODEL_ID), 0)

_SENTINEL = object()


async def _astream(make_gen):
    """Bridge a blocking sync generator to async so streaming doesn't block the loop."""
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def produce():
        try:
            for item in make_gen():
                loop.call_soon_threadsafe(q.put_nowait, item)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

    loop.run_in_executor(None, produce)
    while True:
        item = await q.get()
        if item is _SENTINEL:
            break
        yield item


async def _index_pdf(file):
    """Read, chunk and embed an uploaded PDF; store the index in the session."""
    async with cl.Step(name="Indexing document", type="tool") as step:
        pages = await cl.make_async(load_pdf)(file.path)
        chunks = await cl.make_async(chunk_pages)(pages)
        if not chunks:
            step.output = "No extractable text — this looks like a scanned/image PDF."
            await cl.Message(
                content="⚠️ **No extractable text found.** This PDF appears to be "
                "scanned/image-only. Try a text-based PDF."
            ).send()
            return False
        store = await cl.make_async(VectorStore)(chunks)
        cl.user_session.set("store", store)
        words = sum(len(c["text"].split()) for c in chunks)
        step.output = f"{len(pages)} pages · {len(chunks)} chunks · {words:,} words"

    await cl.Message(
        content=f"✅ **{file.name}** indexed — {len(pages)} pages, {len(chunks)} chunks. "
        "Ask me anything about it!"
    ).send()
    return True


@cl.on_chat_start
async def start():
    cl.user_session.set("settings", {
        "model": _LABELS[_DEFAULT_IDX], "temperature": 0.2,
        "max_tokens": 1024, "top_k": 4,
    })
    await cl.ChatSettings([
        Select(id="model", label="Model", values=_LABELS, initial_index=_DEFAULT_IDX),
        Slider(id="temperature", label="Temperature", initial=0.2, min=0, max=1, step=0.05),
        Slider(id="max_tokens", label="Max answer tokens", initial=1024, min=256, max=8192, step=128),
        Slider(id="top_k", label="Chunks retrieved (k)", initial=4, min=2, max=10, step=1),
    ]).send()

    files = None
    while not files:
        files = await cl.AskFileMessage(
            content="👋 **Welcome to DocuMind.** Upload a **text-based PDF** to start.",
            accept={"application/pdf": [".pdf"]},
            max_size_mb=50,
            timeout=240,
        ).send()
    if not await _index_pdf(files[0]):
        cl.user_session.set("await_reupload", True)


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("settings", settings)


@cl.on_message
async def on_message(message: cl.Message):
    store = cl.user_session.get("store")
    if not store:
        # Allow uploading a (new) PDF at any time.
        files = await cl.AskFileMessage(
            content="Upload a text-based PDF first:",
            accept={"application/pdf": [".pdf"]}, max_size_mb=50, timeout=240,
        ).send()
        if files:
            await _index_pdf(files[0])
        return

    s = cl.user_session.get("settings")
    model_id = MODELS[s["model"]]
    cap = MODEL_MAX_TOKENS.get(model_id, 4096)
    max_tokens = min(int(s["max_tokens"]), cap)
    temperature = float(s["temperature"])
    top_k = int(s["top_k"])

    contexts = await cl.make_async(store.search)(message.content, k=top_k)
    prompt = build_prompt(message.content, contexts)
    usage: dict = {}

    msg = cl.Message(content="")
    await msg.send()

    def gen():
        return chat_stream(prompt, model=model_id, max_tokens=max_tokens,
                           temperature=temperature, usage_out=usage)

    async for tok in _astream(gen):
        await msg.stream_token(tok)

    # Attach sources as side elements + a compact citation footer.
    elements, lines = [], []
    for i, c in enumerate(contexts, start=1):
        pct = int(max(0, min(1, c["score"])) * 100)
        name = f"[{i}] page {c['page']} · {pct}%"
        elements.append(cl.Text(name=name, content=c["text"], display="side"))
        lines.append(f"`{name}`")
    msg.elements = elements

    footer = "\n\n---\n**📚 Sources:** " + "  ".join(lines)
    if usage:
        tok_total = usage.get("totalTokens", "?")
        lat = usage.get("latencyMs")
        footer += f"\n\n*⚡ {s['model'].split('·')[0].strip()} · {tok_total} tokens"
        footer += f" · {lat/1000:.1f}s*" if lat else "*"
    msg.content += footer
    await msg.update()
