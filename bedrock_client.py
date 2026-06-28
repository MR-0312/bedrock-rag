"""Bedrock access via the native Converse API (boto3) for chat + embeddings.

Why Converse instead of the OpenAI SDK: on Bedrock the OpenAI-compatible endpoint
only serves gpt-oss models. The Converse API speaks one uniform format to ALL
models (Nova, Llama, Mistral, Cohere, ...), so switching models is a one-line change.

Auth uses your Bedrock API key (bearer token) — boto3 reads AWS_BEARER_TOKEN_BEDROCK.
Embeddings use Amazon Titan Text Embeddings v2 (fast, server-side, 1024-dim).

Run this file directly for a smoke test:
    python bedrock_client.py
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
CHAT_MODEL_ID = os.getenv("CHAT_MODEL_ID", "us.amazon.nova-pro-v1:0")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

_EMBED_WORKERS = 24

# boto3 picks up the bearer token from AWS_BEARER_TOKEN_BEDROCK automatically.
# botocore clients are thread-safe; raise the pool so embedding fan-out isn't throttled.
_client = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(max_pool_connections=_EMBED_WORKERS),
)


def _to_converse(messages):
    """Split OpenAI-style messages into Converse (system list, message list)."""
    system, conv = [], []
    for m in messages:
        if m["role"] == "system":
            system.append({"text": m["content"]})
        else:
            conv.append({"role": m["role"], "content": [{"text": m["content"]}]})
    return system, conv


def chat(messages, model=None, max_tokens=1024, temperature=0.2):
    """Non-streaming chat. `messages` is a list of {"role", "content"} dicts."""
    system, conv = _to_converse(messages)
    kwargs = dict(
        modelId=model or CHAT_MODEL_ID,
        messages=conv,
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    if system:
        kwargs["system"] = system
    resp = _client.converse(**kwargs)
    return resp["output"]["message"]["content"][0]["text"]


def chat_stream(messages, model=None, max_tokens=1024, temperature=0.2, usage_out=None):
    """Streaming chat. Yields text chunks as they arrive.

    If `usage_out` (a dict) is given, it's populated with token/latency stats
    once the model finishes — handy for showing metrics in a UI.
    """
    system, conv = _to_converse(messages)
    kwargs = dict(
        modelId=model or CHAT_MODEL_ID,
        messages=conv,
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    if system:
        kwargs["system"] = system
    resp = _client.converse_stream(**kwargs)
    for event in resp["stream"]:
        if "contentBlockDelta" in event:
            yield event["contentBlockDelta"]["delta"].get("text", "")
        elif "metadata" in event and usage_out is not None:
            meta = event["metadata"]
            usage_out.update(meta.get("usage", {}))
            usage_out["latencyMs"] = meta.get("metrics", {}).get("latencyMs")


def _embed_one(text):
    resp = _client.invoke_model(modelId=EMBED_MODEL_ID, body=json.dumps({"inputText": text}))
    return json.loads(resp["body"].read())["embedding"]


def embed(texts, max_workers=_EMBED_WORKERS):
    """Embed a string or list of strings via Bedrock Titan. Returns a list of vectors.

    Calls are fanned out across threads so indexing many chunks is fast.
    """
    if isinstance(texts, str):
        return [_embed_one(texts)]
    if not texts:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(texts))) as ex:
        return list(ex.map(_embed_one, texts))


def list_models():
    """List foundation models available to your account (control-plane call)."""
    bedrock = boto3.client("bedrock", region_name=REGION)
    return [m["modelId"] for m in bedrock.list_foundation_models()["modelSummaries"]]


if __name__ == "__main__":
    print(f"Region:      {REGION}")
    print(f"Chat model:  {CHAT_MODEL_ID}")
    print(f"Embed model: {EMBED_MODEL_ID} (Bedrock)\n")

    print("Testing chat...")
    try:
        reply = chat([{"role": "user", "content": "Say 'Bedrock is working!' and nothing else."}])
        print("  ->", reply)
    except Exception as e:
        print(f"  [FAILED] {type(e).__name__}: {e}\n")
        print("Trying to list models your account can access...")
        try:
            for mid in list_models():
                print("  -", mid)
        except Exception as e2:
            print(f"  Could not list models either: {e2}")
        raise SystemExit(1)

    print("\nTesting Bedrock embeddings...")
    vec = embed("hello world")[0]
    print(f"  -> embedding length: {len(vec)} (first 3: {vec[:3]})")

    print("\n[OK] Smoke test passed. Your key works.")
