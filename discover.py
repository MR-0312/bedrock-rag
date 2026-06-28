"""Discover the best non-Anthropic chat model available to THIS key in us-east-1.

1) Lists all text foundation models the account can see.
2) Live-tests a curated 'best' shortlist via Converse to confirm real access.
"""
import os

import boto3
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv("AWS_REGION", "us-east-1")

cp = boto3.client("bedrock", region_name=REGION)        # control plane (catalog)
rt = boto3.client("bedrock-runtime", region_name=REGION)  # data plane (inference)

# --- 1. Catalog ---------------------------------------------------------------
print(f"=== Text models visible in {REGION} (non-Anthropic) ===\n")
catalog = {}
try:
    for m in cp.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]:
        provider = m.get("providerName", "?")
        if provider.lower() == "anthropic":
            continue
        catalog.setdefault(provider, []).append(m["modelId"])
    for provider in sorted(catalog):
        print(f"[{provider}]")
        for mid in sorted(catalog[provider]):
            print(f"   {mid}")
        print()
except Exception as e:
    print(f"(could not list catalog: {type(e).__name__}: {e})\n")

# --- 2. Live test a 'best' shortlist -----------------------------------------
# Ordered roughly best-first. We test inference-profile (us.) and direct IDs.
SHORTLIST = [
    ("OpenAI gpt-oss-120b",   "openai.gpt-oss-120b-1:0"),
    ("Amazon Nova Premier",   "us.amazon.nova-premier-v1:0"),
    ("DeepSeek-R1",           "us.deepseek.r1-v1:0"),
    ("Meta Llama 4 Maverick", "us.meta.llama4-maverick-17b-instruct-v1:0"),
    ("Meta Llama 3.3 70B",    "us.meta.llama3-3-70b-instruct-v1:0"),
    ("Mistral Large",         "mistral.mistral-large-2407-v1:0"),
    ("Amazon Nova Pro",       "us.amazon.nova-pro-v1:0"),
]

print("=== Live access test (Converse) ===\n")
working = []
for name, mid in SHORTLIST:
    try:
        rt.converse(
            modelId=mid,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 5},
        )
        print(f"  [OK]  {name:22} {mid}")
        working.append((name, mid))
    except Exception as e:
        print(f"  [no]  {name:22} {mid}  ({type(e).__name__}: {str(e)[:70]})")

print()
if working:
    name, mid = working[0]
    print(f"[BEST AVAILABLE] {name}")
    print(f"  -> set CHAT_MODEL_ID={mid}")
else:
    print("No shortlist model worked. See the catalog above and enable one in")
    print("AWS Console -> Bedrock -> Model access, then re-run.")
