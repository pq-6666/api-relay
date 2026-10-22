"""
Cassidy -> OpenAI 兼容网关

作用: 把 Cassidy 账号的额度和它提供的模型 (GPT / Claude / Gemini 等 Default Agents)
包装成标准的 OpenAI API:
    地址:  https://你的域名/v1
    密钥:  Authorization: Bearer <RELAY_API_KEY>   (你自己发的, 不是 Cassidy 的 Key)
其他 Agent / 工具按填 OpenAI 自定义接口的方式填这两个值就能调用, 烧的是 Cassidy credits。

原理: Cassidy 官方 API 不是 OpenAI 格式, 是 thread/message 格式:
    POST {CASSIDY_API_BASE}/assistants/thread/create   {"assistant_id": "..."}  -> thread_id
    POST {CASSIDY_API_BASE}/assistants/message/create  {"thread_id": "...", "message": "..."}
    鉴权头: x-api-key: <CASSIDY_API_KEY>
本网关每次收到 /v1/chat/completions 就新建一个 thread, 把多轮 messages 拼成一段文本发过去,
拿回回复再组装成 OpenAI 的返回格式。

已知限制 (先说清):
- Cassidy 是同步接口, 没有真流式; stream=true 时是等全文生成完一次性吐出 (假流式)。
- 不支持 OpenAI 的 tools/function calling、图片输入等参数, 传了会被忽略。
- 每次调用 = 一次 Cassidy Agent 聊天, 按 Cassidy 规则扣 credits (约 1-30/次, Premium 模型更贵)。
"""
import json
import os
import time
import urllib.request
import urllib.error
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CASSIDY_API_BASE = os.getenv("CASSIDY_API_BASE", "https://app.cassidyai.com/api").rstrip("/")
CASSIDY_API_KEY = os.getenv("CASSIDY_API_KEY", "")
RELAY_API_KEY = os.getenv("RELAY_API_KEY", "")  # 你发给其他 Agent 用的密钥
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "")
UPSTREAM_TIMEOUT = int(os.getenv("UPSTREAM_TIMEOUT", "300"))

# 模型名 -> Cassidy Assistant ID 的映射, 从 models.json 读 (见 models.example.json)
MODEL_MAP: dict = {}
_models_file = os.getenv("MODEL_MAP_FILE", os.path.join(os.path.dirname(__file__), "models.json"))
if os.path.exists(_models_file):
    with open(_models_file, encoding="utf-8") as f:
        MODEL_MAP = json.load(f)
elif os.getenv("MODEL_MAP"):
    MODEL_MAP = json.loads(os.environ["MODEL_MAP"])

app = FastAPI(title="cassidy-openai-gateway")


def _err(status: int, message: str):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "gateway_error", "code": status}},
    )


def _check_auth(request: Request):
    if not RELAY_API_KEY:
        return _err(500, "server RELAY_API_KEY not configured")
    auth = request.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if key != RELAY_API_KEY:
        return _err(401, "invalid api key")
    return None


def _cassidy_post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{CASSIDY_API_BASE}/{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": CASSIDY_API_KEY,
            "Content-Type": "application/json",
            # 必须带浏览器 UA: Cassidy 在 Cloudflare 后面, 程序默认 UA 会被 1010 拦截 (2026-10-22 实测)
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "identity",
            "Origin": "https://app.cassidyai.com",
            "Referer": "https://app.cassidyai.com/",
        },
        method="POST",
    )
    # 上游响应是 chunked 编码, 经代理时偶发 IncompleteRead/断连 (2026-10-22 实测),
    # 传输层错误自动重试 1 次; HTTP 状态错误 (401/4xx/5xx) 不重试, 原样抛出。
    last_exc = None
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:500]
            raise RuntimeError(f"cassidy upstream HTTP {e.code}: {body}") from e
        except Exception as e:  # IncompleteRead / RemoteDisconnected / URLError 等
            last_exc = e
    raise RuntimeError(f"cassidy upstream transport error after retry: {last_exc}")


def _flatten(messages: list) -> str:
    """多轮 messages 拼成一段文本。单条 user 消息直接原样发, 不加前缀。"""
    def text_of(m):
        c = m.get("content", "")
        if isinstance(c, list):  # OpenAI 多模态数组: 只取文本部分
            return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        return str(c)

    if len(messages) == 1 and messages[0].get("role") == "user":
        return text_of(messages[0])
    lines = []
    for m in messages:
        role = {"system": "System", "user": "User", "assistant": "Assistant"}.get(m.get("role"), "User")
        lines.append(f"{role}: {text_of(m)}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _chat_completion(model: str, messages: list) -> str:
    assistant_id = MODEL_MAP.get(model)
    if not assistant_id:
        raise LookupError(f"unknown model: {model}")
    thread = _cassidy_post("assistants/thread/create", {"assistant_id": assistant_id})
    thread_id = thread.get("thread_id") or (thread.get("thread") or {}).get("id")
    if not thread_id:
        raise RuntimeError(f"cassidy returned no thread_id: {str(thread)[:300]}")
    # 偶发: thread 刚建好 message 就打过去会 404 (上游一致性延迟, 2026-10-22 实测), 等 2 秒重试一次
    try:
        result = _cassidy_post(
            "assistants/message/create",
            {"thread_id": thread_id, "message": _flatten(messages)},
        )
    except RuntimeError as e:
        if "HTTP 404" not in str(e):
            raise
        time.sleep(2)
        result = _cassidy_post(
            "assistants/message/create",
            {"thread_id": thread_id, "message": _flatten(messages)},
        )
    content = result.get("message") or result.get("response") or result.get("content") or ""
    if isinstance(content, dict):
        content = content.get("content") or content.get("text") or json.dumps(content, ensure_ascii=False)
    return str(content)


def _responses_input_to_messages(body: dict) -> list:
    """OpenAI Responses 的 input 结构转成 chat messages。"""
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": str(body["instructions"])})
    inp = body.get("input")
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or "user"
            c = item.get("content", "")
            if isinstance(c, list):
                text = "".join(
                    p.get("text", "")
                    for p in c
                    if isinstance(p, dict) and p.get("type") in ("input_text", "text", "output_text")
                )
            else:
                text = str(c)
            if text:
                msgs.append({"role": role, "content": text})
    return msgs


@app.post("/v1/responses")
def responses(request: Request, body: dict):
    """最小兼容 OpenAI Responses API: 只做文本生成, 映射到同一条 Cassidy 链路。
    2026-10-22: 已有客户端在打这个端点 (此前 404), 故补上。"""
    if (e := _check_auth(request)):
        return e
    if not CASSIDY_API_KEY:
        return _err(500, "server CASSIDY_API_KEY not configured")
    model = body.get("model") or DEFAULT_MODEL
    messages = _responses_input_to_messages(body)
    if not messages:
        return _err(400, "input is required")
    try:
        content = _chat_completion(model, messages)
    except LookupError as e:
        return _err(404, str(e))
    except Exception as e:
        return _err(502, str(e))

    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    out_item = {
        "type": "message",
        "id": msg_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }
    resp = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": [out_item],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    if body.get("stream"):
        def gen():
            seq = 0

            def ev(name, payload):
                nonlocal seq
                payload = dict(payload)
                payload["type"] = name
                payload["sequence_number"] = seq
                seq += 1
                return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

            base = dict(resp)
            base["status"] = "in_progress"
            base["output"] = []
            yield ev("response.created", {"response": base})
            yield ev("response.output_item.added", {"output_index": 0, "item": {**out_item, "status": "in_progress", "content": []}})
            yield ev("response.content_part.added", {"item_id": msg_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
            yield ev("response.output_text.delta", {"item_id": msg_id, "output_index": 0, "content_index": 0, "delta": content})
            yield ev("response.output_text.done", {"item_id": msg_id, "output_index": 0, "content_index": 0, "text": content})
            yield ev("response.content_part.done", {"item_id": msg_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": content, "annotations": []}})
            yield ev("response.output_item.done", {"output_index": 0, "item": out_item})
            yield ev("response.completed", {"response": resp})

        return StreamingResponse(gen(), media_type="text/event-stream")
    return resp


@app.get("/health")
def health():
    return {"ok": True, "models": list(MODEL_MAP.keys())}


@app.get("/v1/models")
def list_models(request: Request):
    if (e := _check_auth(request)):
        return e
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "cassidy"}
            for m in MODEL_MAP
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: Request, body: dict):
    if (e := _check_auth(request)):
        return e
    if not CASSIDY_API_KEY:
        return _err(500, "server CASSIDY_API_KEY not configured")
    model = body.get("model") or DEFAULT_MODEL
    messages = body.get("messages") or []
    if not messages:
        return _err(400, "messages is required")
    try:
        content = _chat_completion(model, messages)
    except LookupError as e:
        return _err(404, str(e))
    except Exception as e:  # 上游错误原样带回, 方便排查
        return _err(502, str(e))

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        # Cassidy 不返回 token 数, 这里给 0, 需要计费统计的客户端请注意
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if body.get("stream"):
        def gen():
            chunk = dict(result)
            chunk["object"] = "chat.completion.chunk"
            chunk["choices"] = [
                {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}
            ]
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            end = dict(chunk)
            end["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    return result
