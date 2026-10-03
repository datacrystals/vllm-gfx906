#!/usr/bin/env python3
"""Send an image/video request to the MiMo omni server and print text + usage.

Usage: omni_send.py <image|video|text> <path-or-none> <prompt> [max_tokens]
"""
import base64, json, sys, urllib.request

def main():
    kind = sys.argv[1]
    path = sys.argv[2]
    prompt = sys.argv[3]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    content = []
    if kind == "image":
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    elif kind == "video":
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        content.append({"type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    body = {
        "model": "mimo-v2.6-flash",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:9700/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    print(json.dumps({
        "content": resp["choices"][0]["message"].get("content"),
        "reasoning": resp["choices"][0]["message"].get("reasoning_content"),
        "finish_reason": resp["choices"][0].get("finish_reason"),
        "usage": resp.get("usage"),
    }, indent=2))

if __name__ == "__main__":
    main()
