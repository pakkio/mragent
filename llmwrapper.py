import base64
import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


@dataclass
class Prompt:
    user: str
    system: Optional[str] = None
    history: list[dict] = field(default_factory=list)

    def to_messages(self) -> list[dict]:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(self.history)
        messages.append({"role": "user", "content": self.user})
        return messages


class LLMWrapper:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_APIKEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_APIKEY not set in environment or .env")

    def _request(self, payload: dict) -> dict:
        req = urllib.request.Request(
            OPENROUTER_API_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                self.last_model = data.get("model", "unknown")
                print(f"[openrouter] model used: {self.last_model}")
                return data
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"OpenRouter error {e.code}: {body}") from e

    def call(self, prompt: Prompt) -> str:
        data = self._request({"model": self.model, "messages": prompt.to_messages()})
        return data["choices"][0]["message"]["content"]

    def call_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Send messages with tool definitions; return the raw assistant message dict."""
        payload: dict = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = self._request(payload)
        if "choices" not in data:
            raise RuntimeError(f"Unexpected API response (no choices): {data}")
        return data["choices"][0]["message"]

    def chat(self, user_msg: str, system: Optional[str] = None) -> str:
        return self.call(Prompt(user=user_msg, system=system))

    def extract_pdf_openrouter(self, path: str, engine: str = "mistral-ocr") -> str:
        """Send a PDF through OpenRouter's file-parser plugin and return extracted text."""
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": self.model,
            "plugins": [{"id": "file-parser", "pdf": {"engine": engine}}],
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": os.path.basename(path),
                            "file_data": f"data:application/pdf;base64,{b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Output ALL text content extracted from this document. "
                            "Preserve headings and paragraph breaks. "
                            "Output only the document text, nothing else."
                        ),
                    },
                ],
            }],
        }
        data = self._request(payload)
        return data["choices"][0]["message"]["content"]

    def extract_pdf_mistral_ocr4(self, path: str) -> str:
        """Call Mistral OCR 4 directly (requires MISTRAL_API_KEY). Returns markdown."""
        mistral_key = os.environ.get("MISTRAL_API_KEY")
        if not mistral_key:
            raise ValueError("MISTRAL_API_KEY not set — required for Mistral OCR 4")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_base64",
                "document_base64": b64,
                "document_name": os.path.basename(path),
            },
            "include_image_base64": False,
        }
        req = urllib.request.Request(
            "https://api.mistral.ai/v1/ocr",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {mistral_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"Mistral OCR error {e.code}: {body}") from e
        pages = data.get("pages", [])
        print(f"[mistral-ocr-4] extracted {len(pages)} page(s)")
        return "\n\n".join(p.get("markdown", "") for p in pages)
