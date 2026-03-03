import base64
import mimetypes
import os

import requests
from anthropic import Anthropic
from openai import OpenAI


def _extract_openai_text(response):
    if not response.choices:
        return ""
    message = response.choices[0].message
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(chunk for chunk in text_chunks if chunk).strip()
    return str(content)


def _download_image_as_base64(image_url):
    image_response = requests.get(image_url, timeout=30)
    image_response.raise_for_status()
    content_type = image_response.headers.get("Content-Type", "").split(";")[0].strip()
    if not content_type:
        guessed_type, _ = mimetypes.guess_type(image_url)
        content_type = guessed_type or "image/jpeg"
    return base64.b64encode(image_response.content).decode("utf-8"), content_type


class AIClient:
    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "OPENAI").strip().upper()
        if self.provider not in {"OPENAI", "CLAUDE"}:
            raise ValueError("AI_PROVIDER must be OPENAI or CLAUDE")

        if self.provider == "OPENAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required when AI_PROVIDER=OPENAI")
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when AI_PROVIDER=CLAUDE")
            self.model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
            self.client = Anthropic(api_key=api_key)

    def analyze_images(self, prompt, image_urls):
        if self.provider == "OPENAI":
            content = [{"type": "text", "text": prompt}]
            content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
            )
            return _extract_openai_text(response)

        content = [{"type": "text", "text": prompt}]
        for url in image_urls:
            image_b64, media_type = _download_image_as_base64(url)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                }
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": content}],
        )
        text_chunks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "\n".join(text_chunks).strip()

