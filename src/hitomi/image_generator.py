"""Image Generator supporting multiple providers."""

import os
import base64
import httpx
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Engine for generating images using Gemini or Qwen Wanx."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or os.getenv("IMAGE_PROVIDER", "qwen")

        if self.provider == "qwen":
            self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            if not self.api_key:
                logger.warning("No QWEN_API_KEY found, trying Gemini...")
                self.provider = "gemini"
            else:
                self.model = model or os.getenv("QWEN_IMAGE_MODEL", "wanx-v1")
                self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
                self.task_url = "https://dashscope.aliyuncs.com/api/v1/tasks"
                logger.info(f"ImageGenerator initialized with Qwen Wanx model: {self.model}")
                return

        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY or QWEN_API_KEY is required")
            self.model = model or os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
            logger.info(f"ImageGenerator initialized with Gemini model: {self.model}")

    def _get_wanx_size(self, aspect_ratio: str) -> str:
        size_map = {
            "1:1": "1024*1024",
            "16:9": "1280*720",
            "9:16": "720*1280",
            "4:3": "1024*768",
            "3:4": "768*1024",
        }
        return size_map.get(aspect_ratio, "1024*1024")

    async def submit_task(self, prompt: str, aspect_ratio: str = "1:1") -> str:
        """Submit a wanx task and return task_id immediately."""
        size = self._get_wanx_size(aspect_ratio)

        payload = {
            "model": self.model,
            "input": {"prompt": prompt},
            "parameters": {"style": "<auto>", "size": size, "n": 1}
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-Async": "enable",
                },
                json=payload,
            )

            if response.status_code != 200:
                raise Exception(f"Wanx error ({response.status_code}): {response.text}")

            result = response.json()
            task_id = result.get("output", {}).get("task_id")
            if not task_id:
                raise Exception(f"No task_id: {result}")

            logger.info(f"Wanx task submitted: {task_id}")
            return task_id

    async def check_task(self, task_id: str) -> dict:
        """Check status of a wanx task."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.task_url}/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

            if response.status_code != 200:
                raise Exception(f"Poll error ({response.status_code}): {response.text}")

            result = response.json()
            status = result.get("output", {}).get("task_status")

            if status == "SUCCEEDED":
                results = result.get("output", {}).get("results", [])
                image_url = results[0]["url"] if results else None
                return {"status": "SUCCEEDED", "image_url": image_url}
            elif status == "FAILED":
                return {"status": "FAILED", "error": result.get("output", {}).get("message", "")}
            else:
                return {"status": status}

    async def generate_image(self, prompt: str, aspect_ratio: str = "1:1") -> dict:
        """Full generate: submit + poll until done. For Gemini or sync use."""
        if self.provider == "gemini":
            return await self._generate_with_gemini(prompt, aspect_ratio)

        task_id = await self.submit_task(prompt, aspect_ratio)

        max_attempts = 60
        for attempt in range(max_attempts):
            await asyncio.sleep(3)
            result = await self.check_task(task_id)

            if result["status"] == "SUCCEEDED":
                image_url = result["image_url"]
                async with httpx.AsyncClient(timeout=60.0) as client:
                    img_resp = await client.get(image_url)
                    if img_resp.status_code != 200:
                        raise Exception(f"Download failed: {img_resp.status_code}")
                    image_base64 = base64.b64encode(img_resp.content).decode("utf-8")
                    mime_type = "image/png"
                    if image_url.lower().endswith((".jpg", ".jpeg")):
                        mime_type = "image/jpeg"
                    return {"image_base64": image_base64, "mime_type": mime_type, "text": ""}

            elif result["status"] == "FAILED":
                raise Exception(f"Wanx failed: {result.get('error')}")

        raise Exception("Wanx task timed out")

    async def _generate_with_gemini(self, prompt: str, aspect_ratio: str) -> dict:
        url = f"{self.base_url}/models/{self.model}:generateContent"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio}
            }
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                params={"key": self.api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
            )

            if response.status_code != 200:
                raise Exception(f"Gemini error ({response.status_code}): {response.text}")

            result = response.json()
            parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])

            image_data = None
            text_response = ""
            for part in parts:
                if "inlineData" in part:
                    image_data = part["inlineData"]
                elif "text" in part:
                    text_response = part["text"]

            if image_data:
                return {
                    "image_base64": image_data.get("data"),
                    "mime_type": image_data.get("mimeType", "image/png"),
                    "text": text_response,
                }
            raise Exception(f"No image generated: {text_response}")

    async def generate_with_reference(self, reference_description: str, scene_prompt: str, style_hints: Optional[str] = None) -> dict:
        full_prompt = f"Based on this character description:\n{reference_description}\n\nGenerate an image of: {scene_prompt}"
        if style_hints:
            full_prompt += f"\nStyle: {style_hints}"
        return await self.generate_image(full_prompt)
