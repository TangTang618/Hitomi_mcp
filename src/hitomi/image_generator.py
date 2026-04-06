"""Image Generator supporting multiple providers.

Supports image generation from text prompts using Gemini or Qwen (Wanx).
"""

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
        # Determine provider: qwen or gemini
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
                raise ValueError("GEMINI_API_KEY or QWEN_API_KEY is required for image generation")
            self.model = model or os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
            logger.info(f"ImageGenerator initialized with Gemini model: {self.model}")

    def _get_wanx_size(self, aspect_ratio: str) -> str:
        """Convert aspect ratio to Wanx supported size."""
        size_map = {
            "1:1": "1024*1024",
            "16:9": "1280*720",
            "9:16": "720*1280",
            "4:3": "1024*768",
            "3:4": "768*1024",
        }
        return size_map.get(aspect_ratio, "1024*1024")

    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
    ) -> dict:
        """Generate an image from a text prompt."""
        if self.provider == "qwen":
            return await self._generate_with_qwen(prompt, aspect_ratio)
        else:
            return await self._generate_with_gemini(prompt, aspect_ratio)

    async def _generate_with_qwen(self, prompt: str, aspect_ratio: str) -> dict:
        """Generate image using Qwen Wanx via async task API."""
        size = self._get_wanx_size(aspect_ratio)

        # Step 1: Create task
        payload = {
            "model": self.model,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "style": "<auto>",
                "size": size,
                "n": 1
            }
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
                raise Exception(f"Wanx create task error ({response.status_code}): {response.text}")

            result = response.json()
            task_id = result.get("output", {}).get("task_id")
            if not task_id:
                raise Exception(f"No task_id in response: {result}")

            logger.info(f"Wanx task created: {task_id}")

        # Step 2: Poll for result
        task_url = f"{self.task_url}/{task_id}"
        max_attempts = 60
        poll_interval = 3

        for attempt in range(max_attempts):
            await asyncio.sleep(poll_interval)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    task_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )

                if response.status_code != 200:
                    raise Exception(f"Wanx poll error ({response.status_code}): {response.text}")

                result = response.json()
                status = result.get("output", {}).get("task_status")
                logger.info(f"Wanx task {task_id} status: {status} (attempt {attempt + 1})")

                if status == "SUCCEEDED":
                    results = result.get("output", {}).get("results", [])
                    if not results or "url" not in results[0]:
                        raise Exception(f"No image URL in results: {result}")

                    image_url = results[0]["url"]
                    logger.info(f"Wanx image generated: {image_url}")

                    # Step 3: Download image and convert to base64
                    async with httpx.AsyncClient(timeout=60.0) as dl_client:
                        img_response = await dl_client.get(image_url)
                        if img_response.status_code != 200:
                            raise Exception(f"Failed to download image: {img_response.status_code}")

                        image_base64 = base64.b64encode(img_response.content).decode("utf-8")

                        # Detect mime type from URL
                        mime_type = "image/png"
                        if image_url.lower().endswith(".jpg") or image_url.lower().endswith(".jpeg"):
                            mime_type = "image/jpeg"

                        return {
                            "image_base64": image_base64,
                            "mime_type": mime_type,
                            "text": "",
                        }

                elif status == "FAILED":
                    error_code = result.get("output", {}).get("code", "")
                    error_msg = result.get("output", {}).get("message", "")
                    raise Exception(f"Wanx task failed: {error_code} - {error_msg}")

                elif status in ("PENDING", "RUNNING"):
                    continue
                else:
                    raise Exception(f"Unknown task status: {status}")

        raise Exception(f"Wanx task timed out after {max_attempts * poll_interval} seconds")

    async def _generate_with_gemini(self, prompt: str, aspect_ratio: str) -> dict:
        """Generate image using Gemini API."""
        url = f"{self.base_url}/models/{self.model}:generateContent"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio
                }
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
                error_text = response.text
                logger.error(f"Gemini API error: {error_text}")
                raise Exception(f"Gemini API error ({response.status_code}): {error_text}")

            result = response.json()

            try:
                candidates = result.get("candidates", [])
                if not candidates:
                    raise Exception("No candidates in response")

                parts = candidates[0].get("content", {}).get("parts", [])

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
                else:
                    raise Exception(f"No image generated. Model response: {text_response}")

            except KeyError as e:
                logger.error(f"Unexpected response format: {result}")
                raise Exception(f"Unexpected response format: {e}")

    async def generate_with_reference(
        self,
        reference_description: str,
        scene_prompt: str,
        style_hints: Optional[str] = None,
    ) -> dict:
        """Generate an image based on a character/reference description and a new scene."""
        full_prompt = f"""Based on this character description:
{reference_description}

Generate an image of: {scene_prompt}
"""
        if style_hints:
            full_prompt += f"\nStyle: {style_hints}"

        return await self.generate_image(full_prompt)
