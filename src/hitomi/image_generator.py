"""Image Generator supporting multiple providers.

Supports image generation from text prompts using Gemini or Qwen (Wanx).
"""

import os
import base64
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Engine for generating images using Gemini or Qwen."""

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
                # Fallback to Gemini if no Qwen key
                logger.warning("No QWEN_API_KEY found, trying Gemini...")
                self.provider = "gemini"
            else:
                # Qwen image models: qwen-image-max, qwen-image-plus
                self.model = model or os.getenv("QWEN_IMAGE_MODEL", "qwen-image-max")
                # Use chat/completions API (like GPT-4o)
                self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
                logger.info(f"ImageGenerator initialized with Qwen model: {self.model}")
                return

        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY or QWEN_API_KEY is required for image generation")
            # Use gemini-3-pro-image-preview (Nano Banana Pro) for best quality
            self.model = model or os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
            logger.info(f"ImageGenerator initialized with Gemini model: {self.model}")

    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
    ) -> dict:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)

        Returns:
            Dict with 'image_base64' and 'mime_type'
        """
        if self.provider == "qwen":
            return await self._generate_with_qwen(prompt, aspect_ratio)
        else:
            return await self._generate_with_gemini(prompt, aspect_ratio)

    async def _generate_with_qwen(self, prompt: str, aspect_ratio: str) -> dict:
        """Generate image using Qwen multimodal model via chat/completions."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"请根据以下描述生成一张图片：\n\n{prompt}\n\n要求：生成高质量的图片，符合描述的内容和风格。"
                }
            ],
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            response_text = response.text
            logger.info(f"Qwen response status: {response.status_code}")
            logger.info(f"Qwen response: {response_text[:500]}")

            if response.status_code != 200:
                logger.error(f"Qwen API error: {response_text}")
                raise Exception(f"Qwen API error ({response.status_code}): {response_text}")

            result = response.json()

            # Check if model returned image content
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")

                # If content is a list (multimodal response)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            image_data = item.get("image", "")
                            if image_data:
                                return {
                                    "image_base64": image_data,
                                    "mime_type": "image/png",
                                    "text": "",
                                }

                # Model doesn't support image generation, return error with helpful message
                raise Exception(f"该模型不支持图片生成。模型返回: {str(content)[:200]}")

            raise Exception(f"API响应格式错误: {result}")

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

            # Extract image from response
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
        """Generate an image based on a character/reference description and a new scene.

        Args:
            reference_description: Description of the character/reference (from vision analysis)
            scene_prompt: The new scene to generate
            style_hints: Optional style hints (art style, mood, etc.)

        Returns:
            Dict with 'image_base64' and 'mime_type'
        """
        # Combine reference description with scene prompt
        full_prompt = f"""Based on this character description:
{reference_description}

Generate an image of: {scene_prompt}
"""
        if style_hints:
            full_prompt += f"\nStyle: {style_hints}"

        return await self.generate_image(full_prompt)
