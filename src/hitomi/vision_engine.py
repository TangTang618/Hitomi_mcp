"""Vision Engine for image analysis.

Supports multiple vision providers: DeepSeek-VL, OpenAI, Qwen, Ollama, etc.
"""

import os
import base64
import httpx
import io
from typing import Optional
from pathlib import Path
from PIL import Image

# Smart compression settings - will try from largest to smallest
IMAGE_SIZES = [1024, 896, 768, 640, 512, 384, 256]  # pixels
MAX_BASE64_SIZE = 800000  # ~800KB initial limit (will reduce on retry)


# Supported providers
PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-vl",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # Singapore (intl)
        "model": "qwen-vl-max",
        "env_key": ["QWEN_API_KEY", "DASHSCOPE_API_KEY"],  # Support both names
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llava",
        "env_key": None,
    },
}


def compress_image(image_data: bytes, max_size: int = 1024, max_b64_size: int = MAX_BASE64_SIZE) -> tuple[bytes, str]:
    """Compress image to reduce size.

    Args:
        image_data: Raw image bytes
        max_size: Maximum width/height in pixels
        max_b64_size: Maximum base64 encoded size in bytes

    Returns:
        Tuple of (compressed_bytes, mime_type)
    """
    img = Image.open(io.BytesIO(image_data))

    # Convert to RGB if necessary (for JPEG compatibility)
    if img.mode in ('RGBA', 'LA', 'P'):
        # Create white background for transparent images
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize if too large
    if img.width > max_size or img.height > max_size:
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Compress with decreasing quality until size is acceptable
    for quality in [85, 70, 50, 30]:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        compressed = buffer.getvalue()

        # Check base64 size
        b64_size = len(base64.b64encode(compressed))
        if b64_size <= max_b64_size:
            return compressed, 'image/jpeg'

    # If still too large, use lowest quality
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=20, optimize=True)
    return buffer.getvalue(), 'image/jpeg'


class VisionEngine:
    """Engine for analyzing images using various vision models."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or os.getenv("VISION_PROVIDER", "deepseek")

        if self.provider not in PROVIDERS and self.provider != "custom":
            raise ValueError(f"Unknown provider: {self.provider}. Supported: {list(PROVIDERS.keys())}")

        if self.provider == "custom":
            self.base_url = base_url or os.getenv("CUSTOM_BASE_URL")
            self.model = model or os.getenv("CUSTOM_VISION_MODEL")
            self.api_key = api_key or os.getenv("CUSTOM_API_KEY")
            if not self.base_url:
                raise ValueError("CUSTOM_BASE_URL is required for custom provider")
        else:
            config = PROVIDERS[self.provider]
            self.base_url = base_url or os.getenv(f"{self.provider.upper()}_BASE_URL", config["base_url"])
            self.model = model or os.getenv(f"{self.provider.upper()}_VISION_MODEL", config["model"])

            env_keys = config["env_key"]
            if env_keys:
                # Support both single key and list of keys
                if isinstance(env_keys, list):
                    self.api_key = api_key
                    for key_name in env_keys:
                        if not self.api_key:
                            self.api_key = os.getenv(key_name)
                    if not self.api_key:
                        raise ValueError(f"One of {env_keys} is required")
                else:
                    self.api_key = api_key or os.getenv(env_keys)
                    if not self.api_key:
                        raise ValueError(f"{env_keys} is required")
            else:
                self.api_key = None

    async def _fetch_image_bytes(self, image_input: str) -> Optional[bytes]:
        """Fetch raw image bytes from various sources."""
        # Already a data URL - decode it
        if image_input.startswith("data:image/"):
            try:
                _, data_part = image_input.split(",", 1)
                return base64.b64decode(data_part)
            except Exception:
                return None

        # URL - fetch image
        elif image_input.startswith("http://") or image_input.startswith("https://"):
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(image_input)
                response.raise_for_status()
                return response.content

        # Local file path
        elif image_input.startswith("/") or image_input.startswith("~"):
            path = Path(image_input).expanduser()
            if path.exists():
                return path.read_bytes()

        # Assume raw base64
        else:
            try:
                return base64.b64decode(image_input)
            except Exception:
                return None

        return None

    def _compress_to_base64(self, image_bytes: bytes, max_size: int = 1024) -> str:
        """Compress image bytes and return base64 data URL."""
        compressed_bytes, mime_type = compress_image(image_bytes, max_size=max_size)
        b64 = base64.b64encode(compressed_bytes).decode()
        return f"data:{mime_type};base64,{b64}"

    async def _image_to_base64(self, image_input: str, max_size: int = 1024) -> str:
        """Convert image input to base64 data URL with compression."""
        image_bytes = await self._fetch_image_bytes(image_input)

        if image_bytes:
            return self._compress_to_base64(image_bytes, max_size)

        return image_input

    async def analyze_image(
        self,
        image: str,
        question: Optional[str] = None,
        detail: str = "auto",
    ) -> str:
        """Analyze a single image.

        Args:
            image: Image URL, base64 data, or file path
            question: Optional question about the image
            detail: Detail level - "low", "high", or "auto"

        Returns:
            Text description/analysis of the image
        """
        if self.provider == "ollama":
            return await self._analyze_with_ollama(image, question)
        else:
            return await self._analyze_with_openai_compatible(image, question, detail)

    async def analyze_images(
        self,
        images: list[str],
        question: Optional[str] = None,
        detail: str = "auto",
    ) -> str:
        """Analyze multiple images (for comparison).

        Args:
            images: List of image URLs, base64 data, or file paths
            question: Optional question about the images
            detail: Detail level

        Returns:
            Text analysis comparing the images
        """
        if self.provider == "ollama":
            return await self._analyze_with_ollama(images[0], question, images[1:])
        else:
            return await self._analyze_with_openai_compatible_multi(images, question, detail)

    async def _analyze_with_openai_compatible(
        self,
        image: str,
        question: Optional[str],
        detail: str,
    ) -> str:
        """Analyze using OpenAI-compatible API with smart compression retry."""
        # First, fetch the image bytes once
        image_bytes = await self._fetch_image_bytes(image)
        if not image_bytes:
            raise Exception(f"Could not fetch image: {image[:100]}...")

        prompt = question or "请详细描述这张图片的内容，包括：主要元素、场景、颜色、文字（如有）、以及任何值得注意的细节。"

        # Try different sizes from largest to smallest
        last_error = None
        for size in IMAGE_SIZES:
            image_data = self._compress_to_base64(image_bytes, max_size=size)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data,
                                "detail": "high" if detail == "high" else "low" if detail == "low" else "auto",
                            },
                        },
                    ],
                }
            ]

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 4096,
                    },
                )

                # Success!
                if response.status_code == 200:
                    result = response.json()
                    # Log successful size for debugging
                    import logging
                    logging.info(f"Smart compression: succeeded with {size}px")
                    return result["choices"][0]["message"]["content"]

                # Token limit error - try smaller size
                error_text = response.text
                if "maximum context length" in error_text or "token" in error_text.lower():
                    import logging
                    logging.warning(f"Smart compression: {size}px too large, trying smaller...")
                    last_error = error_text
                    continue

                # Other error - don't retry
                raise Exception(f"API error ({response.status_code}): {error_text}")

        # All sizes failed
        raise Exception(f"Image too large even at minimum size. Last error: {last_error}")

    async def _analyze_with_openai_compatible_multi(
        self,
        images: list[str],
        question: Optional[str],
        detail: str,
    ) -> str:
        """Analyze multiple images using OpenAI-compatible API with smart compression."""
        # First, fetch all image bytes
        images_bytes = []
        for image in images:
            image_bytes = await self._fetch_image_bytes(image)
            if not image_bytes:
                raise Exception(f"Could not fetch image: {image[:100]}...")
            images_bytes.append(image_bytes)

        prompt = question or "请详细比较这些图片的异同点。"

        # For multiple images, start with smaller sizes (divide by number of images)
        # to stay within token limits
        multi_sizes = [s for s in IMAGE_SIZES if s <= 768]  # Start smaller for multi

        last_error = None
        for size in multi_sizes:
            content = [{"type": "text", "text": prompt}]

            for img_bytes in images_bytes:
                image_data = self._compress_to_base64(img_bytes, max_size=size)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data,
                        "detail": "high" if detail == "high" else "low" if detail == "low" else "auto",
                    },
                })

            messages = [{"role": "user", "content": content}]

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 4096,
                    },
                )

                # Success!
                if response.status_code == 200:
                    result = response.json()
                    import logging
                    logging.info(f"Smart compression (multi): succeeded with {size}px")
                    return result["choices"][0]["message"]["content"]

                # Token limit error - try smaller size
                error_text = response.text
                if "maximum context length" in error_text or "token" in error_text.lower():
                    import logging
                    logging.warning(f"Smart compression (multi): {size}px too large, trying smaller...")
                    last_error = error_text
                    continue

                # Other error - don't retry
                raise Exception(f"API error ({response.status_code}): {error_text}")

        # All sizes failed
        raise Exception(f"Images too large even at minimum size. Last error: {last_error}")

    async def _analyze_with_ollama(
        self,
        image: str,
        question: Optional[str],
        extra_images: Optional[list[str]] = None,
    ) -> str:
        """Analyze using Ollama local model."""
        images_b64 = []

        # Process main image
        image_data = await self._image_to_base64(image)
        # Extract pure base64 (remove data URL prefix)
        if "base64," in image_data:
            images_b64.append(image_data.split("base64,")[1])
        else:
            images_b64.append(image_data)

        # Process extra images if any
        if extra_images:
            for img in extra_images:
                img_data = await self._image_to_base64(img)
                if "base64," in img_data:
                    images_b64.append(img_data.split("base64,")[1])
                else:
                    images_b64.append(img_data)

        prompt = question or "Describe this image in detail, including main elements, scene, colors, text (if any), and notable details."

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": images_b64,
                    "stream": False,
                },
            )

            if response.status_code != 200:
                raise Exception(f"Ollama error ({response.status_code}): {response.text}")

            result = response.json()
            return result["response"]
