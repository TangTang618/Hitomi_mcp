"""Hitomi MCP Server.

Provides SSE-based MCP server with vision and image generation tools.
"""

import os
import uuid
import time
import base64
import logging
from typing import Optional, Dict

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ImageContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse, Response, HTMLResponse
from starlette.requests import Request
import uvicorn

from .vision_engine import VisionEngine
from .image_generator import ImageGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp_server = Server("hitomi")

# Lazy-initialized engines
_engine: Optional[VisionEngine] = None
_generator: Optional[ImageGenerator] = None

# Image storage (in-memory, with expiration)
_image_store: Dict[str, dict] = {}
IMAGE_EXPIRY_SECONDS = 3600  # 1 hour


def get_engine() -> VisionEngine:
    """Get or create the Vision engine instance."""
    global _engine
    if _engine is None:
        _engine = VisionEngine()
    return _engine


def get_generator() -> Optional[ImageGenerator]:
    """Get or create the Image generator instance."""
    global _generator
    if _generator is None:
        try:
            _generator = ImageGenerator()
        except ValueError as e:
            logger.warning(f"Image generator not available: {e}")
            return None
    return _generator


def get_base_url() -> str:
    """Get the base URL for image endpoints."""
    domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
    if domain:
        return f"https://{domain}"
    port = os.getenv("PORT", "8000")
    return f"http://localhost:{port}"


def store_image(image_base64: str, mime_type: str) -> str:
    """Store image and return its ID."""
    image_id = str(uuid.uuid4())
    _image_store[image_id] = {
        "data": image_base64,
        "mime_type": mime_type,
        "created_at": time.time()
    }
    cleanup_expired_images()
    return image_id


def cleanup_expired_images():
    """Remove expired images from store."""
    now = time.time()
    expired = [
        img_id for img_id, img in _image_store.items()
        if now - img["created_at"] > IMAGE_EXPIRY_SECONDS
    ]
    for img_id in expired:
        del _image_store[img_id]


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available vision tools."""
    return [
        Tool(
            name="get_upload_url",
            description="获取图片上传页面的网址。当用户想要上传图片、需要图片链接、或者询问如何发送图片时，调用此工具获取上传页面地址。",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="analyze_image",
            description="分析图片内容并返回详细描述。支持识别图片中的物体、文字、场景等。当用户发送图片或询问图片相关问题时使用此工具。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "图片的 URL 地址或 base64 编码的图片数据"
                    },
                    "question": {
                        "type": "string",
                        "description": "关于图片的具体问题（可选）。如果不提供，将返回图片的通用描述。"
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["low", "high", "auto"],
                        "default": "auto",
                        "description": "分析精度：low=快速概览，high=详细分析，auto=自动选择"
                    }
                },
                "required": ["image"]
            }
        ),
        Tool(
            name="extract_text",
            description="从图片中提取文字内容（OCR）。适用于截图、文档、照片中的文字识别。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "图片的 URL 地址或 base64 编码的图片数据"
                    },
                    "language": {
                        "type": "string",
                        "default": "auto",
                        "description": "预期的文字语言（如 'zh' 中文，'en' 英文，'auto' 自动检测）"
                    }
                },
                "required": ["image"]
            }
        ),
        Tool(
            name="compare_images",
            description="比较两张图片的异同，找出差异或相似之处。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image1": {
                        "type": "string",
                        "description": "第一张图片的 URL 或 base64 数据"
                    },
                    "image2": {
                        "type": "string",
                        "description": "第二张图片的 URL 或 base64 数据"
                    },
                    "focus": {
                        "type": "string",
                        "enum": ["differences", "similarities", "all"],
                        "default": "all",
                        "description": "比较的重点"
                    }
                },
                "required": ["image1", "image2"]
            }
        ),
        Tool(
            name="generate_image",
            description="根据文字描述生成图片。使用 Gemini 模型进行图片生成。",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "图片描述，详细说明想要生成的图片内容、风格、场景等"
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4"],
                        "default": "1:1",
                        "description": "图片宽高比"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="transform_image",
            description="根据参考图片生成新场景。先分析参考图片中的角色/物体特征，然后根据新的场景描述生成新图片。适用于：用人设图画新场景、角色转换场景等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "reference_image": {
                        "type": "string",
                        "description": "参考图片的 URL 或 base64 数据（如人设图）"
                    },
                    "scene_prompt": {
                        "type": "string",
                        "description": "新场景描述，如：'坐在沙发上喝咖啡'、'在海边玩耍'"
                    },
                    "style_hints": {
                        "type": "string",
                        "description": "可选的风格提示，如：'保持原画风'、'写实风格'、'动漫风格'"
                    },
                    "focus_on": {
                        "type": "string",
                        "description": "可选，指定要关注的特征，如：'角色外貌和服装'、'整体画风'"
                    }
                },
                "required": ["reference_image", "scene_prompt"]
            }
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls for image analysis."""
    engine = get_engine()

    try:
        if name == "get_upload_url":
            base_url = get_base_url()
            upload_url = f"{base_url}/upload-page"
            response_text = f"""📷 图片上传页面

请访问以下网址上传图片：
{upload_url}

使用步骤：
1. 打开上面的链接
2. 选择要上传的图片
3. 点击"上传"按钮
4. 复制生成的图片链接
5. 将链接发送给我进行分析或处理

上传的图片链接有效期为 1 小时。"""
            return [TextContent(type="text", text=response_text)]

        elif name == "analyze_image":
            logger.info(f"Analyzing image...")
            result = await engine.analyze_image(
                image=arguments["image"],
                question=arguments.get("question"),
                detail=arguments.get("detail", "auto"),
            )
            return [TextContent(type="text", text=result)]

        elif name == "extract_text":
            logger.info(f"Extracting text from image...")
            result = await engine.analyze_image(
                image=arguments["image"],
                question="请提取并返回图片中的所有文字内容，保持原有的格式和布局。",
                detail="high",
            )
            return [TextContent(type="text", text=result)]

        elif name == "compare_images":
            logger.info(f"Comparing images...")
            focus = arguments.get("focus", "all")
            if focus == "differences":
                question = "请比较这两张图片，重点关注它们之间的差异。"
            elif focus == "similarities":
                question = "请比较这两张图片，重点关注它们的相似之处。"
            else:
                question = "请详细比较这两张图片的异同点。"

            result = await engine.analyze_images(
                images=[arguments["image1"], arguments["image2"]],
                question=question,
                detail="high",
            )
            return [TextContent(type="text", text=result)]

        elif name == "generate_image":
            generator = get_generator()
            if not generator:
                return [TextContent(type="text", text="错误：图片生成功能不可用，请配置 QWEN_API_KEY 或 GEMINI_API_KEY")]

            logger.info(f"Generating image: {arguments['prompt'][:50]}...")
            result = await generator.generate_image(
                prompt=arguments["prompt"],
                aspect_ratio=arguments.get("aspect_ratio", "1:1"),
            )

            # Store the generated image and return URL
            image_id = store_image(result["image_base64"], result["mime_type"])
            base_url = get_base_url()
            image_url = f"{base_url}/images/{image_id}"

            response_text = f"""🎨 图片已生成！

📎 备用链接：{image_url}
⏰ 链接有效期：1小时"""
            if result.get("text"):
                response_text += f"\n\n💬 模型说明：{result['text']}"

            # Return both image and text so Kelivo can render the image directly
            return [
                ImageContent(type="image", data=result["image_base64"], mimeType=result["mime_type"]),
                TextContent(type="text", text=response_text)
            ]

        elif name == "transform_image":
            generator = get_generator()
            if not generator:
                return [TextContent(type="text", text="错误：图片生成功能不可用，请配置 QWEN_API_KEY 或 GEMINI_API_KEY")]

            logger.info(f"Transforming image to new scene: {arguments['scene_prompt'][:50]}...")

            # Step 1: Analyze the reference image
            focus_on = arguments.get("focus_on", "角色的外貌特征、服装、发型、整体画风")
            analysis_prompt = f"请详细描述这张图片中的角色/主体，重点关注：{focus_on}。描述需要足够详细，以便用于生成新图片。"

            reference_description = await engine.analyze_image(
                image=arguments["reference_image"],
                question=analysis_prompt,
                detail="high",
            )

            logger.info(f"Reference analyzed, generating new scene...")

            # Step 2: Generate new image with reference description
            result = await generator.generate_with_reference(
                reference_description=reference_description,
                scene_prompt=arguments["scene_prompt"],
                style_hints=arguments.get("style_hints"),
            )

            # Store the generated image and return URL
            image_id = store_image(result["image_base64"], result["mime_type"])
            base_url = get_base_url()
            image_url = f"{base_url}/images/{image_id}"

            response_text = f"""🎨 新场景图片已生成！

📎 备用链接：{image_url}
⏰ 链接有效期：1小时

📝 参考图分析：
{reference_description[:300]}..."""
            if result.get("text"):
                response_text += f"\n\n💬 生成说明：{result['text']}"

            # Return both image and text so Kelivo can render the image directly
            return [
                ImageContent(type="image", data=result["image_base64"], mimeType=result["mime_type"]),
                TextContent(type="text", text=response_text)
            ]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# SSE Transport handler
sse_transport = SseServerTransport("/messages/")


# Create raw ASGI handlers for SSE (they handle responses internally)
async def handle_sse_asgi(scope, receive, send):
    """Handle SSE connection for MCP (raw ASGI)."""
    logger.info("New SSE connection")
    async with sse_transport.connect_sse(scope, receive, send) as streams:
        await mcp_server.run(
            streams[0], streams[1], mcp_server.create_initialization_options()
        )


async def handle_messages_asgi(scope, receive, send):
    """Handle POST messages for SSE transport (raw ASGI)."""
    logger.info("Received message")
    await sse_transport.handle_post_message(scope, receive, send)


async def handle_messages_asgi(scope, receive, send):
    """Handle POST messages for SSE transport (raw ASGI)."""
    logger.info("Received message")
    await sse_transport.handle_post_message(scope, receive, send)


async def health_check(request):
    """Health check endpoint."""
    provider = os.getenv("VISION_PROVIDER", "deepseek")
    return JSONResponse({
        "status": "healthy",
        "service": "hitomi",
        "version": "1.0.0",
        "provider": provider,
    })


async def index(request):
    """Index endpoint with server info."""
    base_url = get_base_url()
    return JSONResponse({
        "name": "Hitomi",
        "version": "1.0.0",
        "description": "Vision & Image Generation MCP Server",
        "endpoints": {
            "sse": "/sse",
            "messages": "/messages/",
            "health": "/health",
            "upload": "/upload",
            "upload_page": "/upload-page",
        },
        "tools": ["get_upload_url", "analyze_image", "extract_text", "compare_images", "generate_image", "transform_image"],
        "upload_url": f"{base_url}/upload-page",
    })


async def upload_page(request):
    """Simple HTML page for uploading images."""
    base_url = get_base_url()
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>上传图片 - Hitomi</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .upload-area {{ border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }}
            .upload-area:hover {{ border-color: #666; }}
            input[type="file"] {{ margin: 10px 0; }}
            button {{ background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }}
            button:hover {{ background: #0056b3; }}
            .result {{ margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; word-break: break-all; }}
            .result a {{ color: #007bff; }}
            .preview {{ max-width: 300px; margin: 10px 0; }}
            .copy-btn {{ background: #28a745; margin-left: 10px; }}
        </style>
    </head>
    <body>
        <h1>📷 上传图片</h1>
        <p>上传图片后获取 URL，然后发给 DeepSeek 分析。</p>

        <div class="upload-area">
            <input type="file" id="imageInput" accept="image/*">
            <br><br>
            <button onclick="uploadImage()">上传</button>
        </div>

        <div id="result" class="result" style="display:none;">
            <strong>图片 URL：</strong><br>
            <a id="imageUrl" href="#" target="_blank"></a>
            <button class="copy-btn" onclick="copyUrl()">复制</button>
            <br><br>
            <strong>发给 DeepSeek：</strong><br>
            <code id="prompt"></code>
            <button class="copy-btn" onclick="copyPrompt()">复制</button>
            <br><br>
            <img id="preview" class="preview">
        </div>

        <script>
            async function uploadImage() {{
                const input = document.getElementById('imageInput');
                if (!input.files[0]) {{
                    alert('请选择图片');
                    return;
                }}

                const file = input.files[0];
                const reader = new FileReader();

                reader.onload = async function(e) {{
                    const base64 = e.target.result.split(',')[1];
                    const mimeType = file.type;

                    try {{
                        const response = await fetch('/upload', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ image_base64: base64, mime_type: mimeType }})
                        }});

                        const data = await response.json();

                        if (data.url) {{
                            document.getElementById('result').style.display = 'block';
                            document.getElementById('imageUrl').href = data.url;
                            document.getElementById('imageUrl').textContent = data.url;
                            document.getElementById('prompt').textContent = '请分析这张图片：' + data.url;
                            document.getElementById('preview').src = data.url;
                        }} else {{
                            alert('上传失败：' + (data.error || '未知错误'));
                        }}
                    }} catch (err) {{
                        alert('上传失败：' + err.message);
                    }}
                }};

                reader.readAsDataURL(file);
            }}

            function copyUrl() {{
                navigator.clipboard.writeText(document.getElementById('imageUrl').textContent);
                alert('已复制 URL');
            }}

            function copyPrompt() {{
                navigator.clipboard.writeText(document.getElementById('prompt').textContent);
                alert('已复制提示词');
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


async def upload_image(request: Request):
    """Handle image upload and return URL."""
    try:
        body = await request.json()
        image_base64 = body.get("image_base64")
        mime_type = body.get("mime_type", "image/png")

        if not image_base64:
            return JSONResponse({"error": "Missing image_base64"}, status_code=400)

        image_id = store_image(image_base64, mime_type)
        base_url = get_base_url()
        image_url = f"{base_url}/images/{image_id}"

        logger.info(f"Image uploaded: {image_id}")

        return JSONResponse({
            "success": True,
            "image_id": image_id,
            "url": image_url,
            "expires_in": IMAGE_EXPIRY_SECONDS,
        })
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_image(request: Request):
    """Serve stored images."""
    image_id = request.path_params["image_id"]

    if image_id not in _image_store:
        return JSONResponse({"error": "Image not found or expired"}, status_code=404)

    image_data = _image_store[image_id]
    image_bytes = base64.b64decode(image_data["data"])

    return Response(
        content=image_bytes,
        media_type=image_data["mime_type"]
    )


# Create Starlette app for non-SSE routes
starlette_app = Starlette(
    routes=[
        Route("/", index, methods=["GET"]),
        Route("/health", health_check, methods=["GET"]),
        Route("/upload", upload_image, methods=["POST"]),
        Route("/upload-page", upload_page, methods=["GET"]),
        Route("/images/{image_id}", get_image, methods=["GET"]),
    ]
)


# Custom ASGI app that handles SSE routes specially
async def app(scope, receive, send):
    """Main ASGI app with SSE handling."""
    if scope["type"] == "http":
        path = scope.get("path", "")

        # Handle SSE endpoint
        if path == "/sse":
            await handle_sse_asgi(scope, receive, send)
            return

        # Handle messages endpoint (with or without trailing slash)
        if path.startswith("/messages"):
            await handle_messages_asgi(scope, receive, send)
            return

    # All other routes go to Starlette
    await starlette_app(scope, receive, send)


def main():
    """Run the MCP server."""
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting Hitomi MCP Server on {host}:{port}")
    logger.info(f"Provider: {os.getenv('VISION_PROVIDER', 'deepseek')}")
    logger.info("SSE endpoint: /sse")
    logger.info("Messages endpoint: /messages/")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
