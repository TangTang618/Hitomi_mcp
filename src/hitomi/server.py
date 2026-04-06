"""Hitomi MCP Server."""

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
from starlette.responses import JSONResponse, Response, HTMLResponse, RedirectResponse
from starlette.requests import Request
import uvicorn
import httpx as httpx_lib

from .vision_engine import VisionEngine
from .image_generator import ImageGenerator

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp_server = Server("hitomi")

_engine: Optional[VisionEngine] = None
_generator: Optional[ImageGenerator] = None
_image_store: Dict[str, dict] = {}
IMAGE_EXPIRY_SECONDS = 3600


def get_engine() -> VisionEngine:
    global _engine
    if _engine is None:
        _engine = VisionEngine()
    return _engine


def get_generator() -> Optional[ImageGenerator]:
    global _generator
    if _generator is None:
        try:
            _generator = ImageGenerator()
        except ValueError as e:
            logger.warning(f"Image generator not available: {e}")
            return None
    return _generator


def get_base_url() -> str:
    domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
    if domain:
        return f"https://{domain}"
    port = os.getenv("PORT", "8000")
    return f"http://localhost:{port}"


def store_image(image_base64: str, mime_type: str) -> str:
    image_id = str(uuid.uuid4())
    _image_store[image_id] = {
        "data": image_base64,
        "mime_type": mime_type,
        "created_at": time.time()
    }
    cleanup_expired_images()
    return image_id


def cleanup_expired_images():
    now = time.time()
    expired = [k for k, v in _image_store.items() if now - v["created_at"] > IMAGE_EXPIRY_SECONDS]
    for k in expired:
        del _image_store[k]


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
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
            description="分析图片内容并返回详细描述。支持识别图片中的物体、文字、场景等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "图片的 URL 地址或 base64 编码的图片数据"},
                    "question": {"type": "string", "description": "关于图片的具体问题（可选）"},
                    "detail": {"type": "string", "enum": ["low", "high", "auto"], "default": "auto", "description": "分析精度"}
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
                    "image": {"type": "string", "description": "图片的 URL 地址或 base64 编码的图片数据"},
                    "language": {"type": "string", "default": "auto", "description": "预期的文字语言"}
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
                    "image1": {"type": "string", "description": "第一张图片的 URL 或 base64 数据"},
                    "image2": {"type": "string", "description": "第二张图片的 URL 或 base64 数据"},
                    "focus": {"type": "string", "enum": ["differences", "similarities", "all"], "default": "all", "description": "比较的重点"}
                },
                "required": ["image1", "image2"]
            }
        ),
        Tool(
            name="generate_image",
            description="根据文字描述生成图片。提交后返回查看链接，点击即可查看生成结果。",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "图片描述，详细说明想要生成的图片内容、风格、场景等"},
                    "aspect_ratio": {"type": "string", "enum": ["1:1", "16:9", "9:16", "4:3", "3:4"], "default": "1:1", "description": "图片宽高比"}
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="transform_image",
            description="根据参考图片生成新场景。先分析参考图片中的角色/物体特征，然后根据新的场景描述生成新图片。",
            inputSchema={
                "type": "object",
                "properties": {
                    "reference_image": {"type": "string", "description": "参考图片的 URL 或 base64 数据"},
                    "scene_prompt": {"type": "string", "description": "新场景描述"},
                    "style_hints": {"type": "string", "description": "可选的风格提示"},
                    "focus_on": {"type": "string", "description": "可选，指定要关注的特征"}
                },
                "required": ["reference_image", "scene_prompt"]
            }
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    engine = get_engine()

    try:
        if name == "get_upload_url":
            base_url = get_base_url()
            upload_url = f"{base_url}/upload-page"
            return [TextContent(type="text", text=f"📷 图片上传页面\n\n请访问以下网址上传图片：\n{upload_url}\n\n使用步骤：\n1. 打开上面的链接\n2. 选择要上传的图片\n3. 点击上传按钮\n4. 复制生成的图片链接\n5. 将链接发送给我进行分析")]

        elif name == "analyze_image":
            logger.info("Analyzing image...")
            result = await engine.analyze_image(
                image=arguments["image"],
                question=arguments.get("question"),
                detail=arguments.get("detail", "auto"),
            )
            return [TextContent(type="text", text=result)]

        elif name == "extract_text":
            logger.info("Extracting text from image...")
            result = await engine.analyze_image(
                image=arguments["image"],
                question="请提取并返回图片中的所有文字内容，保持原有的格式和布局。",
                detail="high",
            )
            return [TextContent(type="text", text=result)]

        elif name == "compare_images":
            logger.info("Comparing images...")
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

            task_id = await generator.submit_task(
                prompt=arguments["prompt"],
                aspect_ratio=arguments.get("aspect_ratio", "1:1"),
            )

            base_url = get_base_url()
            poll_url = f"{base_url}/tasks/{task_id}"

            response_text = f"🎨 图片生成任务已提交！\n\n⏳ 任务ID：{task_id}\n🔗 查看结果：{poll_url}\n\n图片生成需要20-30秒，请稍后点击链接查看结果。"
            return [TextContent(type="text", text=response_text)]

        elif name == "transform_image":
            generator = get_generator()
            if not generator:
                return [TextContent(type="text", text="错误：图片生成功能不可用，请配置 QWEN_API_KEY 或 GEMINI_API_KEY")]

            logger.info(f"Transforming image to new scene: {arguments['scene_prompt'][:50]}...")

            focus_on = arguments.get("focus_on", "角色的外貌特征、服装、发型、整体画风")
            analysis_prompt = f"请详细描述这张图片中的角色/主体，重点关注：{focus_on}。描述需要足够详细，以便用于生成新图片。"
            reference_description = await engine.analyze_image(
                image=arguments["reference_image"],
                question=analysis_prompt,
                detail="high",
            )

            logger.info("Reference analyzed, submitting generation task...")

            full_prompt = f"Based on this character description:\n{reference_description}\n\nGenerate an image of: {arguments['scene_prompt']}"
            if arguments.get("style_hints"):
                full_prompt += f"\nStyle: {arguments['style_hints']}"

            task_id = await generator.submit_task(prompt=full_prompt)

            base_url = get_base_url()
            poll_url = f"{base_url}/tasks/{task_id}"

            response_text = f"🎨 新场景图片生成任务已提交！\n\n⏳ 任务ID：{task_id}\n🔗 查看结果：{poll_url}\n\n📝 参考图分析：\n{reference_description[:300]}...\n\n图片生成需要20-30秒，请稍后点击链接查看结果。"
            return [TextContent(type="text", text=response_text)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# SSE Transport
sse_transport = SseServerTransport("/messages/")


async def handle_sse_asgi(scope, receive, send):
    logger.info("New SSE connection")
    async with sse_transport.connect_sse(scope, receive, send) as streams:
        await mcp_server.run(
            streams[0], streams[1],
            mcp_server.create_initialization_options()
        )


async def handle_messages_asgi(scope, receive, send):
    logger.info("Received message")
    await sse_transport.handle_post_message(scope, receive, send)


async def health_check(request):
    provider = os.getenv("IMAGE_PROVIDER", "qwen")
    return JSONResponse({
        "status": "healthy",
        "service": "hitomi",
        "version": "1.0.0",
        "provider": provider,
    })


async def index(request):
    base_url = get_base_url()
    return JSONResponse({
        "name": "Hitomi",
        "version": "1.0.0",
        "description": "Vision & Image Generation MCP Server",
        "endpoints": {"sse": "/sse", "messages": "/messages/", "health": "/health", "upload": "/upload", "upload_page": "/upload-page"},
        "tools": ["get_upload_url", "analyze_image", "extract_text", "compare_images", "generate_image", "transform_image"],
        "upload_url": f"{base_url}/upload-page",
    })


async def upload_page(request):
    base_url = get_base_url()
    html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Upload - Hitomi</title></head>
<body>
<h1>📷 上传图片</h1>
<p>上传图片后获取 URL，然后发给 AI 分析。</p>
<input type="file" id="file" accept="image/*">
<button onclick="upload()">上传</button>
<div id="result"></div>
<script>
async function upload() {
    const file = document.getElementById('file').files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async function(e) {
        const base64 = e.target.result.split(',')[1];
        const resp = await fetch('/upload', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image_base64: base64, mime_type: file.type})
        });
        const data = await resp.json();
        if (data.url) {
            document.getElementById('result').innerHTML = '<p>图片链接：<br><input type="text" value="' + data.url + '" style="width:100%;" onclick="this.select()"></p>';
        } else {
            document.getElementById('result').innerHTML = '<p>上传失败</p>';
        }
    };
    reader.readAsDataURL(file);
}
</script>
</body>
</html>"""
    return HTMLResponse(html)


async def upload_image(request: Request):
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
    image_id = request.path_params["image_id"]
    if image_id not in _image_store:
        return JSONResponse({"error": "Image not found or expired"}, status_code=404)

    image_data = _image_store[image_id]
    image_bytes = base64.b64decode(image_data["data"])
    return Response(content=image_bytes, media_type=image_data["mime_type"])


async def check_task(request: Request):
    """Check wanx task status and return image if ready."""
    task_id = request.path_params["task_id"]
    generator = get_generator()
    if not generator:
        return JSONResponse({"error": "Generator not available"}, status_code=500)

    try:
        result = await generator.check_task(task_id)
        status = result.get("status")

        if status == "SUCCEEDED":
            image_url = result.get("image_url")
            async with httpx_lib.AsyncClient(timeout=60.0) as client:
                img_resp = await client.get(image_url)
                if img_resp.status_code == 200:
                    image_base64 = base64.b64encode(img_resp.content).decode("utf-8")
                    mime_type = "image/png"
                    if image_url.lower().endswith((".jpg", ".jpeg")):
                        mime_type = "image/jpeg"
                    image_id = store_image(image_base64, mime_type)
                    base_url = get_base_url()
                    local_url = f"{base_url}/images/{image_id}"
                    return RedirectResponse(url=local_url)
            return JSONResponse({"status": "succeeded", "dashscope_url": image_url})

        elif status == "FAILED":
            error_msg = result.get("error", "Unknown error")
            return HTMLResponse(f"<h2>❌ 生成失败</h2><p>{error_msg}</p>")

        else:
            return HTMLResponse(
                f"<h2>⏳ 还在画...</h2><p>状态：{status}</p>"
                f"<script>setTimeout(function(){{location.reload()}}, 3000)</script>"
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Starlette app
starlette_app = Starlette(
    routes=[
        Route("/", index, methods=["GET"]),
        Route("/health", health_check, methods=["GET"]),
        Route("/upload", upload_image, methods=["POST"]),
        Route("/upload-page", upload_page, methods=["GET"]),
        Route("/images/{image_id}", get_image, methods=["GET"]),
        Route("/tasks/{task_id}", check_task, methods=["GET"]),
    ]
)


async def app(scope, receive, send):
    if scope["type"] == "http":
        path = scope.get("path", "")
        if path == "/sse":
            await handle_sse_asgi(scope, receive, send)
            return
        if path.startswith("/messages"):
            await handle_messages_asgi(scope, receive, send)
            return
    await starlette_app(scope, receive, send)


def main():
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting Hitomi MCP Server on {host}:{port}")
    logger.info(f"SSE endpoint: /sse")
    logger.info(f"Messages endpoint: /messages/")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
