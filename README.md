# Hitomi

通过 MCP (Model Context Protocol) 为 DeepSeek 等纯文本模型添加 **看图** 和 **画图** 能力。

## 功能

- **看图 (Vision)**：分析图片内容、提取文字 (OCR)、比较图片
- **画图 (Image Generation)**：根据文字描述生成图片、根据参考图生成新场景

## 工作原理

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Kelivo    │────▶│  DeepSeek   │────▶│  MCP Server     │
│  (Client)   │     │   V3 API    │     │ (Vision + Gen)  │
└─────────────┘     └─────────────┘     └────────┬────────┘
                                                 │
                              ┌──────────────────┼──────────────────┐
                              ▼                                     ▼
                     ┌─────────────────┐                   ┌─────────────────┐
                     │  Vision Model   │                   │  Image Model    │
                     │  (Qwen-VL-Max)  │                   │  (Gemini 3 Pro) │
                     └─────────────────┘                   └─────────────────┘
```

## 支持的后端

### Vision (看图)

| 提供商 | 模型 | 说明 |
|--------|------|------|
| **通义千问** | qwen-vl-max | 阿里云通义千问视觉模型 (推荐) |
| **OpenAI** | gpt-4o | OpenAI 视觉模型 |
| **Ollama** | llava | 本地运行的开源视觉模型 |

### Image Generation (画图)

| 提供商 | 模型 | 说明 |
|--------|------|------|
| **Gemini** | gemini-3-pro-image-preview | Google Gemini 图片生成 (推荐) |

## 部署

### Railway 云端部署（推荐）

1. **Fork 仓库到你的 GitHub**

2. **在 Railway 创建项目**
   - 登录 [Railway](https://railway.app)
   - 点击 "New Project" → "Deploy from GitHub repo"
   - 选择你的仓库

3. **配置环境变量**
   在 Railway 的 Variables 中添加：
   ```
   # Vision (看图) - 使用通义千问
   VISION_PROVIDER=qwen
   QWEN_API_KEY=your_dashscope_api_key

   # Image Generation (画图) - 使用 Gemini
   GEMINI_API_KEY=your_google_api_key
   ```

4. **部署完成后获取 URL**
   Railway 会给你一个类似 `https://your-app.up.railway.app` 的 URL

5. **在 Kelivo 中配置 MCP**
   ```json
   {
     "mcpServers": {
       "vision": {
         "transport": "sse",
         "url": "https://your-app.up.railway.app/sse"
       }
     }
   }
   ```

### 本地运行

```bash
# 安装依赖
pip install -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 运行
python -m hitomi.server
```

## 环境变量

```bash
# ===== Vision 配置 =====
VISION_PROVIDER=qwen              # 可选: qwen, openai, ollama
QWEN_API_KEY=your_key             # 通义千问 API Key

# ===== Image Generation 配置 =====
GEMINI_API_KEY=your_key           # Google Gemini API Key

# ===== 服务器配置 =====
HOST=0.0.0.0
PORT=8000
```

## 提供的工具

### 1. `get_upload_url` - 获取上传页面
获取图片上传页面的网址，用户可以上传图片获取链接。

### 2. `analyze_image` - 图片分析
分析图片内容，返回详细描述。

**参数：**
- `image` (必需): 图片 URL 或 base64 数据
- `question` (可选): 关于图片的具体问题
- `detail` (可选): 分析精度 - `low`/`high`/`auto`

### 3. `extract_text` - 文字提取 (OCR)
从图片中提取文字内容。

**参数：**
- `image` (必需): 图片 URL 或 base64 数据
- `language` (可选): 预期语言 - `zh`/`en`/`auto`

### 4. `compare_images` - 图片对比
比较两张图片的异同。

**参数：**
- `image1` (必需): 第一张图片
- `image2` (必需): 第二张图片
- `focus` (可选): 比较重点 - `differences`/`similarities`/`all`

### 5. `generate_image` - 生成图片
根据文字描述生成图片。

**参数：**
- `prompt` (必需): 图片描述
- `aspect_ratio` (可选): 宽高比 - `1:1`/`16:9`/`9:16`/`4:3`/`3:4`

### 6. `transform_image` - 场景转换
分析参考图片，然后根据描述生成新场景。适合"用这个人设画 xxx 场景"这类需求。

**参数：**
- `reference_image` (必需): 参考图片 URL 或 base64
- `scene_prompt` (必需): 新场景描述
- `style_hints` (可选): 风格提示
- `focus_on` (可选): 要关注的特征

## 示例对话

```
用户: 帮我看看这张图片 [发送图片链接]
AI: [调用 analyze_image] 这张图片显示的是...

用户: 画一只在月球上喝咖啡的猫
AI: [调用 generate_image] 🎨 图片已生成！[显示图片]

用户: 用这张人设图画一个在海边玩耍的场景 [发送人设图链接]
AI: [调用 transform_image] 🎨 新场景已生成！[显示图片]
```

## API Key 获取

- **通义千问 (Qwen)**: [阿里云 DashScope](https://dashscope.aliyun.com/)
- **Gemini**: [Google AI Studio](https://aistudio.google.com/)

## License

MIT

---

<p align="center">
  <sub>Built with 🌀 <a href="https://github.com/anthropics/claude-code">Claude Code</a></sub>
</p>

