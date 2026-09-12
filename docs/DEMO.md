# 体验 Memory Garden

示例使用合成笔记，不读取你的原始笔记库和既有对话。

## 离线体验

完成 README 中的依赖安装后，Windows 双击 `start-demo.bat`，打开 <http://127.0.0.1:8876>。

macOS / Linux 可运行：

```sh
MG_PUBLIC_DEMO_MODE=true MG_DEMO_USE_MODEL=false uv run memory-garden serve --port 8876
```

1. 点击“自主判断”等示例主题，查看前后记录。
2. 点击回答旁的来源按钮，核对原文。
3. 切换到时间线或关系图，浏览记录之间的联系。
4. 在可判断的回溯结果下选择“我想补充或修正”，再到“记忆”查看。

离线模式不会生成自然语言讨论；普通提问会提示连接对话模型。

## 体验模型对话

先在 `.env` 中配置生成模型的服务地址、模型名和密钥。Windows 双击 `start-agent-demo.bat`。

macOS / Linux 可运行：

```sh
MG_PUBLIC_DEMO_MODE=true MG_DEMO_USE_MODEL=true uv run memory-garden serve --port 8876
```

此模式会将你输入的示例对话及相关合成记录片段发送到配置的模型服务。请勿把不希望发送的私人内容输入示例对话。

## 连接自己的笔记

点击“连接笔记库”，输入本机 Obsidian 文件夹路径。新库默认本地模式；需要对话模型时，在该库设置中另行配置。
详细说明见[连接本地笔记库](LOCAL_LIBRARIES.md)。
