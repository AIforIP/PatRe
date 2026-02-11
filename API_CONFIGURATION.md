# API 密钥配置说明

为确保项目的安全性，请按照以下方法之一配置 API 密钥：

## 方法一：环境变量（推荐）

1. 复制 `.env.example` 文件为 `.env`：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，替换为您实际的 API 密钥：
   ```bash
   OPENAI_API_KEY=sk-your_actual_api_key_here
   OPENAI_BASE_URL=https://api.lqshop.cc/v1
   ```

3. 确保 `.env` 文件已被添加到 `.gitignore` 中，避免提交到版本控制系统。

## 方法二：配置文件

编辑 `config/eval_config.yaml` 文件，将 `YOUR_API_KEY_HERE` 替换为实际的 API 密钥：

```yaml
models:
  # ... 其他配置
  api_key: "sk-your_actual_api_key_here"  # 替换为实际的 API 密钥
  base_url: "https://api.lqshop.cc/v1"
```

## 优先级

API 密钥的加载遵循以下优先级顺序（从高到低）：

1. 环境变量 `OPENAI_API_KEY`
2. 配置文件中的 `api_key` 字段
3. 配置文件中的 `models.api_key` 字段

## 安全提醒

- **切勿**将包含实际 API 密钥的文件提交到版本控制系统
- 定期轮换 API 密钥以增强安全性
- 限制 API 密钥的权限范围，仅授予必要的访问权限