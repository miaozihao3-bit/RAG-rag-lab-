# 备考知识库 RAG（rag-lab）

> 一句话：把 8 份四级 AI 备考笔记做成一个**可检索、可引用、可拒答**的最小 RAG，
> 既能当复习工具，也能作为 45A 操作题/综合题的素材库。

## 1. 目标与边界

**目标**

- 用一份真实语料跑通 RAG 全链路：解析 → 切分 → 索引 → 召回 → 重排 → 生成 → 引用 → 拒答。
- 每个核心机制由你手写，AI 只写工程壳；每阶段结束项目必须可运行。
- 产出可复述的数字：chunk 数、recall@3、拒答准确率、平均延迟。

**非目标（考后再说）**

- 不做 UI、登录、多租户、权限、Docker、线上部署。
- 不追求 SOTA 指标；不接多个模型；不做 Agent。

## 2. DoD（唯一验收标准）

- [ ] `python -m rag.cli index` 成功，输出 **来源数 / chunk 数 / 平均 chunk 长度**。
- [ ] `python -m rag.cli ask "Serving 优化有哪些手段？"` 返回答案 + **≥1 个来源** + 延迟。
- [ ] 3 个"笔记能答"的问题答对且带来源；3 个"笔记答不了"的问题明确拒答，不编造。
- [ ] `pytest -q` 退出码 0（M2–M4 的 `xfail` 是待完成任务，不是失败）。
- [ ] `python -m rag.cli eval` 输出 **recall@3、拒答准确率、引用覆盖率、平均延迟**。
- [ ] `uvicorn rag.api:app --port 8000` 后，`POST /ask` 可用。
- [ ] README 记录：chunk 大小 / overlap / top_k / 拒答阈值 / 3 个失败案例。

## 3. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| 语料 | `data/corpus/*.md`（8 份备考笔记） | 真实、可引用、无版权问题 |
| 切分 | 标题感知 + 段落打包 + overlap | 对应考纲"按章节/条款切" |
| 检索 | TF-IDF 字符 n-gram（默认） | 零下载、可离线、先跑通 |
| 重排 | MMR（最大边际相关性） | 练"相关性 vs 多样性"的取舍 |
| 生成 | 抽取式（默认）/ OpenAI 兼容 API | 没 key 也能跑；有 key 走真生成 |
| 服务 | FastAPI | 对应综合题 3 的 API 形态 |
| 评测 | 20 题小集（15 可答 + 5 不可答） | 对应 RAG 评测指标 |

> Python **3.8+** 兼容：不用 `int | None`、`list[str]` 这类 3.10+ 语法。

## 4. 目录

```
rag-lab/
├── README.md            # 本文件：目标与验收
├── 学习地图.md           # 进度、已验收能力、待加固
├── SKILL.md             # 教练规则（已按本项目适配）
├── requirements.txt
├── .env.example
├── data/corpus/         # 8 份备考笔记
├── eval/questions.jsonl # 20 题评测集
├── rag/
│   ├── config.py        # 路径、参数
│   ├── types.py         # Chunk / Hit / Answer
│   ├── ingest.py        # 读取 + 切分（M2 你写 chunk_markdown）
│   ├── retrieve.py      # TF-IDF 召回 + MMR（M3 你写 mmr_select）
│   ├── generate.py      # Prompt + 引用 + 拒答（M4 你写两个函数）
│   ├── pipeline.py      # 串起来
│   ├── api.py           # FastAPI /ask
│   ├── cli.py           # index / ask / eval
│   └── evaluate.py      # 20 题评测
├── tests/               # 每个核心机制都有测试
└── outputs/index/       # 生成的索引
```

## 5. 运行

```bash
cd rag-lab
pip install -r requirements.txt

# 建索引
python -m rag.cli index

# 问一个问题
python -m rag.cli ask "Serving 优化有哪些手段？"

# 评测
python -m rag.cli eval

# 起服务
uvicorn rag.api:app --port 8000
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"RAG 的完整链路是什么？\",\"top_k\":3}"
```

> PowerShell 5 控制台如果中文乱码，先执行 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`，或改用 `curl.exe`。

### M1 基线（AI 预跑，待你复现）

| 指标 | 数值 |
|---|---|
| 来源数 / chunk 数 | 8 / 141 |
| 平均 chunk 长度 | 506.6 字符 |
| recall@3 | 0.8667 |
| 拒答准确率 | 0.85 |
| 引用覆盖率 | 1.0 |
| 平均延迟 | 5.3 ms |

> 这是"最丑但能跑"的基线。M2 切分、M3 重排、M4 拒答之后，数字应该变好；变好多少，就是你的通关证据。

## 6. 配置（可选）

复制 `.env.example` 为 `.env`，填写：

```bash
LLM_BASE_URL=https://api.example.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=your-model
```

不填也能跑：默认走**抽取式回答**（返回检索片段 + 来源，不调用大模型）。

## 7. 阶段（详见 `学习地图.md`）

| 阶段 | 谁写 | 内容 | 收口 |
|---|---|---|---|
| M1 | AI | 最丑但能跑：索引 + TF-IDF + 抽取式回答 | 你跑 `index` + `ask`，贴输出 |
| M2 | 你 | `chunk_markdown` 标题感知切分 + overlap | `pytest tests/test_chunking.py` 通过 |
| M3 | 你 | `mmr_select` 重排（相关性 vs 多样性） | `pytest tests/test_retrieve.py` 通过 |
| M4 | 你 | `build_prompt` + `should_refuse` 引用与拒答 | `pytest tests/test_generate.py` 通过 |
| M5 | AI 壳 / 你验收 | FastAPI + 20 题评测 + README 数字 | `eval` 出四项指标 |

## 8. 考试映射

| 项目动作 | 45A 对应考点 | 60 秒能说的话 |
|---|---|---|
| 切分 | RAG 切分、overlap、问答错位 | "按标题/段落切，600 字、overlap 100；太大稀释，太小断语义" |
| TF-IDF 召回 | 向量化、余弦、Top-K | "先字符 n-gram 做基线，再换 Embedding 比召回" |
| MMR 重排 | 重排序、相关性/多样性 | "粗排 20 → MMR 选 5，λ 平衡相关与去重" |
| 引用 + 拒答 | 忠实度、幻觉、人在回路 | "低分拒答；答案带来源；低于阈值转人工" |
| 20 题评测 | recall@k、忠实度、延迟 | "recall@3 = X%，拒答准确率 Y%，P50 延迟 Z ms" |

## 9. 实测数字与失败案例（DoD 第 7 条）

**当前参数**：`CHUNK_SIZE=600` / `overlap=100` / `top_k=3` / `candidate_k=20` / `MMR λ=0.5` / `拒答阈值=0.04`

**20 题评测（抽取式回答，未调大模型）**

| 指标 | M1 基线 | M2–M4 之后 |
|---|---|---|
| recall@3 | 0.8667 | **0.8000** |
| 拒答准确率 | 0.8500 | **0.7500** |
| 引用覆盖率 | 1.0000 | 1.0000 |
| 平均延迟 | 5.3 ms | 7.8 ms |

**三个失败案例**

1. **词面误判**：问"红烧肉怎么做？"，最高分 **0.2620**，比真问题"Serving 优化有哪些手段？"（0.1588）还高。TF-IDF 只认字不认义，"怎么做"在笔记里出现过，就判成相关。
2. **概念分散**：问"RAG 的完整链路是什么？"，前 3 条分别是选择题答案表（0.0755）、备考心法（0.0645）、自查清单（0.0435），没有一条讲链路。链路知识散在 5 篇笔记的二十多个块里，单块检索拼不出来。
3. **指标与体感脱节**：recall@3 从 0.8667 降到 0.8（15 道能答题少命中 1 道），但 top-3 的重复度确实下降了。指标降不等于答案变差 —— 反过来也一样。

## 10. 已知限制

- TF-IDF 是词面匹配，同义改写会漏；Embedding 是考后升级项。
- 抽取式回答不是生成，只用于离线跑通；真实答案需要 LLM。
- MMR 只做多样性重排，不是 cross-encoder 精排。
- 评测集只有 20 题，指标只用于自测，不代表线上效果。
