const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs');
const OpenAI = require('openai');
const pdfParse = require('pdf-parse');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, '..', 'public')));

// ===== Vercel 兼容：内存存储（无磁盘依赖）=====
const IS_VERCEL = !!process.env.VERCEL;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB
  fileFilter: (req, file, cb) => {
    const allowed = /jpeg|jpg|png|gif|webp|pdf/;
    const ext = allowed.test(path.extname(file.originalname).toLowerCase());
    const mime = allowed.test(file.mimetype);
    cb(null, ext || mime);
  }
});

// AI client
console.log('[boot] MODEL=%s BASE=%s KEY=%s...', process.env.DASHSCOPE_MODEL, process.env.DASHSCOPE_BASE_URL, (process.env.DASHSCOPE_API_KEY||'').substring(0,10));
const client = new OpenAI({
  apiKey: process.env.DASHSCOPE_API_KEY,
  baseURL: process.env.DASHSCOPE_BASE_URL,
});
const MODEL = process.env.DASHSCOPE_MODEL || 'qwen-plus';
const ENABLE_THINKING = process.env.ENABLE_THINKING === 'true';

// ============ PROMPTS ============

const INDICATORS_KNOWLEDGE = fs.readFileSync(
  path.join(__dirname, 'indicators_lite.md'), 'utf-8'
);

function buildOcrPrompt() {
  return `你是一位专业的体检报告OCR助手。请从图片中提取所有体检指标数据。

输出要求（严格JSON格式）：
{
  "patient": {
    "name": "姓名",
    "gender": "性别",
    "age": 年龄数字,
    "height": 身高cm,
    "weight": 体重kg,
    "bmi": BMI值,
    "blood_pressure": "收缩压/舒张压",
    "heart_rate": 心率
  },
  "indicators": [
    {
      "category": "模块名(如肝功能/血脂/血常规)",
      "name": "指标中文名",
      "value": "检测值(含单位)",
      "reference": "参考范围",
      "status": "normal/high/low",
      "arrow": "↑/↓/空"
    }
  ],
  "imaging": [
    {
      "type": "检查类型(如胸部CT/腹部B超)",
      "findings": "发现描述",
      "conclusion": "结论"
    }
  ]
}

规则：
1. 尽可能提取所有指标，不要遗漏
2. status 根据参考范围判断：超出为 high，低于为 low，正常为 normal
3. 如果图片模糊看不清某个值，用 "unclear" 标记
4. patient 信息如果图片中没有，对应字段填 null
5. 只输出 JSON，不要任何其他文字`;
}

function buildPdfOcrPrompt(pdfText) {
  return `你是一位专业的体检报告解析助手。以下是从体检报告 PDF 中提取的文本内容，请从中提取所有体检指标数据。

## PDF 文本内容
${pdfText}

## 输出要求（严格JSON格式）：
{
  "patient": {
    "name": "姓名",
    "gender": "性别",
    "age": 年龄数字,
    "height": 身高cm,
    "weight": 体重kg,
    "bmi": BMI值,
    "blood_pressure": "收缩压/舒张压",
    "heart_rate": 心率
  },
  "indicators": [
    {
      "category": "模块名(如肝功能/血脂/血常规)",
      "name": "指标中文名",
      "value": "检测值(含单位)",
      "reference": "参考范围",
      "status": "normal/high/low",
      "arrow": "↑/↓/空"
    }
  ],
  "imaging": [
    {
      "type": "检查类型(如胸部CT/腹部B超)",
      "findings": "发现描述",
      "conclusion": "结论"
    }
  ]
}

规则：
1. 尽可能提取所有指标，不要遗漏
2. status 根据参考范围判断：超出为 high，低于为 low，正常为 normal
3. 如果文本中某个值不清楚，用 "unclear" 标记
4. patient 信息如果文本中没有，对应字段填 null
5. 只输出 JSON，不要任何其他文字`;
}

function buildAnalysisPrompt(mode, ocrData, userInfo) {
  const modeDesc = mode === 'parent'
    ? '爸妈版（给50+岁长辈看，大字简明，只讲人话，不用专业术语）'
    : '专业版（给自己看，完整分析）';

  return `你是一位资深全科医生+AI健康顾问。请基于以下体检数据生成健康解读报告。

## 输出模式：${modeDesc}

## 参考知识库
${INDICATORS_KNOWLEDGE}

## 体检数据
${JSON.stringify(ocrData, null, 2)}

## 用户补充信息
${userInfo || '用户未提供额外信息'}

## 输出要求（严格JSON格式）

${mode === 'parent' ? buildParentOutputSpec() : buildProOutputSpec()}

重要：
1. 只输出 JSON，不要 markdown 代码块标记，不要任何其他文字
2. 所有判断必须基于知识库，不要臆断
3. ⚠️ 免责声明必须包含：本报告由AI生成，仅供参考，不替代医生诊断`;
}

function buildProOutputSpec() {
  return `{
  "score": {
    "total": 分数0-100,
    "level": "优秀/良好/一般/偏差/较差",
    "summary": "一段话总结（50字内）",
    "deductions": [{"item": "扣分项", "reason": "原因", "points": 扣分数}]
  },
  "red_flags": [
    {
      "title": "问题名称",
      "severity": "high/medium",
      "indicators": [{"name":"指标名","value":"值","reference":"参考范围","status":"high/low"}],
      "explanation": "通俗解释（100字内）",
      "cause": "可能原因",
      "action": "建议行动"
    }
  ],
  "yellow_flags": [
    {
      "title": "问题名称",
      "value": "检测值",
      "description": "简短说明"
    }
  ],
  "green_flags": ["正常指标1", "正常指标2"],
  "patterns": [
    {
      "name": "关联模式名",
      "status": "hit/near",
      "description": "解释"
    }
  ],
  "risks": [
    {
      "name": "风险名",
      "level": 1-3,
      "path": "演进路径"
    }
  ],
  "plan": {
    "diet": {
      "principle": "饮食原则（一句话）",
      "recommend": ["推荐食物1", "推荐食物2"],
      "avoid": ["避免食物1", "避免食物2"],
      "sample_menu": "一日三餐示例"
    },
    "exercise": {
      "principle": "运动原则（一句话）",
      "weekly": [
        {"day": "周一", "type": "运动类型", "duration": "时长", "icon": "emoji"}
      ],
      "caution": "运动禁忌"
    },
    "supplements": [
      {"name": "补剂名", "dose": "剂量", "timing": "服用时间", "reason": "原因", "priority": 1-3}
    ],
    "sleep": {
      "bedtime": "建议入睡时间",
      "tips": ["睡眠建议1", "建议2"]
    },
    "followup": [
      {"time": "时间", "department": "科室", "action": "做什么", "urgency": "high/medium/low"}
    ],
    "lifestyle": ["生活习惯建议1", "建议2"]
  },
  "monthly_plan": {
    "goals": ["目标1", "目标2", "目标3"],
    "weeks": [
      {
        "week": 1,
        "theme": "主题",
        "checklist": ["每日任务1", "任务2"],
        "target": "本周目标"
      }
    ]
  },
  "disclaimer": "⚠️ 本报告由AI基于体检数据生成，仅供健康管理参考，不构成医疗诊断或治疗建议。异常指标请务必前往对应专科就诊，遵医嘱处理。"
}`;
}

function buildParentOutputSpec() {
  return `{
  "greeting": "称呼语（如：阿姨/叔叔，您的体检结果我帮您看了）",
  "good_news": ["好消息1（纯大白话）", "好消息2"],
  "attention": [
    {
      "icon": "emoji",
      "title": "问题名（大白话，如：甲状腺有点问题）",
      "description": "简短说明（2句话以内，不用任何英文缩写和专业术语）"
    }
  ],
  "doctors": [
    {
      "time": "什么时候去（如：2周内）",
      "department": "挂什么科",
      "action": "去做什么（大白话）",
      "urgency": "high/medium/low"
    }
  ],
  "do_list": [
    {"action": "要做的事（如：每天喝一杯牛奶300ml）", "reason": "为什么（一句话）", "icon": "emoji"}
  ],
  "dont_list": [
    {"action": "不能做的事", "reason": "为什么（一句话）", "icon": "emoji"}
  ],
  "supplements": [
    {"name": "补剂名（中文）", "how": "怎么吃（大白话）", "icon": "emoji"}
  ],
  "safety_tips": ["家居安全建议1", "建议2"],
  "encouragement": "温暖的收尾话（3-4句，像晚辈关心长辈）",
  "disclaimer": "⚠️ 本报告由AI辅助生成，仅供参考，不代替医生诊断。有问题请去医院看医生。"
}`;
}

// ============ ROUTES ============

// Step 1: Upload image(s)/PDF → OCR extract
app.post('/api/ocr', upload.array('images', 10), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: '请上传体检报告（图片或PDF）' });
    }

    const allIndicators = { patient: null, indicators: [], imaging: [] };

    for (const file of req.files) {
      // Vercel 兼容：直接从 buffer 读取，不走磁盘
      const fileBuffer = file.buffer;
      const isPdf = file.mimetype === 'application/pdf' ||
                    path.extname(file.originalname).toLowerCase() === '.pdf';

      let response;

      try {
        if (isPdf) {
          // PDF → 提取文本 → 纯文本 prompt
          const pdfData = await pdfParse(fileBuffer);
          const pdfText = pdfData.text;

          if (!pdfText || pdfText.trim().length < 20) {
            console.warn('PDF text too short, possibly scanned image PDF:', file.originalname);
            allIndicators._scannedPdfDetected = true;
            continue;
          }

          // 截取前 15000 字符防止 token 超限
          const truncated = pdfText.substring(0, 15000);

          response = await client.chat.completions.create({
            model: MODEL,
            messages: [
              { role: 'user', content: buildPdfOcrPrompt(truncated) }
            ],
            max_tokens: 16384,
            temperature: 0.1,
            enable_thinking: ENABLE_THINKING,
          });
        } else {
          // 图片 → base64 → vision
          const base64 = fileBuffer.toString('base64');
          const mimeType = file.mimetype || 'image/jpeg';

          response = await client.chat.completions.create({
            model: MODEL,
            messages: [
              {
                role: 'user',
                content: [
                  { type: 'text', text: buildOcrPrompt() },
                  { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64}` } }
                ]
              }
            ],
            max_tokens: 16384,
            temperature: 0.1,
            enable_thinking: ENABLE_THINKING,
          });
        }

        let text = response.choices[0]?.message?.content || '';
        console.log('=== AI Response Start ===');
        console.log('Length:', text.length);
        console.log('First 300:', text.substring(0, 300));
        text = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
        text = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        const jsonStart = text.indexOf('{');
        const jsonEnd = text.lastIndexOf('}');
        if (jsonStart >= 0 && jsonEnd > jsonStart) {
          text = text.substring(jsonStart, jsonEnd + 1);
        }
        console.log('After clean, first 200:', text.substring(0, 200));

        try {
          const parsed = JSON.parse(text);
          console.log('Parsed OK, indicators:', parsed.indicators?.length, 'imaging:', parsed.imaging?.length);
          if (parsed.patient && !allIndicators.patient) {
            allIndicators.patient = parsed.patient;
          }
          if (parsed.indicators) {
            allIndicators.indicators.push(...parsed.indicators);
          }
          if (parsed.imaging) {
            allIndicators.imaging.push(...parsed.imaging);
          }
        } catch (e) {
          console.error('OCR parse error for file:', file.originalname, e.message);
          console.error('Text after clean:', text.substring(0, 500));
        }
      } catch (innerErr) {
        console.error('Process file error:', file.originalname, innerErr.message);
      }
    }

    if (allIndicators.indicators.length === 0 && allIndicators.imaging.length === 0) {
      if (allIndicators._scannedPdfDetected) {
        return res.status(422).json({ error: '这是扫描件 PDF（内部没有可读文字），请把每页分别截图后以图片方式上传' });
      }
      return res.status(422).json({ error: '未能从报告中识别出体检指标，请确认文件清晰度，或尝试拍照上传' });
    }

    delete allIndicators._scannedPdfDetected;
    res.json({ success: true, data: allIndicators });
  } catch (err) {
    console.error('OCR error:', err);
    res.status(500).json({ error: '识别失败: ' + err.message });
  }
});

// Step 2: Analyze indicators → Generate report (SSE streaming)
app.post('/api/analyze', async (req, res) => {
  const { ocrData, mode = 'pro', userInfo = '' } = req.body;

  if (!ocrData) {
    return res.status(400).json({ error: '缺少体检数据' });
  }

  let prompt;
  try {
    prompt = buildAnalysisPrompt(mode, ocrData, userInfo);
  } catch (e) {
    return res.status(400).json({ error: 'prompt 构造失败: ' + e.message });
  }

  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  if (typeof res.flushHeaders === 'function') res.flushHeaders();

  // 每 15s 一个心跳注释，防止 CDN 切长连接
  const heartbeat = setInterval(() => {
    try { res.write(`: heartbeat ${Date.now()}\n\n`); } catch (_) {}
  }, 15000);

  const sendEvent = (obj) => {
    try { res.write(`data: ${JSON.stringify(obj)}\n\n`); } catch (_) {}
  };

  let aborted = false;
  res.on('close', () => {
    if (!res.writableEnded) {
      aborted = true;
      console.log('[analyze] client disconnected, aborting');
    }
  });

  console.log('[analyze] start, mode=%s, indicators=%d, prompt_len=%d, MODEL=%s', mode, ocrData?.indicators?.length || 0, prompt.length, MODEL);

  try {
    const stream = await client.chat.completions.create({
      model: MODEL,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 8192,
      temperature: 0.3,
      stream: true,
      enable_thinking: ENABLE_THINKING,
    });
    console.log('[analyze] stream opened');

    let fullText = '';
    let chunkCount = 0;

    let rawCount = 0;
    for await (const chunk of stream) {
      rawCount++;
      if (rawCount <= 3) console.log('[analyze] raw chunk #%d:', rawCount, JSON.stringify(chunk).substring(0, 200));
      if (aborted) break;
      const content = chunk.choices[0]?.delta?.content || '';
      if (content) {
        chunkCount++;
        fullText += content;
        sendEvent({ type: 'chunk', content });
      }
    }
    console.log('[analyze] stream done, raw=%d, content_chunks=%d, text_len=%d', rawCount, chunkCount, fullText.length);

    if (aborted) return;

    let cleaned = fullText.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
    cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    const jsonStart = cleaned.indexOf('{');
    const jsonEnd = cleaned.lastIndexOf('}');
    if (jsonStart >= 0 && jsonEnd > jsonStart) {
      cleaned = cleaned.substring(jsonStart, jsonEnd + 1);
    }

    try {
      const parsed = JSON.parse(cleaned);
      sendEvent({ type: 'done', report: parsed });
    } catch (e) {
      console.error('Report JSON parse failed:', e.message);
      console.error('Raw length:', cleaned.length, 'preview:', cleaned.substring(0, 300));
      sendEvent({ type: 'done', report: null, raw: cleaned, parse_error: e.message });
    }
  } catch (err) {
    console.error('Analyze error:', err);
    sendEvent({ type: 'error', message: err.message || '分析失败' });
  } finally {
    clearInterval(heartbeat);
    try { res.end(); } catch (_) {}
  }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', model: MODEL, env: IS_VERCEL ? 'vercel' : 'local' });
});

// SPA fallback
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'index.html'));
});

// ===== 本地开发：直接启动 / Vercel：导出 app =====
if (!IS_VERCEL) {
  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`🏥 xiaohui-health running at http://localhost:${PORT}`);
  });
}

module.exports = app;
