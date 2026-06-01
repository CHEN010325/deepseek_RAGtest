import { chatCompletion, LLMQuotaExceededError, messageText } from "../llm.js";
import { isAnswerCorrect, stripThinking } from "./scoring.js";

export { LLMQuotaExceededError };

function stripJsonFences(text) {
  return String(text ?? "")
    .trim()
    .replace(/^```(?:json)?\s*/iu, "")
    .replace(/\s*```$/u, "")
    .trim();
}

function repairJsonCandidate(text) {
  return String(text)
    .replace(/,\s*([}\]])/gu, "$1")
    .replace(/\\(?!["\\/bfnrtu])/gu, "\\\\");
}

function balancedJsonCandidates(text) {
  const candidates = [];
  let start = -1;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (inString) {
      if (escape) escape = false;
      else if (char === "\\") escape = true;
      else if (char === "\"") inString = false;
      continue;
    }
    if (char === "\"") inString = true;
    else if (char === "{") {
      if (depth === 0) start = index;
      depth += 1;
    } else if (char === "}" && depth) {
      depth -= 1;
      if (depth === 0 && start >= 0) {
        candidates.push(text.slice(start, index + 1));
        start = -1;
      }
    }
  }
  return candidates;
}

export function extractJsonObject(rawText) {
  const text = stripJsonFences(rawText);
  const candidates = [text];
  for (const candidate of balancedJsonCandidates(text)) {
    if (!candidates.includes(candidate)) candidates.push(candidate);
  }
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start >= 0 && end > start) {
    const snippet = text.slice(start, end + 1);
    if (!candidates.includes(snippet)) candidates.push(snippet);
  }

  let lastError = "empty response";
  for (const candidate of candidates) {
    for (const attempt of [candidate, repairJsonCandidate(candidate)]) {
      try {
        const payload = JSON.parse(attempt);
        if (payload && typeof payload === "object" && !Array.isArray(payload)) return payload;
        lastError = "top-level JSON is not an object";
      } catch (error) {
        lastError = error.message;
      }
    }
  }
  throw new Error(`judge did not return a JSON object: ${lastError}`);
}

function jsonBool(value) {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return Boolean(value);
  if (typeof value === "string") return ["1", "true", "yes", "y", "correct", "对", "正确"].includes(value.trim().toLowerCase());
  return false;
}

async function extractJudgePayloadOrRetry(rawText, timeout, config) {
  try {
    return extractJsonObject(rawText);
  } catch {
    const repairPrompt = `
Convert the following judge output into one valid JSON object.
Return only JSON. No markdown. No explanation.
Schema: {"correct": true, "reason": "short reason"}

Original output:
${String(rawText).slice(0, 4000)}
`.trim();
    const repairData = await chatCompletion(
      [
        { role: "system", content: "You repair malformed JSON. Return only one valid JSON object." },
        { role: "user", content: repairPrompt }
      ],
      256,
      timeout,
      "judge json repair",
      config
    );
    return extractJsonObject(stripThinking(messageText(repairData)));
  }
}

export async function judgeAnswerWithModel(query, answers, prediction, goldQuotes, timeout = 300, config = null) {
  const prompt = `
你是 RAG 测评裁判。请判断“模型回答”是否覆盖“标准答案”的全部关键要点。
判定规则：
1. 不要求逐字一致，允许同义改写、顺序变化、引用编号和少量无害补充。
2. 数字、日期、人名、药名、方名、列表项等关键事实不能错漏。
3. 标准答案有多个并列要点时，模型回答必须覆盖全部核心要点。
4. 如果模型回答与标准答案矛盾、缺少关键项、答非所问，判为 false。
5. 只输出 JSON，不要输出额外文字。
输出格式：{"correct": true, "reason": "简短理由"}

问题：${query}

标准答案：${JSON.stringify(answers)}

标准证据：${JSON.stringify(goldQuotes.slice(0, 3))}

模型回答：${prediction}
`.trim();
  const data = await chatCompletion(
    [
      { role: "system", content: "你是严格、稳定的 RAG 答案测评裁判，只输出 JSON。" },
      { role: "user", content: prompt }
    ],
    512,
    timeout,
    "judge",
    config
  );
  const payload = await extractJudgePayloadOrRetry(stripThinking(messageText(data)), timeout, config);
  return {
    correct: jsonBool(payload.correct),
    reason: String(payload.reason ?? "").trim()
  };
}

export async function scoreAnswerWithFallback(query, answers, prediction, goldQuotes, timeout = 300, config = null) {
  if (isAnswerCorrect(prediction, answers)) {
    return { correct: true, method: "rule", judge: null };
  }
  const judge = await judgeAnswerWithModel(query, answers, prediction, goldQuotes, timeout, config);
  return { correct: Boolean(judge.correct), method: "model_judge", judge };
}
