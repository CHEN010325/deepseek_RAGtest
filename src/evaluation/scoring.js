export function normalizeText(value) {
  return String(value ?? "")
    .normalize("NFKC")
    .replace(/\s+/gu, "")
    .toLowerCase();
}

export function stripThinking(text) {
  return String(text ?? "").replace(/<think>.*?<\/think>/gis, "").trim();
}

export function normalizeAnswerText(value) {
  return normalizeText(
    String(value ?? "")
      .replace(/【\d+】|\[\d+\]|\(\d+\)/gu, "")
      .replace(/(\d+)\.0+(?=%|[^\d]|$)/gu, "$1")
  );
}

export function compactAnswerText(value) {
  return normalizeAnswerText(value).replace(/[\s，。；;：:、！!？?（）()[\]【】《》“”"'`·\-—/\\]+/gu, "");
}

export function answerItemCorrect(predictionNorm, answer) {
  const answerNorm = normalizeAnswerText(answer);
  const predCompact = compactAnswerText(predictionNorm);
  const answerCompact = compactAnswerText(answerNorm);
  if (!answerCompact) return false;
  if (predCompact.includes(answerCompact)) return true;
  if (answerCompact.length <= 6) return false;

  const answerChars = [...answerCompact].filter((char) => /[\p{Script=Han}A-Za-z0-9]/u.test(char));
  if (!answerChars.length) return false;
  const matchedChars = answerChars.filter((char) => predCompact.includes(char)).length;
  return matchedChars / answerChars.length >= 0.72;
}

export function isAnswerCorrect(prediction, answers) {
  const pred = normalizeAnswerText(prediction);
  return Array.isArray(answers) && answers.length > 0 && answers.every((answer) => answerItemCorrect(pred, answer));
}

export function hitMatchesEvidence(hitText, quote, answers) {
  const hitNorm = normalizeText(hitText);
  const quoteNorm = normalizeText(quote);
  if (!hitNorm || !quoteNorm) return false;
  if (hitNorm.includes(quoteNorm) || quoteNorm.includes(hitNorm)) return true;
  if (quoteNorm.length >= 24 && hitNorm.includes(quoteNorm.slice(0, 24))) return true;
  return Array.isArray(answers) && answers.length > 0 && answers.every((answer) => hitNorm.includes(normalizeText(answer)));
}

export function scoreRetrievalFromTexts(hitTexts, goldQuotes, answers) {
  const matchedRanks = [];
  for (const quote of goldQuotes) {
    for (let index = 0; index < hitTexts.length; index += 1) {
      if (hitMatchesEvidence(hitTexts[index], quote, answers)) {
        matchedRanks.push(index + 1);
        break;
      }
    }
  }

  const matchedCount = matchedRanks.length;
  const goldCount = Math.max(1, goldQuotes.length);
  const hitCount = hitTexts.length;
  const firstRank = matchedRanks.length ? Math.min(...matchedRanks) : null;
  return {
    evidence_hit: matchedCount > 0,
    evidence_recall: Math.min(matchedCount / goldCount, 1.0),
    evidence_precision: hitCount ? matchedCount / hitCount : 0.0,
    mrr: firstRank ? 1.0 / firstRank : 0.0,
    first_evidence_rank: firstRank,
    retrieved_count: hitCount
  };
}

export function summarizeDeepLocals(results) {
  const total = results.length;
  if (!total) {
    return {
      total: 0,
      evidence_hit_rate: 0.0,
      evidence_recall: 0.0,
      mrr: 0.0,
      qa_accuracy: 0.0,
      qa_correct: 0,
      qa_total: 0,
      qa_answer_errors: 0,
      qa_quota_errors: 0,
      avg_retrieved_count: 0.0
    };
  }
  const qaResults = results.filter((item) => Object.hasOwn(item, "answer_correct"));
  const qaCorrect = qaResults.filter((item) => item.answer_correct).length;
  return {
    total,
    evidence_hit_rate: results.filter((item) => item.retrieval?.evidence_hit).length / total,
    evidence_recall: results.reduce((sum, item) => sum + Number(item.retrieval?.evidence_recall ?? 0), 0) / total,
    mrr: results.reduce((sum, item) => sum + Number(item.retrieval?.mrr ?? 0), 0) / total,
    qa_accuracy: qaResults.length ? qaCorrect / qaResults.length : 0.0,
    qa_correct: qaCorrect,
    qa_total: qaResults.length,
    qa_answer_errors: qaResults.filter((item) => item.answer_judge_method === "mimo_answer_error").length,
    qa_quota_errors: qaResults.filter((item) => item.answer_judge_method === "api_quota_exhausted").length,
    avg_retrieved_count: results.reduce((sum, item) => sum + Number(item.retrieval?.retrieved_count ?? 0), 0) / total
  };
}

export function summarizeFlatQa(results) {
  const total = results.length;
  if (!total) {
    return {
      total: 0,
      qa_accuracy: 0.0,
      qa_correct: 0,
      qa_total: 0,
      evidence_hit_rate: 0.0,
      evidence_recall: 0.0,
      mrr: 0.0,
      avg_retrieved_count: 0.0
    };
  }
  const qaCorrect = results.filter((item) => item.answer_correct).length;
  return {
    total,
    qa_accuracy: qaCorrect / total,
    qa_correct: qaCorrect,
    qa_total: total,
    evidence_hit_rate: results.filter((item) => item.retrieval?.evidence_hit).length / total,
    evidence_recall: results.reduce((sum, item) => sum + Number(item.retrieval?.evidence_recall ?? 0), 0) / total,
    mrr: results.reduce((sum, item) => sum + Number(item.retrieval?.mrr ?? 0), 0) / total,
    avg_retrieved_count: results.reduce((sum, item) => sum + Number(item.retrieval?.retrieved_count ?? 0), 0) / total
  };
}
