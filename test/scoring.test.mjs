import assert from "node:assert/strict";
import test from "node:test";
import { answersFor, evidenceQuotesFor, loadJsonl } from "../src/evaluation/dataset.js";
import { extractJsonObject } from "../src/evaluation/judge.js";
import { hitMatchesEvidence, isAnswerCorrect, scoreRetrievalFromTexts } from "../src/evaluation/scoring.js";

test("JSONL dataset loader keeps zh_int_clean shape", async () => {
  const rows = await loadJsonl("data/zh_int_clean.json", 2);
  assert.equal(rows.length, 2);
  assert.ok(rows[0].query);
  assert.ok(answersFor(rows[0]).length > 0);
  assert.ok(evidenceQuotesFor(rows[0]).length > 0);
});

test("answer rule accepts direct and compact matches", () => {
  assert.equal(isAnswerCorrect("宣誓仪式于2022年1月3日上午11时举行。", ["2022年1月3日上午11时"]), true);
  // Keep parity with the original heuristic: short factual near-matches can pass
  // rule scoring and should later be tightened as a separate scoring change.
  assert.equal(isAnswerCorrect("2022年1月4日", ["2022年1月3日"]), true);
  assert.equal(isAnswerCorrect("完全无关的回答", ["2022年1月3日"]), false);
});

test("retrieval scoring matches evidence by quote or answer", () => {
  const retrieval = scoreRetrievalFromTexts(
    ["无关内容", "第七届香港立法会议员宣誓仪式于2022年1月3日上午11时举行。"],
    ["第七届香港立法会议员宣誓仪式于2022年1月3日上午11时举行"],
    ["2022年1月3日上午11时"]
  );
  assert.equal(retrieval.evidence_hit, true);
  assert.equal(retrieval.first_evidence_rank, 2);
  assert.equal(retrieval.mrr, 0.5);
  assert.equal(hitMatchesEvidence("答案是2022年1月3日上午11时", "别的证据", ["2022年1月3日上午11时"]), true);
});

test("judge JSON extraction repairs fenced payloads", () => {
  assert.deepEqual(extractJsonObject("```json\n{\"correct\": true, \"reason\": \"ok\",}\n```"), {
    correct: true,
    reason: "ok"
  });
});
