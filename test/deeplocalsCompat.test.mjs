import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";
import { buildMineruMarkdownChunkDocs, writeChunkSidecar } from "../src/evaluation/deeplocalsCompat.js";

test("buildMineruMarkdownChunkDocs preserves headings and page metadata", () => {
  const markdown = [
    "# 第一章",
    "",
    "这里介绍香港治理趋势。",
    "",
    "# 第二章",
    "",
    "这里介绍经济发展趋势。"
  ].join("\n");
  const docs = buildMineruMarkdownChunkDocs(
    markdown,
    "source.md",
    "kb",
    [{ page: 3, start: 0, end: markdown.length }],
    []
  );

  assert.equal(docs.length, 2);
  assert.equal(docs[0].title, "source.md");
  assert.equal(docs[0].page, "3");
  assert.match(docs[0].content, /第一章/u);
  assert.match(docs[1].content, /第二章/u);
});

test("writeChunkSidecar writes DeepLocals compatible schema", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "rageval-sidecar-"));
  const markdownPath = path.join(dir, "doc.mineru.md");
  const markdown = "# 标题\n\n这个文档用于测评。";
  await fs.writeFile(markdownPath, markdown, "utf8");
  await fs.writeFile(`${markdownPath}.mineru.json`, "{}\n", "utf8");
  await fs.writeFile(
    `${markdownPath}.meta.json`,
    JSON.stringify({ page_spans: [{ page: 1, start: 0, end: markdown.length }], semantic_spans: [] }),
    "utf8"
  );

  const sidecarPath = await writeChunkSidecar(markdownPath, "demo");
  const sidecar = JSON.parse(await fs.readFile(sidecarPath, "utf8"));

  assert.equal(sidecar.schema, "deepseekmine-mineru-chunks-v1");
  assert.equal(sidecar.chunks.length, 1);
  assert.equal(sidecar.chunks[0].page, "1");
  assert.match(sidecar.chunks[0].content, /这个文档用于测评/u);
});
