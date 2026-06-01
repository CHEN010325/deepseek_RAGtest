# RAG Dataset Builder

`rag_dataset_builder.py` turns local documents into auditable JSONL QA datasets for the existing `E:\RAG` evaluation scripts.

For interactive use, run the local UI:

```powershell
python E:\RAG\rag_dataset_ui.py --host 127.0.0.1 --port 7861
```

Then open:

```text
http://127.0.0.1:7861
```

## Output Shape

Each output line is centered on a question, answer, and auditable evidence:

```json
{"id": 0, "query": "...", "answer": ["..."], "evidence": [{"quote": "...", "source_file": "..."}]}
```

For compatibility with the existing `E:\RAG` evaluator, the builder also writes `positive` as the same supporting evidence quote and `negative` as an empty list by default. Negative examples are not required for the real QA/evidence benchmark.

It also adds review fields:

- `evidence`: exact source quote, source file, page/chunk, character offsets.
- `qa_type`: `fact`, `numeric`, `date`, `list`, or `comparison`.
- `difficulty`: rough annotation difficulty.
- `answer_aliases`: optional acceptable variants.
- `meta`: generator, timestamp, schema, source chunk id.

The builder rejects model outputs when the evidence quote is not copied from the source chunk or when the answer text is not present in that quote.

The UI fixes generation to one QA item per source chunk to avoid low-quality question stuffing.

## Supported Inputs

The extension list mirrors the current `deepseekmine` upload boundary:

`.pdf`, `.doc`, `.docx`, `.ppt`, `.pptx`, `.txt`, `.md`, `.markdown`, `.canvas`, `.log`, `.eml`, `.epub`, `.xls`, `.xlsx`, `.csv`, `.json`, `.jsonl`, `.py`, `.js`, `.ts`, `.html`, `.htm`, `.css`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tiff`, `.tif`, `.gif`, `.heic`, `.wps`, `.wpt`, `.et`, `.ett`, plus `.zip` archives.

The command-line builder itself does not OCR images. Use the UI MinerU option for image-only or scanned documents, or OCR them first with `deepseekmine`/MinerU/PaddleOCR and feed the resulting PDF, Markdown, or text into the CLI builder.

The UI uses MiMo for QA generation by default:

- endpoint: `https://token-plan-cn.xiaomimimo.com/v1/chat/completions`
- model: `mimo-v2.5-pro`
- auth: `api-key`
- thinking: disabled

The API key is read from `MIMO_API_KEY`, or from the local ignored file `E:\RAG\.mimo_api_key`.

The UI can call MinerU directly for OCR/parse. The MinerU token is fixed through `MINERU_API_TOKEN`, or from the local ignored file `E:\RAG\.mineru_api_key`:

```powershell
python E:\RAG\rag_dataset_ui.py
```

The MinerU integration uses the precise parsing API:

- `POST /api/v4/file-urls/batch`
- `PUT` to the returned upload URL
- `GET /api/v4/extract-results/batch/{batch_id}`
- download `full_zip_url` and extract Markdown

When a PDF has more than 200 pages, the UI physically splits it into smaller PDF parts of at most 200 pages, submits those parts, then merges the returned Markdown in original page order.

## Examples

Generate 100 QA rows from one file or a directory with local Ollama:

```powershell
python E:\RAG\rag_dataset_builder.py `
  --input E:\docs\my_report.pdf E:\docs\policy_folder `
  --output E:\RAG\data\my_eval.json `
  --backend ollama `
  --model qwen3.5:4b `
  --target-count 100
```

Use MiMo/OpenAI-compatible scoring style generation:

```powershell
$env:MIMO_API_KEY="..."
python E:\RAG\rag_dataset_builder.py `
  --input E:\docs `
  --output E:\RAG\data\my_eval.json `
  --backend mimo `
  --model qwen3.5:4b
```

Smoke-test the extraction pipeline without an LLM:

```powershell
python E:\RAG\rag_dataset_builder.py `
  --input E:\docs\sample.txt `
  --output E:\RAG\temp_docs\builder_smoke.json `
  --backend none `
  --allow-heuristic `
  --target-count 3
```

The heuristic mode is only for plumbing checks, not final benchmark data.

## QA Review Checklist

Before using a generated dataset as benchmark truth:

- Spot-check evidence: every answer should be directly supported by `evidence.quote`.
- Remove broad summary questions and ambiguous questions.
- Prefer exact, stable answers: dates, names, definitions, counts, percentages, table values.
- Keep multi-answer questions only when the query clearly asks for all listed entities.
- Inspect `*.report.json` for skipped files, rejected model outputs, and parser limitations.

Run existing evaluation with the generated dataset name, for example if the file is `E:\RAG\data\my_eval.json`:

```powershell
python E:\RAG\run_current_retrieval_eval.py --dataset my_eval --mode retrieval
```

## Real deepseekmine RAG Evaluation

The intended business benchmark has two stages:

1. Generate the gold dataset from PDFs/documents.
   - output: `E:\RAG\data\<dataset>.json`
   - corpus uploaded for evaluation: `E:\RAG\data\<dataset>.corpus.md`
   - gold fields: `query`, `answer`, `evidence`

2. Run the generated dataset through the real deepseekmine RAG chain.
   - upload the corpus through `deepseekmine /api/files/upload`
   - call `deepseekmine /api/search` for every `query`
   - use the returned `result_prompt` as the full prompt to MiMo
   - compare final prediction with `answer`
   - compare retrieved `hits` with gold `evidence`

The UI now includes a `真实 RAG 测评` section for this flow. The command-line equivalent is:

```powershell
python E:\RAG\deepseekmine_rag_eval.py `
  --dataset my_eval `
  --api-base http://127.0.0.1:3335 `
  --mode qa
```

The UI evaluates a whole generated dataset at once. Each run uploads the dataset's complete `*.corpus.md` into a fresh deepseekmine knowledge base, so the test mirrors a user uploading one complete document set and asking all benchmark questions against it.

Reports are written to:

```text
E:\RAG\result-zh\deepseekmine_<mode>_<dataset>_<timestamp>.json
```

Main metrics:

- `evidence_hit_rate`: whether deepseekmine retrieved the gold evidence.
- `evidence_recall`: gold evidence coverage.
- `mrr`: rank of the first matched evidence.
- `qa_accuracy`: final answer accuracy after `result_prompt` is sent to MiMo.
