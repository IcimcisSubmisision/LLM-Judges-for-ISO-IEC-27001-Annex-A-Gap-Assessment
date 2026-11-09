# LLM Judges for ISO/IEC 27001 Annex A Gap Assessment — replication package

Code, prompts, queries, settings, panel labels, and per-control verdicts for the paper
*LLM Judges for ISO/IEC 27001 Annex A Gap Assessment: Agreement With Practitioners, Evidence Safeguards, and Repeatability*
(ICIMCIS 2026, under review).

A retrieval pipeline quotes policy sentences for each applicable Annex A control. A judge model (gemma4:31b or
gpt-oss:120b) rates the control Compliant or Not Compliant against a documentation rubric. Two safeguards can
overturn a Compliant verdict: an evidence gate and a contrastive verifier. The verdicts are compared with blind
labels from three GRC practitioners on 1,946 controls.

## Reproduce the reported figures

```bash
python scripts/reproduce_tables.py
```

This needs only the Python standard library and runs in under a minute. It recomputes, from `labels/` and `results/`:

- Table III (both judges, the tuning and held-out company split, and judging time per company);
- Tables IV–V (government documents and public corpora), including the paired tests against the majority-class predictor;
- Table VI (safeguard ablation) and the post-hoc evidence-gate threshold sweep on the nine development companies;
- the gpt-oss:120b own-query run;
- the practitioners' inter-rater agreement on the government documents;
- repeatability over three runs on the government documents;
- the first failing stage of every false Not Compliant verdict (Section IV-E), gate firing rates (Section IV-D), and
  the agreement between the constructed policies' design and the panel (Section III-B).

Point estimates, counts, and p-values match the paper exactly. Bootstrap intervals use seed 20261107 and a different
resampling order than the original analysis, so their bounds can differ from the paper by up to ±0.02.

Not recomputed here:

- **Re-run on seven constructed companies (620 controls; 0.6% and 4.5% of verdicts changed).** Its raw records are not
  part of this package, so these figures are reported in the paper only.
- **Processing times other than the mean judging time** (retrieval minutes and per-company ranges): the per-run
  minutes are in `results/run_times.csv`. The mean judging time leaves out runs with LLM errors or a quota stop,
  whose minutes include retry waits (one gpt-oss:120b run).
- **Qualitative error analysis, the panel's reported assessment time, and realism ratings.** These rest on quoted
  policy text, judge rationales, or the practitioners' notes, which are not released (see "Not included").
- **Retrieval selection on SecureNusantara (evidence recall 47% to 76%).** It needs the element-level evidence
  annotation, which is not released.

## Contents

| Path | What it is |
|---|---|
| `scripts/reproduce_tables.py` | Recomputes the figures listed above from the CSV files below |
| `scripts/prepare_inputs.py` | Builds the input layout that `code/` expects (see "Running the pipeline") |
| `results/verdicts.csv` | One row per set × control × judge × query source × run (5,026 rows), giving each stage's outcome: evidence status, gate fired, top reranker score among the evidence passed to the judge (rounded to 4 decimals), verifier result, assessor votes, and, for the constructed corpus, whether the section the policy was designed to evidence the control with was retrieved. `run` is `reported` for the runs in the paper's tables and `repeat_1` / `repeat_2` for the two repeats on the government documents |
| `results/run_times.csv` | Retrieval and judging minutes per run segment, with LLM errors and quota stops |
| `labels/constructed_panel.csv` | 18 constructed companies: sector, compliance tier, tuning/held-out split, designed status, and panel label per control |
| `labels/public_panel.csv` | Hack23 and SecureNusantara panel labels; `applicable = 0` marks the nine physical controls that Hack23 lists as Not Applicable |
| `labels/government_raters.csv` | Bappenas and DJPb: the three practitioners' independent labels and the majority reference |
| `prompts/` | The exact system, extractor, assessor, and verifier prompts of the reported runs; their SHA-256 prefixes are in `config/frozen_settings.json` |
| `queries/` | Evidence queries written by gemma4:31b (used in all reported runs) and by gpt-oss:120b (own-query runs), plus the Annex A control titles |
| `config/frozen_settings.json` | Retrieval, gate, verifier, and judge settings, run dates, input hashes, and code hashes |
| `code/` | The pipeline (`r5_job.py`, `r5_job_ossq.py` and the modules they import), the scripts that regenerate the query files (`make_*.py`), and the scripts that turn the public and government documents into pipeline input (`ingest_*.py`, `extract_pdf_text.py`, `ocr_lines.py`) |

Labels use the panel's three-level scale: No Gap → Compliant; Minor Gap and Major Gap → Not Compliant. These are
documentation-gap grades for this study, not the nonconformity grades of a certification audit.

The evidence gate fires when the top reranker score is below 0.15, unless a sentence added by control-specific keyword
matching scores above 0.05 (`FORCED_CE_MINIMUM` in `code/main.py`). In the reported runs this exception applied only to
SecureNusantara A.6.2 (stored score 0.0906, gate not fired). On the constructed corpus all 3,166 assessed decisions
(both judges) equal `top_reranker_score < 0.15`, and the threshold sweep uses its nine development companies only.

## Running the pipeline

Step 1 runs from the repository root; steps 3 and 4 run from `code/`. A run needs Python with `requirements.txt`,
Ollama with the two cloud models, and the inputs below.

**1. Set up the layout.**

```bash
pip install -r requirements.txt
python scripts/prepare_inputs.py
```

This copies `queries/*.json` into `code/synthetic_iso27001_uupdp_dataset/`, creates its empty `segmented_policies/`,
writes the golden files (the controls to judge per document set) to `code/golden/` from
`labels/`, and writes two stand-in files that do not affect Annex A verdicts:

- `expanded_queries-v1.1.json` (English control elements). With `EXPANSION=off`, as in every reported run, all ISO
  entries of this file are dropped right after loading, so an empty JSON object gives identical verdicts. The original
  file is derived from the copyrighted Annex A text and is not released.
- `UUPDP_ID.txt` and `UUPDP_EN.txt` (Indonesia's Personal Data Protection Law, from an earlier experiment). The loader
  needs them to exist; empty files parse to no clauses.

**2. Supply the ISO/IEC 27001:2022 Annex A text.** ISO holds the copyright, so take it from a licensed copy:

- `code/iso.txt`: the control text;
- `code/synthetic_iso27001_uupdp_dataset/iso_annex_a_id.json` (Indonesian translation) and
  `iso_annex_a_elements_id.json` (Indonesian element queries), which `make_iso_id.py` and `make_iso_elements.py`
  regenerate from `iso.txt`.

**3. Build the document corpora** into `code/synthetic_iso27001_uupdp_dataset/segmented_policies/`, then compare each
file's SHA-256 with `inputs.segmented_corpus_sha256` in `config/frozen_settings.json`.

| Set | Source | Command |
|---|---|---|
| Hack23 | `Hack23/ISMS-PUBLIC`, commit 9424eb3, Apache 2.0 | `python ingest_hack23.py --src <checkout> --policies-only` |
| SecureNusantara | `hazwanjaafar/mssp-isms-iso27001-simulation`, commit 8d2def4, MIT | `python -I ingest_mssp21.py --src <checkout>` |
| Bappenas | guideline No. 2/2024, JDIH Bappenas | `python extract_pdf_text.py <pdf> inputs/bappenas_text.txt` then `python ingest_bappenas.py` |
| DJPb | regulation PER-1/PB/2021, JDIH Kemenkeu (scan) | OCR each page into `inputs/djpb/img/p<n>.tsv` (x, y, w, h, word), then `python ocr_lines.py inputs/djpb` and `python ingest_djpb.py inputs/djpb/djpb_text.txt` |

The source PDFs and extracted texts should match the SHA-256 values in `inputs.source_files`. The DJPb text of the
reported run came from the Windows OCR engine; another OCR engine gives different text, so check the corpus hash.
All corpus files have CRLF line endings, which the ingest scripts write on every platform.

**4. Run one document set with one judge.**

```bash
EXPANSION=off CONTEXT_MODE=provision RUBRIC=v2 GATE_MODE=assess AGGREGATION=threshold QUOTE_MODE=full_sentence \
VERIFY=record VERIFY_MODE=annexa EVAL_WORKERS=4 \
python r5_job.py --set BAPPENAS --client BAPPENAS-PPN --golden golden/golden_bappenas.json \
                 --judge gemma4:31b-cloud --outdir hasil
```

Clients: `HACK23-ORG` (`golden/golden_hack23.json`), `MSSP21-SecureNusantara` (`golden/golden_mssp21.json`),
`BAPPENAS-PPN`, and `DJPB-PER1`.

`r5_job.py` retrieves the evidence for a set once, caches it in `<outdir>/retrieval/<set>.json`, and then judges every
control, so both judges see identical evidence. `r5_job_ossq.py` does the same with the gpt-oss-written evidence
queries. **Use a separate `--outdir` for `r5_job.py` and `r5_job_ossq.py`**: the two runners use different evidence
queries, and a cache built with one must not be judged with the other. The released runners record the query file's
hash in the cache and stop if it does not match (caches written by the original run code carry no hash).

The gate and verifier outcomes are recorded and applied when scoring, as `scripts/reproduce_tables.py` does.
Cloud-served models may change over time, so a new run can differ from the reported verdicts; run dates are in
`config/frozen_settings.json`.

## Changes from the run code

`code/` holds the code that produced the results, with development-only parts removed and a few additions. Every
removed part needed files that are not released:

- `v3_eval.py` keeps only the `company` mode that `r5_job.py` uses. The development modes (`dev`, `hack23`, and an
  internal document) are removed.
- `v5_retrieval.py` keeps the R5 retrieval patches; the development metric on blind labels is removed.
- `analyze_v2.py` (development decision rules) is not included; `scripts/reproduce_tables.py` implements its verdict
  rule.
- Docstrings no longer point to the private pre-registration log, two comments in `main.py` no longer name an internal
  document, and line endings are LF.
- `r5_job.py` and `r5_job_ossq.py` record the evidence-query hash in the retrieval cache and refuse a mismatching cache.
- The ingestion scripts take their source paths as arguments, no longer write golden files
  (`scripts/prepare_inputs.py` does), and write CRLF line endings on every platform, as the run's files have;
  `extract_pdf_text.py` is the text-extraction step of the original Bappenas ingestion.

None of these changes touch the path that produced the verdicts: `r5_job.py` → `v3_eval.py` (company mode) → R5
retrieval → frozen analyzer. `evaluate_golden.py` is byte-identical to the run version. `config/frozen_settings.json`
lists the SHA-256 of both the run code and the released code.

## Not included

- Retrieved quotes and judge rationales (the raw run records). They quote document text, and the records carry
  local paths.
- The raw records of the re-run on seven constructed companies (620 controls).

## Data notes

- SecureNusantara was ingested unmodified: its Statement of Applicability, including the implementation-status column,
  is part of the corpus, and the judges saw an SoA row with "Status: Implemented" or "Partial" for 86 of its 93
  controls. The SoA declares 75 controls Implemented; the panel rated 39 of them Minor Gap (`labels/public_panel.csv`).
  Running `code/ingest_mssp21.py` on the unmodified checkout at commit 8d2def4 reproduces the corpus SHA-256 in
  `config/frozen_settings.json`.
- Hack23's compliance checklist shows its status as badge images, which ingestion strips, so no status values from it
  reach the judge.

## License

Code: MIT (see `LICENSE`). Labels, results, prompts, and queries: CC BY 4.0.

## Citation

To be added after publication.
