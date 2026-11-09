"""Pipeline v3 candidate (the pre-registration log, "Pipeline v3 candidate"). The frozen modules are
imported and patched at runtime; nothing in them is changed.

  retrieval(True)   retrieval queries = English control + Indonesian translation +
                    Indonesian element queries (retrieval only; never assessed)
  make_analyzer()   LLMAnalyzer with the ELEMENT CHECK added to the assessor prompt
                    and the per-element instruction added to the extractor prompt

The assessor still judges each control as ONE obligation; there is no decomposition
and no conjunction over machine-made sub-obligations.
"""
import json

from main import CrossEncoderReranker, LLMAnalyzer, MultiQueryEnsembleRetriever, RegulatoryDocumentParser

ID = json.load(open("synthetic_iso27001_uupdp_dataset/iso_annex_a_id.json", encoding="utf-8"))["translations"]
ELEMENTS = json.load(open("synthetic_iso27001_uupdp_dataset/iso_annex_a_elements_id.json", encoding="utf-8"))["elements"]
REGS = RegulatoryDocumentParser().parse_document(open("iso.txt", encoding="utf-8").readlines(), doc_type="ISO27001")
EXTRA_QUERIES = {r["id"]: [ID[r["id"]]] + ELEMENTS.get(r["id"], []) for r in REGS if r["id"] in ID}
_BY_TEXT = {r["text"]: EXTRA_QUERIES[r["id"]] for r in REGS if r["id"] in EXTRA_QUERIES}
PER_SUB_TOP = 2

ELEMENT_CHECK_BLOCK = (
    "ELEMENT CHECK (do this for every obligation before choosing its status).\n"
    "List the elements the obligation names: each distinct required action, object, scope, timing or "
    "condition (for example 'defined', 'approved by management', 'communicated to personnel', 'reviewed at "
    "planned intervals'). Evaluative words such as 'appropriate', 'relevant' or 'adequate' are not separate "
    "elements. Where the obligation offers alternatives ('rules, procedures or agreements'; 'at planned "
    "intervals or when significant changes occur'), one alternative is enough; where it joins elements with "
    "'and', each is required.\n"
    "For each element state whether the quoted evidence establishes it and which quote does. YES only if "
    "every element is established; PARTIAL if at least one is established and at least one is not; NO if "
    "none is. Do not invent elements the obligation does not name, and do not demand more for an element "
    "than rule 2b already requires.\n"
    "Begin the rationale with the checklist: 'Elements: <element> - established (<quote>); <element> - not "
    "established; ...'.\n\n"
)
STRICT_2B_BLOCK = (
    "ROLE AND CADENCE MUST BELONG TO THE SAME ACTIVITY (rule 2b, strict).\n"
    "A role, procedure, frequency or criteria satisfies rule 2b only if the sentence that states it is about "
    "the very activity the obligation requires (for entry controls: who controls entry or how entry is "
    "controlled; for software installation: who installs or approves installations). It does not count when "
    "it is stated for a different activity in the same provision or document - for example, verifying the "
    "construction of a perimeter is not operating entry controls. The owner, author or approver of the "
    "document itself ('Penanggung jawab panduan ini adalah ...', 'Dokumen ini disetujui oleh ...') is never "
    "the responsible role for an activity. If the only role or cadence available belongs to another activity "
    "or to the document, the obligation is a restatement: PARTIAL.\n\n"
)
EXTRACTOR_ELEMENT_LINE = (
    "For every obligation, quote the evidence for EACH element it names (action, object, scope, timing, "
    "condition), not only its most general statement; where the elements are addressed in different "
    "clauses, quote each of those clauses.\n\n"
)

SEVERITY_SYSTEM_PROMPT = (
    "You classify a documentation gap found in a Stage 1 (test of design) review of a company policy "
    "against one ISO/IEC 27001:2022 Annex A control. The assessor has already decided the control is not "
    "fully established; you only decide how severe the gap is.\n"
    "MAJOR GAP: the control's core requirement is not addressed at all; OR it is mentioned but with no "
    "auditable element - no responsible party, no mechanism or procedure, and no explicit reference to a "
    "governing procedure; OR most of the elements the control names are missing.\n"
    "MINOR GAP: the core requirement is present and auditable (a responsible party and/or a mechanism is "
    "clear), but one or a few complementary elements the control names are missing.\n"
    "Base the decision only on the assessor's finding and the quoted evidence given.\n"
    'Output ONLY JSON: {"gap_level": "Major" or "Minor", "reason": "one sentence"}'
)


def classify_gap(analyzer, control_text, finding, quotes):
    """Minor/Major severity for an assessor PARTIAL (pre-specified in the pre-registration log,
    'Gap severity'). Uses the analyzer's own LLM call (same model, same retries)."""
    import re
    quoted = "\n".join(f"- {q}" for q in quotes) or "- none"
    raw = analyzer._llm_call(
        system_prompt=SEVERITY_SYSTEM_PROMPT,
        user_prompt=f"CONTROL:\n{control_text}\n\nASSESSOR FINDING (status PARTIAL):\n{finding}\n\nQUOTED EVIDENCE:\n{quoted}",
        response_format={"type": "json_object"},
    )
    try:
        d = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
        level = str(d.get("gap_level", "")).strip().capitalize()
        if level in ("Major", "Minor"):
            return {"gap_level": level + " Gap", "reason": str(d.get("reason", ""))[:400]}
    except (ValueError, AttributeError):
        pass
    return {"gap_level": None, "reason": f"unparseable: {raw[:200]}"}


_ORIG_RETRIEVE = MultiQueryEnsembleRetriever.retrieve
_ORIG_RERANK = CrossEncoderReranker.rerank


def _v3_retrieve(self, expanded_queries, regulations, **kw):
    q = dict(expanded_queries)
    for r in regulations:
        if r["id"] in EXTRA_QUERIES and r["id"] not in q:      # controls judged as one obligation
            q[r["id"]] = [r["text"]] + EXTRA_QUERIES[r["id"]]
    return _ORIG_RETRIEVE(self, expanded_queries=q, regulations=regulations, **kw)


def _v3_rerank(self, query_text, candidates, guaranteed_ids=None, sub_queries=None, per_sub_top=3):
    extra = _BY_TEXT.get(query_text)
    if extra and (not sub_queries or list(sub_queries) == [query_text]):
        return _ORIG_RERANK(self, query_text=query_text, candidates=candidates, guaranteed_ids=guaranteed_ids,
                            sub_queries=[query_text] + extra, per_sub_top=PER_SUB_TOP)
    return _ORIG_RERANK(self, query_text=query_text, candidates=candidates, guaranteed_ids=guaranteed_ids,
                        sub_queries=sub_queries, per_sub_top=per_sub_top)


def retrieval(v3: bool):
    """Switches the v3 retrieval queries on or off (class-level patch)."""
    MultiQueryEnsembleRetriever.retrieve = _v3_retrieve if v3 else _ORIG_RETRIEVE
    CrossEncoderReranker.rerank = _v3_rerank if v3 else _ORIG_RERANK


def make_analyzer(variant="v3", **kw):
    """A frozen LLMAnalyzer with the v3 prompt additions (instance attributes only).
    variant 'v3b' also adds the strict same-activity rule for 2b."""
    if variant not in ("v3", "v3b"):
        raise ValueError(variant)
    an = LLMAnalyzer(**kw)
    anchor = "Output ONLY valid JSON.\n\n"
    if an.assessor_system_prompt.count(anchor) != 1:
        raise RuntimeError("assessor prompt anchor not found exactly once")
    block = ELEMENT_CHECK_BLOCK + (STRICT_2B_BLOCK if variant == "v3b" else "")
    an.assessor_system_prompt = an.assessor_system_prompt.replace(anchor, block + anchor, 1)
    head = "You are an objective evidence extractor.\n\n"
    if an.extractor_system_prompt.count(head) != 1:
        raise RuntimeError("extractor prompt head not found exactly once")
    an.extractor_system_prompt = an.extractor_system_prompt.replace(head, head + EXTRACTOR_ELEMENT_LINE, 1)
    return an
