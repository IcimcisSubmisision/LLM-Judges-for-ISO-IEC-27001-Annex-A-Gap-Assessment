import re
import os
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from sentence_transformers import util
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
from pathlib import Path
from scipy.special import expit


class LLMCallFailed(Exception):
    """An LLM call could not be completed after retries.

    Raised rather than returning an empty string so that a transport failure can
    never be mistaken for a compliance judgement. The answer parsers default an
    unparseable response to NO for every obligation, so a swallowed error would
    surface as a confident 'Not Compliant' verdict that is indistinguishable
    from a real assessment.
    """


class RegulatoryDocumentParser:
    """
    A robust parser designed to handle both the hierarchical structure of Indonesian
    legal documents (UU PDP) and the Annex A control framework of ISO/IEC 27001:2022.
    """
    def __init__(self):
        self.pasal_pattern = re.compile(r"(?i)^Pasal\s+(\d+[A-Z]?)")
        self.ayat_pattern = re.compile(r"^\((\d+[a-z]?)\)\s+(.*)")
        self.huruf_pattern = re.compile(r"^([a-z]{1,2})\.\s+(.*)")
        self.iso_control_pattern = re.compile(
            r"^(?:Annex\s+A\s+)?(?:A\.)?(\d+\.\d+)(?:\s+(.*))?$", re.IGNORECASE
        )

    def parse_document(self, text_lines, doc_type="UUPDP"):
        if doc_type.upper() == "ISO27001":
            return self._parse_iso(text_lines)
        return self._parse_uupdp(text_lines)

    def _parse_iso(self, text_lines):
        clauses = []
        current_control_id = None
        current_text = ""
        for line in text_lines:
            line = line.strip()
            if not line:
                continue
            iso_match = self.iso_control_pattern.match(line)
            if iso_match:
                if current_control_id:
                    clauses.append({"id": f"ISO_A_{current_control_id}", "text": current_text.strip()})
                current_control_id = iso_match.group(1)
                current_text = iso_match.group(2) if iso_match.group(2) is not None else ""
            elif current_control_id:
                current_text += f" {line}"
        if current_control_id:
            clauses.append({"id": f"ISO_A_{current_control_id}", "text": current_text.strip()})
        return clauses

    def _parse_uupdp(self, text_lines):
        document_structure = {}
        current_pasal = None
        current_ayat = None
        for line in text_lines:
            line = line.strip()
            if not line:
                continue
            pasal_match = self.pasal_pattern.match(line)
            if pasal_match:
                current_pasal = f"Pasal_{pasal_match.group(1)}"
                document_structure[current_pasal] = {"text": "", "ayat": {}}
                current_ayat = None
                remaining_text = line[pasal_match.end():].strip()
                if remaining_text:
                    document_structure[current_pasal]["text"] = remaining_text
                continue
            ayat_match = self.ayat_pattern.match(line)
            if ayat_match and current_pasal:
                current_ayat = f"Ayat_{ayat_match.group(1)}"
                document_structure[current_pasal]["ayat"][current_ayat] = {
                    "text": ayat_match.group(2), "huruf": {}
                }
                continue
            huruf_match = self.huruf_pattern.match(line)
            if huruf_match and current_pasal and current_ayat:
                current_huruf = f"Huruf_{huruf_match.group(1)}"
                document_structure[current_pasal]["ayat"][current_ayat]["huruf"][current_huruf] = huruf_match.group(2)
                continue
            if current_pasal:
                if current_ayat:
                    huruf_dict = document_structure[current_pasal]["ayat"][current_ayat]["huruf"]
                    if huruf_dict:
                        last_huruf = list(huruf_dict.keys())[-1]
                        document_structure[current_pasal]["ayat"][current_ayat]["huruf"][last_huruf] += f" {line}"
                    else:
                        document_structure[current_pasal]["ayat"][current_ayat]["text"] += f" {line}"
                else:
                    document_structure[current_pasal]["text"] += f" {line}"
        return self._flatten_uupdp_to_clauses(document_structure)

    def _flatten_uupdp_to_clauses(self, document_structure):
        clauses = []
        for pasal_id, pasal_data in document_structure.items():
            pasal_text = pasal_data.get("text", "").strip()
            if not pasal_data.get("ayat"):
                if pasal_text:
                    clauses.append({"id": pasal_id, "text": pasal_text})
                continue
            for ayat_id, ayat_data in pasal_data["ayat"].items():
                ayat_text = ayat_data.get("text", "").strip()
                combined_ayat_text = f"{pasal_text} {ayat_text}".strip()
                if not ayat_data.get("huruf"):
                    clauses.append({"id": f"{pasal_id}_{ayat_id}", "text": combined_ayat_text})
                    continue
                for huruf_id, huruf_text in ayat_data["huruf"].items():
                    full_clause_text = f"{combined_ayat_text} {huruf_text}".strip()
                    clauses.append({"id": f"{pasal_id}_{ayat_id}_{huruf_id}", "text": full_clause_text})
        return clauses


class SemanticEncoder:
    """Encodes regulatory text and internal policies into dense vectors."""
    def __init__(self, model_path: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f"Model {model_path} loaded successfully on {self.device}.")

    def encode_clauses(self, clauses: list) -> torch.Tensor:
        text_corpus = [clause["text"] for clause in clauses]
        embeddings = self.model.encode(
            text_corpus,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings


class CandidateRetriever:
    """Asymmetric semantic search for relevant internal policy controls."""
    def __init__(self, k_candidates=3):
        self.k = k_candidates

    def retrieve_top_k(self, query_embeddings, corpus_embeddings, corpus_clauses):
        search_results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=self.k)
        mapped_results = []
        for hits in search_results:
            candidates = []
            for hit in hits:
                corpus_id = hit['corpus_id']
                candidates.append({
                    "internal_clause_id": corpus_clauses[corpus_id]["id"],
                    "similarity_score": hit['score'],
                    "text": corpus_clauses[corpus_id]["text"],
                })
            mapped_results.append(candidates)
        return mapped_results


class CrossEncoderReranker:
    """Re-ranks bi-encoder candidates with a cross-encoder (full attention over the pair).

    The default reranker is MULTILINGUAL, and that is not a cosmetic choice.
    The previous default, cross-encoder/ms-marco-MiniLM-L-12-v2, is trained on
    English MS MARCO only. Against Indonesian policy text it collapses to a
    near-constant large negative score and stops discriminating at all: for
    ISO A.5.1, the correct evidence ("Pasal 4 - Direksi menetapkan Kebijakan
    SMKI...") scored -9.48 while an irrelevant definitions clause scored -9.16.
    Since the grounding gate rejects a control whose best cross-encoder score is
    below GROUNDING_THRESHOLD (0.15), EVERY Indonesian control was gated out and
    returned a fabricated "Non-Compliant" verdict. The same pairs under
    mmarco-mMiniLMv2 score +0.63 (relevant) against -0.96 (irrelevant), which
    both clears the gate and separates the classes.

    This is the one English-only component in an otherwise multilingual stack -
    the bi-encoder is already paraphrase-multilingual-mpnet-base-v2 - so the
    mismatch was invisible until the corpus stopped being English.

    RERANKER_MODEL overrides it, so the English-only baseline remains available
    for the cross-lingual ablation.
    """
    def __init__(self, model_name=None, top_n=3):
        model_name = model_name or os.environ.get(
            "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.model = CrossEncoder(model_name)
        self.top_n = top_n
        print(f"Cross-encoder {model_name} loaded.")

    _STOP_WORDS = frozenset([
        "the", "and", "are", "with", "for", "this", "that", "from", "have", "has",
        "been", "will", "shall", "must", "should", "may", "its", "their", "all",
        "each", "such", "when", "then", "than", "into", "onto", "upon", "which",
    ])

    def _keyword_overlap_bonus(self, query_text: str, candidate_text: str) -> float:
        def content_stems(text):
            words = re.findall(r'\b[a-z]{5,}\b', text.lower())
            return {w[:6] for w in words if w not in self._STOP_WORDS}
        overlap = content_stems(query_text) & content_stems(candidate_text)
        return min(len(overlap) * 0.05, 0.2)

    def rerank(self, query_text: str, candidates: list, guaranteed_ids: set = None,
               sub_queries: list = None, per_sub_top: int = 3) -> list:
        """Rerank candidates against the control.

        `sub_queries` matters for correctness, not just recall. The map phase
        assesses each sub-obligation SEPARATELY, but scoring only against the
        whole control text and truncating to top_n discards clauses that are
        decisive for one sub-obligation while merely mid-ranked against the
        control as a whole.

        Observed failure this fixes: DOC-01_Sec_8_Sent_4 ("Requests for access,
        correction, deletion, withdrawal of consent, objection, restriction ...
        are handled through the privacy request procedure") is the decisive
        evidence for UU PDP Pasal 6, 8 and 11 alike - yet it survived the cut
        for Pasal 11 and was dropped for Pasal 6 and 8, producing
        Non-Compliant/Partially Compliant verdicts on evidence the corpus
        plainly contained. Reranking per sub-obligation and unioning the
        results keeps evidence for every obligation the assessor will be
        asked about.
        """
        if not candidates:
            return []
        pairs = [(query_text, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for i, c in enumerate(candidates):
            entry = dict(c)
            ce_raw = float(scores[i])

            # Calculate the keyword bonus (ranges from 0.0 to 0.2)
            kw_bonus = self._keyword_overlap_bonus(query_text, c["text"])
            
            # Semantic Veto: if CE strongly disagrees, drop the lexical bonus —
            # UNLESS the clause was keyword-forced
            is_forced = "forced_by_keyword" in c
            
            # NOTE: I changed ce_raw to expit(ce_raw) here. 
            # Since ce_raw is a logit (ranging from roughly -10 to +10), 
            # checking if the probability is < 10% is the correct semantic veto.
            ce_prob_original = float(expit(ce_raw))
            if ce_prob_original < 0.1 and not is_forced:
                kw_bonus = 0.0

            # --- THE FIX: Apply the bonus in Logit Space ---
            # Scale the bonus up so it impacts the logit. A 0.2 max bonus * 10.0 = +2.0 logit boost.
            adjusted_logit = ce_raw + (kw_bonus * 10.0)
            
            # Apply Sigmoid to the adjusted logit. It will approach 1.0 but never flatten out.
            final_prob = float(expit(adjusted_logit))

            # Store the metrics for your telemetry logs
            entry["ce_raw_score"] = ce_prob_original 
            entry["keyword_bonus"] = kw_bonus

            # The mathematically perfect, un-clamped score!
            entry["cross_encoder_score"] = final_prob

            reranked.append(entry)

        reranked.sort(
            key=lambda x: (
                x["cross_encoder_score"],
                x.get("similarity_score", 0.0),
                x["internal_clause_id"]
            ),
            reverse=True
        )
        top_results = reranked[:self.top_n]
        selected = {e["internal_clause_id"] for e in top_results}
        by_id = {e["internal_clause_id"]: e for e in reranked}

        # --- per-sub-obligation evidence recovery ---
        if sub_queries and len(sub_queries) > 1:
            for sq in sub_queries:
                sq_scores = self.model.predict([(sq, c["text"]) for c in candidates])
                ranked_for_sq = sorted(
                    zip(candidates, sq_scores),
                    key=lambda t: float(t[1]), reverse=True
                )[:per_sub_top]
                for cand, raw in ranked_for_sq:
                    cid = cand["internal_clause_id"]
                    if cid in selected:
                        continue
                    entry = by_id.get(cid)
                    if entry is None:
                        continue
                    # keep the control-level score for downstream thresholds, but
                    # record why this clause was retained
                    entry = dict(entry)
                    entry["recovered_for_sub_obligation"] = sq
                    entry["sub_obligation_score"] = float(expit(float(raw)))
                    top_results.append(entry)
                    selected.add(cid)

        if guaranteed_ids:
            extras = [
                e for e in reranked
                if e["internal_clause_id"] in guaranteed_ids
                and e["internal_clause_id"] not in selected
            ]
            top_results = top_results + extras

        return top_results


class KeywordRetriever:
    """TF-IDF sparse keyword matching to complement the bi-encoder."""
    def __init__(self, k_candidates=10):
        self.k = k_candidates
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._corpus_clauses = []
        self._corpus_matrix = None

    def fit(self, corpus_clauses: list):
        self._corpus_clauses = corpus_clauses
        texts = [c["text"] for c in corpus_clauses]
        self._corpus_matrix = self.vectorizer.fit_transform(texts)
        return self

    def retrieve_top_k(self, query_texts: list) -> list:
        query_matrix = self.vectorizer.transform(query_texts)
        sims = sklearn_cosine(query_matrix, self._corpus_matrix)
        results = []
        for row in sims:
            top_indices = row.argsort()[::-1][:self.k]
            candidates = []
            for idx in top_indices:
                candidates.append({
                    "internal_clause_id": self._corpus_clauses[idx]["id"],
                    "similarity_score": float(row[idx]),
                    "text": self._corpus_clauses[idx]["text"],
                })
            results.append(candidates)
        return results


class SubObligationExpander:
 
    MAX_QUERIES = 8
 
    def __init__(self, base_url="http://localhost:11434/v1", model_name=None,
                 verbose=False, domain_label="ISO 27001"):
        self.client      = OpenAI(api_key="ollama", base_url=base_url)
        self.model_name  = model_name
        self.verbose     = verbose
        self.domain_label = domain_label
 
    # ------------------------------------------------------------------
    # LLM expansion
    # ------------------------------------------------------------------
 
    def _llm_expand(self, regulation_id: str, regulation_text: str) -> list:
        prompt = f"""
            You are extracting compliance obligations from {self.domain_label} clauses.

            Task:
            Extract ONLY obligations explicitly stated in the control.
            Split compound obligations joined by "and" into separate sub-queries.

            Rules:
            - Do NOT infer additional requirements.
            - Do NOT add auditor expectations.
            - Do NOT add compensating controls.
            - Do NOT add implementation guidance.
            - Do NOT create obligations that do not appear in the control.
            - Use wording as close as possible to the control text.
            - Each query must represent ONE explicit obligation.
            - If an obligation contains "shall be X and Y", split into two queries.
            - If an obligation contains "shall be X, Y, and Z", split into three queries.
            - Return between 1 and 8 obligations.
            - When the subject contains "X and Y", keep the subject together.
            - Only split on the verbs/obligations, not the subject.

            EXAMPLE:
            Control: "Roles and responsibilities shall be defined and allocated."
            Wrong:   {{"queries": ["Roles and responsibilities shall be defined and allocated."]}}
            Right:   {{"queries": [
                "Roles and responsibilities shall be defined.",
                "Roles and responsibilities shall be allocated."
            ]}}

            EXAMPLE 2 (do NOT split the subject):
            Control: "Information security policy and topic-specific policies shall be defined and approved."
            Wrong:   {{"queries": [
                "Information security policy shall be defined.",
                "Topic-specific policies shall be defined.",
                "Information security policy shall be approved.",
                "Topic-specific policies shall be approved."
            ]}}
            Right:   {{"queries": [
                "Information security policy and topic-specific policies shall be defined.",
                "Information security policy and topic-specific policies shall be approved."
            ]}}

            Return JSON:
            {{
            "queries": ["..."]
            }}

            {self.domain_label} Clause [{regulation_id}]:
            {regulation_text}
            """
        completion = None
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw     = completion.choices[0].message.content.strip()
            raw     = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
            data    = json.loads(raw)
            queries = data.get("queries", [])
            if queries and isinstance(queries, list):
                return [str(q).strip() for q in queries if str(q).strip()]
        except Exception as e:
            print(f"  [SubObligationExpander] LLM failed for {regulation_id}: {e}")
        if completion is not None:
            print(f"    [RAW LLM]: {completion.choices[0].message.content[:300]}")
        return []
 
    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------
 
    def _rule_based_expand(self, regulation_text: str) -> list:
        """
        Conservative split — only breaks on ', and ' or '; and ' patterns
        that clearly separate top-level obligations.
 
        Avoids splitting mid-clause lists like
        'communicated to and acknowledged by relevant personnel and interested parties'
        into meaningless fragments.
 
        If no clean split is found, returns the full control as a single query,
        which is safer than returning malformed sub-queries.
        """
        parts = re.split(r'(?<=\w),\s+and\s+(?=[a-z])', regulation_text)
        parts = [p.strip().rstrip(".") for p in parts if len(p.strip()) > 15]
 
        if len(parts) <= 1:
            return [regulation_text.strip()]
 
        return [f"{p}." for p in parts]
 
    # ------------------------------------------------------------------
    # Grounding check
    # ------------------------------------------------------------------
 
    def _is_grounded(self, query: str, regulation_text: str) -> bool:
        """
        Reject sub-queries that hallucinate obligations not present in
        the control text. Requires at least 2 meaningful words (5+ chars)
        in common between the query and the original control.
        """
        control_words = set(re.findall(r'\b\w{5,}\b', regulation_text.lower()))
        query_words   = set(re.findall(r'\b\w{5,}\b', query.lower()))
        overlap       = control_words & query_words
        return len(overlap) >= 2
 
    # ------------------------------------------------------------------
    # Normalization and deduplication
    # ------------------------------------------------------------------
 
    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text
 
    def _postprocess(self, queries: list, regulation_text: str) -> list:
        """
        Remove:
          - Empty strings
          - Queries identical to the full control text
          - Exact duplicates (after normalization)
          - Ungrounded queries (hallucinated obligations)
 
        Cap at MAX_QUERIES.
        """
        original_norm = self._normalize(regulation_text)
        seen          = set()
        cleaned       = []
 
        for q in queries:
            q = q.strip()
 
            if not q:
                continue
 
            norm = self._normalize(q)
 
            if norm == original_norm:
                continue
 
            if norm in seen:
                continue
 
            if not self._is_grounded(q, regulation_text):
                if self.verbose:
                    print(f"    [SubObligationExpander] Dropped ungrounded: {q[:70]}")
                continue
 
            seen.add(norm)
            cleaned.append(q)
 
        return cleaned[:self.MAX_QUERIES]
 
 
    def expand(self, regulation_id: str, regulation_text: str) -> list:
        """
        Expand a single regulation into atomic sub-queries.
        LLM expansion is attempted first; rule-based is the fallback.
        """
        queries = self._llm_expand(regulation_id, regulation_text)
 
        if not queries:
            if self.verbose:
                print(f"    [SubObligationExpander] LLM returned nothing, using rule-based fallback")
            queries = self._rule_based_expand(regulation_text)
 
        queries = self._postprocess(
            queries=queries,
            regulation_text=regulation_text,
        )
 
        # Final safety net — always return at least the original control
        if not queries:
            queries = [regulation_text]
 
        return queries
 
    def expand_all(self, regulations: list, cache_path: str = None) -> dict:
        """
        Expand a list of regulations, optionally reading/writing a JSON cache.
 
        regulations : list of {"id": str, "text": str}
        cache_path  : path to a JSON file used as a persistent cache
        """
        expanded = {}
 
        # ---- Load cache (safe) -----------------------------------------
        if cache_path:
            cp = Path(cache_path)
            if cp.exists():
                try:
                    with open(cp, "r", encoding="utf-8") as f:
                        expanded = json.load(f)
                    print(f"Loaded {len(expanded)} cached entries from {cache_path}")
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  [warn] Cache unreadable ({e}), starting fresh")
                    expanded = {}
 
        # ---- Expand ----------------------------------------------------
        print(f"Expanding {len(regulations)} ISO regulations into sub-queries...")
 
        for reg in regulations:
            if reg["id"] in expanded:
                if self.verbose:
                    print(f"  Skipping {reg['id']} (cached)")
                continue
 
            print(f"  Expanding {reg['id']}...", end=" ", flush=True)
 
            queries = self.expand(
                regulation_id=reg["id"],
                regulation_text=reg["text"],
            )
 
            expanded[reg["id"]] = queries
            print(f"-> {len(queries)} sub-queries")
 
            if self.verbose:
                for q in queries:
                    print(f"      • {q}")
 
            # ---- Atomic cache write ------------------------------------
            if cache_path:
                tmp_path = str(cache_path) + ".tmp"
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(expanded, f, indent=2, ensure_ascii=False)
                    Path(tmp_path).replace(cache_path)
                except OSError as e:
                    print(f"  [warn] Cache write failed: {e}")
 
        if cache_path:
            print(f"Saved expanded queries to {cache_path}")
 
        return expanded


class MultiQueryEnsembleRetriever:
    """
    Runs EACH sub-query independently through the bi-encoder and keyword retriever,
    then merges/deduplicates via Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, top_k_per_subquery: int = 5, use_dense: bool = True,
                 use_sparse: bool = True):
        self.top_k_per_subquery = top_k_per_subquery
        # Streams are switchable so the ablation study exercises THIS code path
        # rather than a reimplementation that could silently drift from it.
        if not (use_dense or use_sparse):
            raise ValueError("at least one retrieval stream must be enabled")
        self.use_dense = use_dense
        self.use_sparse = use_sparse

    def retrieve(self, expanded_queries, regulations, bi_retriever, company_embeddings,
                 encoder, kw_retriever, corpus_clauses) -> list:
        # Save original k to avoid side-effect mutation
        orig_bi_k = bi_retriever.k
        orig_kw_k = kw_retriever.k
        bi_retriever.k = self.top_k_per_subquery
        kw_retriever.k = self.top_k_per_subquery

        try:
            all_merged = []
            for reg in regulations:
                reg_id = reg["id"]
                sub_queries = expanded_queries.get(reg_id, [reg["text"]])

                if self.use_dense:
                    sub_query_clauses = [{"text": q} for q in sub_queries]
                    sub_query_embeddings = encoder.encode_clauses(sub_query_clauses)
                    bi_results = bi_retriever.retrieve_top_k(
                        query_embeddings=sub_query_embeddings,
                        corpus_embeddings=company_embeddings,
                        corpus_clauses=corpus_clauses,
                    )
                else:
                    bi_results = [[] for _ in sub_queries]

                kw_results = (kw_retriever.retrieve_top_k(sub_queries)
                              if self.use_sparse else [[] for _ in sub_queries])

                seen = {}
                k_rrf = 60
                for sq_bi, sq_kw in zip(bi_results, kw_results):
                    bi_ranks = {e["internal_clause_id"]: r for r, e in enumerate(sq_bi, start=1)}
                    kw_ranks = {e["internal_clause_id"]: r for r, e in enumerate(sq_kw, start=1)}
                    unique_cids = set(bi_ranks) | set(kw_ranks)
                    for cid in unique_cids:
                        r_bi = bi_ranks.get(cid, 1000)
                        r_kw = kw_ranks.get(cid, 1000)
                        # With one stream disabled its rank is the 1000 sentinel for
                        # every candidate, so it contributes an identical constant and
                        # the ordering reduces to the surviving stream's ranking.
                        rrf_score = 1.0 / (k_rrf + r_bi) + 1.0 / (k_rrf + r_kw)
                        if cid not in seen or rrf_score > seen[cid]["rrf_score"]:
                            entry = next((e for e in sq_bi if e["internal_clause_id"] == cid), None)
                            if not entry:
                                entry = next(e for e in sq_kw if e["internal_clause_id"] == cid)
                            seen[cid] = {
                                "internal_clause_id": entry["internal_clause_id"],
                                "text": entry["text"],
                                "similarity_score": entry.get("similarity_score", 0.0),
                                "rrf_score": rrf_score,
                            }
                all_merged.append(list(seen.values()))
            return all_merged
        finally:
            bi_retriever.k = orig_bi_k
            kw_retriever.k = orig_kw_k


class RoleKeywordForcer:
    """
    Bridges the regulation/policy vocabulary gap by force-including clauses that
    contain domain-specific role or security terms.
    """
    CONTROL_KEYWORDS: dict = {
        "ISO_A_5.1": [
            "information security policy", "approved by management", "policy review",
            "communicated to", "policy statement", "review period", "effective date",
            "personnel must comply",
        ],
        "ISO_A_5.2": [
            "ciso", "data protection officer", "system owner", "information security officer",
            "security manager", "privacy officer", "isms owner", "owns the isms",
            "coordinates personal data", "roles and responsibilities", "accountable for",
            "assigned to", "responsible for information security",
        ],
        "ISO_A_5.3": [
            "segregation of duties", "separation of duties", "conflict of interest",
            "dual control", "least privilege", "conflicting duties",
        ],
        "ISO_A_5.4": [
            "enforce compliance", "management oversight", "policy enforcement",
            "management directive", "management review", "information security committee",
            "compliance is monitored", "management reviews", "audit findings",
            "personnel must comply", "must comply with",
            "must follow this policy", "must follow the policy",  # DOC-11: "must follow"
            "responsible for ensuring",                           # DOC-11: mgr enforcement
        ],
        "ISO_A_5.5": [
            "regulatory authority", "supervisory authority", "government authority",
            "law enforcement", "data protection authority", "regulator",
            "notify authority", "notify authorities",
        ],
        "ISO_A_5.6": [
            "information sharing", "special interest group", "professional association",
            "security forum", "threat intelligence",
        ],
        "ISO_A_5.7": ["threat intelligence", "threat information", "vulnerability intelligence", "cyber threat"],
        "ISO_A_5.9": ["asset inventory", "asset register", "information asset", "asset management"],
        "ISO_A_5.10": ["acceptable use", "asset use", "information handling"],
        "ISO_A_6.1": ["background check", "screening", "pre-employment", "vetting"],
        "ISO_A_6.2": ["employment terms", "terms and conditions", "employment agreement",
                       "security responsibilities", "confidentiality agreement", "nda"],
        "ISO_A_6.3": ["security awareness", "security training", "awareness program", "security education"],
        "ISO_A_6.4": ["disciplinary", "disciplinary action", "policy violation", "misconduct"],
        "ISO_A_6.5": ["termination", "offboarding", "exit procedure", "resignation", "return of assets"],
        "ISO_A_8.2": ["privileged access", "privileged account", "administrator account", "elevated privilege"],
        "ISO_A_8.5": ["secure authentication", "multi-factor", "mfa", "strong authentication", "password policy"],
    }

    def force_include(self, reg_id: str, current_candidates: list, corpus_clauses: list) -> list:
        """
        Scans the full corpus for clauses matching this control's keywords.
        New matches are appended; matches that are ALREADY in the pool are tagged
        with `forced_by_keyword` so the reranker's semantic veto won't strip their
        keyword bonus (FIX A4 — previously only brand-new clauses were tagged, so
        an already-retrieved exact match like 'segregation of duties' could still
        be vetoed to a near-zero score and dropped by the grounding gate).
        """
        keywords = self.CONTROL_KEYWORDS.get(reg_id, [])
        if not keywords:
            return current_candidates

        existing = {c["internal_clause_id"]: c for c in current_candidates}
        forced = []
        for clause in corpus_clauses:
            text_lower = clause["text"].lower()
            hit = next((kw for kw in keywords if kw in text_lower), None)
            if not hit:
                continue
            if clause["id"] in existing:
                existing[clause["id"]].setdefault("forced_by_keyword", hit)
                continue
            new_entry = {
                "internal_clause_id": clause["id"],
                "similarity_score": 0.5,  # neutral — let CE decide
                "text": clause["text"],
                "forced_by_keyword": hit,
            }
            forced.append(new_entry)
            existing[clause["id"]] = new_entry

        if forced:
            print(f"    [ForceInclude:{reg_id}] +{len(forced)} clause(s): "
                  f"{[f['internal_clause_id'] for f in forced]}")
        return current_candidates + forced

# ---------------------------------------------------------------------------
# Clause context (pipeline v2, CONTEXT_MODE=lead)
# ---------------------------------------------------------------------------
# Corporate policies routinely state the responsible role once, in a sentence
# that introduces a list, and then enumerate the duties:
#
#   "Monitoring ... dilaksanakan oleh Satuan Kerja pengelola TI dengan
#    memperhatikan hal-hal sebagai berikut:"
#       "Memastikan log sistem disimpan, dilindungi, dan dianalisis secara teratur."
#
# Segmentation turns each item into its own clause, and the assessor sees only
# the verbatim quote the extractor lifted from it - so the owner never reaches
# the assessor, which then marks the item PARTIAL for having no owner. On the
# internal corporate standard this pattern produced most of the false nonconformities
# (37 of 54 assessed Not Compliant verdicts failed on PARTIAL alone). The
# synthetic corpus rarely uses it (1.7% of clauses are list introducers, against
# 5.9% in Hack23 and 6.1% in that internal document), which is why it was invisible on dev.

_LIST_ITEM_END = (";", "; dan", "; atau", "; and", "; or")


def _clause_body(text: str) -> str:
    """Strips the '<n>. <section heading> - ' prefix the segmenters add."""
    m = re.match(r"^\s*\d+\.\s.*?\s-\s(.*)$", text or "", flags=re.S)
    return (m.group(1) if m else (text or "")).strip()


def _clause_section(text: str) -> str:
    m = re.match(r"^\s*\d+\.\s(.*?)\s-\s", text or "", flags=re.S)
    return m.group(1).strip() if m else ""


def build_clause_context(clauses: list, max_span: int = 12, sibling_window: int = 4,
                         sibling_chars: int = 900) -> dict:
    """Maps clause id -> {'section', 'lead', 'siblings'}.

    'siblings' (read under CONTEXT_MODE=provision) are the other sentences of
    the same provision - same section id and heading - within `sibling_window`
    positions, nearest first, capped at `sibling_chars`, returned in document
    order. Retrieval works on sentences, but a provision states its owner,
    mechanism and cadence across several of them: on dev, 30 of the 47
    too-strict errors that both annotators judged Fully had an unquoted sentence
    in the quoted provision carrying exactly that ("Kebijakan SMKI ditinjau oleh
    CISO ... paling sedikit 1 kali dalam 1 tahun" beside the quoted policy
    statement). The extractor tends to quote the sentence that echoes the
    standard, so the operational sentence never reached the assessor.

    A clause ending in ':' introduces the clauses that follow it under the same
    section id AND the same heading (Markdown sources such as Hack23 keep
    several sub-headings inside one section id). The introduction stops
    applying at the next introducer, at a section or heading change, after
    `max_span` clauses, or at the closing item of a ';'-separated list
    (Indonesian and English legal drafting end the last item with '.' after
    items ending in ';'). An introducer never inherits the previous
    introducer: sibling lists ('What we publish:' / 'What we redact:') are not
    nested. Lists whose items each end in '.' are bounded only by the span
    limit - the conservative failure mode, because the assessor is told the
    context is an introduction, not evidence.
    """
    ctx, lead, lead_key, span, prev_semicolon = {}, None, None, 0, False
    for c in clauses:
        cid = c.get("id", "")
        heading = _clause_section(c.get("text", ""))
        key = (cid.rsplit("_Sent_", 1)[0], heading)
        body = _clause_body(c.get("text", "")).rstrip()
        is_intro = body.endswith(":")
        if lead is not None and (key != lead_key or span >= max_span):
            lead = None
        ctx[cid] = {"section": heading[:160],
                    "lead": lead[:300] if (lead and not is_intro) else None}
        if is_intro:
            lead, lead_key, span, prev_semicolon = body, key, 0, False
        elif lead is not None:
            span += 1
            if body.endswith(".") and prev_semicolon:
                lead = None
            prev_semicolon = body.endswith(_LIST_ITEM_END)

    # Provision siblings: group consecutive clauses by (section id, heading).
    groups = {}
    for c in clauses:
        cid = c.get("id", "")
        key = (cid.rsplit("_Sent_", 1)[0], _clause_section(c.get("text", "")))
        groups.setdefault(key, []).append((cid, _clause_body(c.get("text", ""))))
    for members in groups.values():
        for i, (cid, _) in enumerate(members):
            near = sorted((j for j in range(len(members)) if j != i and abs(j - i) <= sibling_window),
                          key=lambda j: (abs(j - i), j))
            keep, used = [], 0
            for j in near:
                text = members[j][1]
                if used + len(text) > sibling_chars:
                    continue
                keep.append(j); used += len(text)
            if cid in ctx:
                ctx[cid]["siblings"] = [members[j][1] for j in sorted(keep)]
    return ctx


def annotate_candidates(candidates: list, context: dict) -> list:
    """Copies section/lead context onto retrieved candidates (in place)."""
    for c in candidates:
        info = context.get(c.get("internal_clause_id"))
        if info:
            c["clause_section"] = info["section"]
            c["clause_lead"] = info["lead"]
            c["clause_siblings"] = info.get("siblings", [])
    return candidates


def _norm_quote(s: str) -> str:
    s = re.sub(r"[\"'“”‘’]", "", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip(" .;:,")


class ComplianceAnalysis(BaseModel):
    """Pydantic schema for the analysis result."""
    reasoning: str = Field(description="A brief paragraph explaining the overall compliance posture.")
    matched_obligations: List[str] = Field(default_factory=list)
    missing_obligations: List[str] = Field(default_factory=list)
    partial_obligations: list[str] = Field(default_factory=list)
    suggestion: str = Field(default="", description="Actionable remediation recommendation.")
    compliance_level: str = Field(default="Non-Compliant", description="Deterministic label injected by the system.")
    confidence_score: float = Field(default=0.5, description="Deterministic score injected by the system.")
    coverage_ratio: float = Field(default=0.0, description="Percentage of obligations addressed.")
    sub_assessments: list = Field(
        default_factory=list,
        description=(
            "Per-sub-obligation audit trail: for each element of the control, its "
            "status (YES/PARTIAL/NO), the auditor rationale, and the verbatim "
            "quotes the decision rested on. The headline label is a deterministic "
            "function of these, so without them a verdict cannot be checked, "
            "disputed, or defended to an auditor - only believed."
        ),
    )
    evidence_considered: list = Field(
        default_factory=list,
        description=(
            "The clauses that reached the assessor, with their retrieval scores, so "
            "a wrong verdict can be traced to retrieval or to judgement."
        ),
    )
    evidence_status: str = Field(
        default="assessed",
        description=(
            "Why compliance_level has its value. 'assessed' = sub-obligations were "
            "evaluated against retrieved evidence. 'insufficient_evidence' = the "
            "grounding gate fired, i.e. nothing cleared the evidence threshold, so "
            "the control was never assessed. Both map to compliance_level "
            "'Non-Compliant' because both mean 'no compliant evidence was found', "
            "but they call for different follow-up: 'insufficient_evidence' may "
            "indicate a retrieval or segmentation failure rather than a genuine gap, "
            "and should be triaged for manual review before being reported as a gap."
        ),
    )
    subject_check: dict | None = Field(
        default=None,
        description=(
            "VERIFY option: the separate subject check run on a Compliant verdict - "
            "{'addresses_subject': bool, 'reason': str}, or {'error': str}. None when "
            "not run."
        ),
    )
    low_retrieval_confidence: bool = Field(
        default=False,
        description=(
            "True when no candidate cleared the grounding threshold. Under "
            "GATE_MODE=ce such a control is not assessed (evidence_status "
            "'insufficient_evidence'); under GATE_MODE=assess it is assessed anyway "
            "and this flag records that the verdict rests on weakly-scored evidence."
        ),
    )


class LLMAnalyzer:
    """Map-Reduce RAG compliance analyzer.

    Label scheme (`binary_labels`, default True)
    --------------------------------------------
    Binary: "Compliant" / "Not Compliant". This follows ISO audit practice, where
    a finding is a conformity or a nonconformity (major/minor) - "partially
    compliant" is a GRC-tooling convention, not a verdict the standard defines.
    If any required element of a control is unevidenced, that is a nonconformity.

    Granularity is NOT lost by collapsing the label: `coverage_ratio` carries the
    continuous degree of conformance, and `partial_obligations` /
    `missing_obligations` carry exactly which elements are unevidenced - which is
    what a remediation plan needs. "Not Compliant, these 3 of 8 obligations are
    unevidenced" is more actionable than "Partially Compliant, 62%".

    Set `binary_labels=False` for the legacy three-class scheme
    (Compliant / Partially Compliant / Non-Compliant), retained so it can be
    reported as a secondary result rather than quietly dropped.
    """

    GROUNDING_THRESHOLD = 0.15
    FORCED_CE_MINIMUM   = 0.05
 
    # Pipeline v2 options. Every default reproduces the frozen v1 configuration
    # (FROZEN_CONFIG.json) byte-for-byte in its prompts and labels; v2 is opted
    # into per option so each change can be ablated on its own.
    #   CONTEXT_MODE  off | lead | provision
    #                                 lead: quotes reach the assessor with their
    #                                 section and list-introducing sentence;
    #                                 provision: also with the other sentences of
    #                                 the same provision
    #   RUBRIC        v1 | v2         v2 adds the context-reading and Indonesian
    #                                 normative-wording clarifications
    #   GATE_MODE     ce | assess     assess = never skip assessment on a low
    #                                 cross-encoder score; flag it instead
    #   AGGREGATION   threshold | no_absent
    #                                 no_absent = Compliant iff no element is NO and
    #                                 coverage >= NO_ABSENT_MIN_COVERAGE
    NO_ABSENT_MIN_COVERAGE = 0.75
    #   QUOTE_MODE    as_extracted | full_sentence
    #                                 full_sentence: when a quote omits part of its
    #                                 source sentence, the assessor also gets the
    #                                 whole sentence
    #   VERIFY        off | record | apply
    #                                 a separate subject check on every Compliant
    #                                 verdict; 'record' stores it without acting,
    #                                 'apply' turns a failed check into Not Compliant
    _OPTIONS = {"context_mode": ("CONTEXT_MODE", ("off", "lead", "provision"), "off"),
                "rubric": ("RUBRIC", ("v1", "v2"), "v1"),
                "gate_mode": ("GATE_MODE", ("ce", "assess"), "ce"),
                "aggregation": ("AGGREGATION", ("threshold", "no_absent"), "threshold"),
                "quote_mode": ("QUOTE_MODE", ("as_extracted", "full_sentence"), "as_extracted"),
                "verify": ("VERIFY", ("off", "record", "apply"), "off"),
                "verify_mode": ("VERIFY_MODE", ("yesno", "annexa"), "yesno")}

    ANNEX_TITLES_PATH = Path(__file__).resolve().parent / "synthetic_iso27001_uupdp_dataset" / "iso_annex_a_titles.json"
    _annex_titles = None

    ANNEXA_VERIFIER_SYSTEM_PROMPT = (
        "You are an ISO/IEC 27001 Lead Auditor. You are given policy evidence and the full "
        "list of ISO/IEC 27001:2022 Annex A controls. List every control whose specific "
        "requirement this evidence directly establishes - the activity or object the "
        "control is about - not controls the evidence merely mentions, supports or relates "
        "to. Several controls share vocabulary; decide by what the evidence actually "
        "requires the organisation to do. Do not judge how strong or well governed the "
        "evidence is.\n"
        "Output ONLY valid JSON: {\"establishes\": [\"ISO_A_x.y\", ...], \"reason\": \"one sentence\"}"
    )

    VERIFIER_SYSTEM_PROMPT = (
        "You are an ISO/IEC 27001 Lead Auditor performing ONE narrow check. An assessor has "
        "concluded that an organisation's policy establishes an Annex A control. Decide only "
        "whether the evidence the assessor relied on is about THIS control's own subject "
        "matter - the specific activity or object the control requires - or whether it "
        "actually concerns a related but different control that shares some vocabulary.\n"
        "Do NOT re-judge how strong, specific or well governed the evidence is; that has "
        "already been decided. Evidence that addresses the control's subject only in part "
        "still addresses it.\n"
        "Output ONLY valid JSON: {\"addresses_subject\": true or false, \"reason\": \"one sentence\"}"
    )

    RUBRIC_V2_BLOCK = (
        "READING EVIDENCE IN CONTEXT (applies to every quote).\n"
        "A. A quote may be followed by the section it comes from, the sentence that "
        "introduces its list, and the other sentences of the same provision. Read the quote "
        "within that provision, as an auditor reads a clause inside its document: a "
        "responsible role, procedure, cadence or criteria stated in the introducing sentence, "
        "or in another sentence of the same provision ABOUT THE SAME ACTIVITY, applies to "
        "what the quote establishes. Do not mark PARTIAL for lacking an owner, mechanism or "
        "cadence that the provision supplies for that activity. Provisions often bundle "
        "several activities (data masking beside data leakage prevention; reviewing the "
        "access-control policy beside granting access rights): a role or cadence given for "
        "one activity never carries over to another. Context only qualifies what the quote "
        "establishes; it is not evidence for a requirement the provision does not address.\n"
        "B. Indonesian normative wording. 'wajib', 'harus', 'perlu' where it directs an action, "
        "and plain declarative or passive statements of what is done ('dilakukan', "
        "'dilaksanakan', 'ditetapkan', 'disimpan') state obligations, exactly like 'shall'. "
        "Example: 'perlu disediakan pembangkit listrik cadangan dan UPS' means 'a backup "
        "generator and UPS shall be provided' - mandatory, not a reason for PARTIAL. "
        "Statements of view or intent are NOT obligations: 'memandang perlu' / 'dipandang "
        "perlu' (deems it necessary) and 'berupaya' (strives) establish nothing. 'dapat' "
        "(may), 'sebaiknya' (should preferably), 'diupayakan' (best efforts) and 'apabila "
        "diperlukan' (if needed) are discretionary.\n"
        "C. Subject match. First identify the specific activity or object the obligation "
        "requires, then use only evidence about that activity. Evidence for a related but "
        "different control does not establish it, however detailed or well governed. For "
        "example: collecting vulnerability information is not threat intelligence; reviewing "
        "the policy document is not reviewing compliance with it, nor an independent review "
        "of the ISMS; perimeter and entry controls are not the security of offices and rooms "
        "inside it. Where the only evidence concerns a different control, assess NO.\n"
        "These clarify how to read the evidence; they do not relax rule 2b. A quote that "
        "merely restates the control, with no role, mechanism or cadence in the quote or "
        "in its context, is still PARTIAL. An unquantified cadence ('secara berkala', "
        "'periodically', 'regularly') does not by itself satisfy rule 2b; it counts only "
        "together with a named role or a mechanism.\n\n"
    )

    def __init__(self, base_url="http://localhost:11434/v1", model_name=None, api_key=None,
                 binary_labels: bool = True, compliant_threshold: float = 1.0,
                 context_mode: str = None, rubric: str = None, gate_mode: str = None,
                 aggregation: str = None, quote_mode: str = None, verify: str = None,
                 verify_mode: str = None):
        given = {"context_mode": context_mode, "rubric": rubric, "gate_mode": gate_mode,
                 "aggregation": aggregation, "quote_mode": quote_mode, "verify": verify,
                 "verify_mode": verify_mode}
        for name, (env, allowed, default) in self._OPTIONS.items():
            value = given[name] or os.environ.get(env, default)
            if value not in allowed:
                raise ValueError(f"{env}={value!r}; expected one of {allowed}")
            setattr(self, name, value)
        self.binary_labels = binary_labels
        # Coverage a control must reach to be judged Compliant.
        #
        # 1.0 means every sub-obligation must be YES. That is correct ISO
        # semantics - any unmet element is a nonconformity - but it is a
        # CONJUNCTION over a MACHINE-GENERATED decomposition, so it compounds
        # per-element false negatives in proportion to decomposition depth.
        # Measured: disabling expansion flipped 8 verdicts, all of them
        # Not Compliant -> Compliant, 7 of 8 correctly; flipped controls averaged
        # 4.4 sub-obligations against 1.7 for unflipped ones, and both
        # 8-element controls flipped. The expander also splits more finely than a
        # human auditor would (8 elements for A.5.1 where an auditor sees ~3-4),
        # so requiring all 8 is stricter than the ground truth was labelled.
        #
        # Exposed as an explicit parameter with a sensitivity sweep rather than
        # silently tuned, so the ISO-strict setting stays the documented default.
        self.compliant_threshold = compliant_threshold
        if api_key is None:
            if "groq" in base_url:
                api_key = os.environ.get("GROQ_API_KEY", "no_key")
            elif "generativelanguage" in base_url:
                api_key = os.environ.get("GEMINI_API_KEY", "no_key")
            elif "sambanova" in base_url:
                api_key = os.environ.get("SAMBANOVA_API_KEY", "no_key")
            elif "cerebras" in base_url:
                api_key = os.environ.get("CEREBRAS_API_KEY", "no_key")
            elif "together" in base_url:
                api_key = os.environ.get("TOGETHER_API_KEY", "no_key")
            else:
                api_key = "ollama"
 
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        # ------------------------------------------------------------------
        # Reduce-phase system prompt
        # ------------------------------------------------------------------
        self.system_prompt = (
            "You are an expert Information Security Lead Auditor taking a pragmatic, "
            "operational view. Your task is to write the reasoning and suggestion that "
            "match an assessment that has ALREADY been decided.\n"
            "CRITICAL: Do NOT output any '<think>' tags or internal reasoning steps. "
            "Start directly with the JSON object.\n"
        )
 
        # ------------------------------------------------------------------
        # Stage 1 system prompt — Evidence Extraction
        # ------------------------------------------------------------------
        self.extractor_system_prompt = (
            "You are an objective evidence extractor.\n\n"
            "Output ONLY valid JSON.\n\n"
            "Schema:\n"
            "{\n"
            '  "extractions":[\n'
            '    {\n'
            '      "index": integer,\n'
            '      "supporting_quotes": ["exact quote 1", ...],\n'
            '      "contradicting_quotes": ["exact quote 2", ...]\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            "Do not output explanations.\n"
            "Do not output markdown.\n"
            "Do not output prose."
        )
 
        # ------------------------------------------------------------------
        # Stage 2 system prompt — Auditor Assessment
        # ------------------------------------------------------------------
        self.assessor_system_prompt = (
            "You are an ISO 27001 Lead Auditor performing a STAGE 1 "
            "(DOCUMENTATION) audit.\n\n"
            "SCOPE OF THIS ASSESSMENT - read this before applying any rule below.\n"
            "The evidence you are given consists of the organization's POLICY "
            "DOCUMENTS. You are assessing DOCUMENTED CONFORMANCE: whether the "
            "organization's policy set establishes the requirement. You are NOT "
            "assessing operational effectiveness, and you must NOT withhold YES "
            "merely because the policy text does not include proof that the "
            "activity was performed (signed approvals, records, logs, tickets, "
            "screenshots). Such artefacts belong to a Stage 2 audit and are out "
            "of scope here; demanding them would make every requirement "
            "unsatisfiable from a policy corpus, which is the only input you have.\n\n"
            "Concretely: a clear, mandatory policy statement that establishes the "
            "requirement ('X shall be done', 'X must be reviewed annually', "
            "'Review Period: Annually', 'requests for X are handled through the Y "
            "procedure') IS sufficient objective evidence for YES at this stage.\n\n"
            "CRITICAL RULES:\n"
            "1. Do not mark a requirement as YES unless the evidence establishes the requirement. "
            "Vague aspiration with no obligation ('we value security', 'security is important', "
            "'we strive to') is not sufficient. A mandatory, specific policy statement is.\n"
            "2. Distinguish between:\n"
            "   - YES = the policy set establishes the requirement in mandatory, specific terms.\n"
            "   - PARTIAL = the requirement is addressed but only in part - some named elements "
            "of the obligation are covered and others are not, or the wording is discretionary "
            "('may', 'where practical') where the control demands obligation.\n"
            "   - NO = the requirement is absent from the policy set, or the evidence explicitly "
            "contradicts it.\n"
            "   Do NOT use PARTIAL merely because implementation records are not shown - that is "
            "expected at this stage and is not a deficiency in the policy.\n"
            "2b. OPERATIONALIZATION TEST - apply this to every YES before you commit to it.\n"
            "   A policy sentence that merely RESTATES the control in general terms, with no "
            "assigned responsible role or function, no procedure or mechanism, and no frequency "
            "or criteria, is PARTIAL - not YES. It raises the topic without establishing how the "
            "organization actually discharges it.\n"
            "   Examples of PARTIAL (restatement only):\n"
            "     'Test information is selected, protected and managed appropriately.'\n"
            "     'Cables carrying power or data are protected from interception or damage.'\n"
            "     'The company integrates information security into the management of all "
            "projects.'\n"
            "   The same obligations reach YES when the policy names WHO is accountable, or HOW "
            "it is carried out, or WHEN/HOW OFTEN - for example 'The IT Division performs "
            "vulnerability scanning at least once every 3 months and remediates according to "
            "severity.'\n"
            "   This distinguishes a policy that governs from a policy that merely echoes the "
            "standard, which is the difference a Stage 1 auditor is there to find. It is NOT a "
            "demand for implementation records: naming a role, a procedure or a frequency is "
            "still a documented statement.\n"
            "3. Contradicting evidence shall reduce confidence and may override weak supporting evidence.\n"
            "4. Absence of evidence for a requirement component should be treated as missing evidence. "
            "If some components are evidenced and others are not, assess PARTIAL. "
            "If no meaningful evidence exists for the requirement, assess NO.\n"
            "5. Terms such as informal, verbal, ad hoc, case by case, when needed, "
            "when someone notices, not documented, not scheduled, not assigned, "
            "not defined, and not routinely performed indicate implementation weakness.\n"
            "   - Use PARTIAL when the evidence demonstrates the activity exists, but it is "
            "informal, inconsistent, incomplete, or weakly governed.\n"
            "   - Use NO when the activity is explicitly stated not to exist, the requirement "
            "is explicitly stated not to be performed, the evidence states the required element is "
            "not defined/assigned/documented/scheduled/maintained, or the evidence directly "
            "states a condition incompatible with the requirement.\n"
            "6. Assess the requirement itself, not the quality of implementation. If the evidence "
            "demonstrates the required activity exists, but weaknesses are identified, assess "
            "PARTIAL rather than NO. Weak implementation does not automatically mean absence of the control.\n"
            "7. A documented policy, process, role, procedure, review, or approval mechanism may "
            "satisfy part of a requirement even if weaknesses are identified. The presence of "
            "weaknesses should generally reduce the assessment from YES to PARTIAL rather than "
            "from PARTIAL to NO, unless the evidence indicates the requirement is absent.\n"
            "8. The existence of personnel performing a security-related activity constitutes "
            "evidence that the activity exists. Where an activity exists but lacks formal "
            "documentation, assignment, governance, ownership, approval, or consistency, "
            "assess PARTIAL rather than NO. Do not assess NO solely because implementation is informal.\n\n"
            "PARTIAL should be used when:\n"
            "- A policy, process, role, control, or activity exists but is incomplete.\n"
            "- Some named elements of the obligation are covered and others are not.\n"
            "- The policy uses discretionary wording where the control requires obligation.\n"
            "- Evidence is informal or inconsistently applied but the activity clearly exists.\n\n"
            "Do not downgrade an existing control to NO solely because it is weak.\n"
            "Use NO when the control is absent or explicitly contradicted by the evidence.\n\n"
            "WORKED EXAMPLES (calibration - follow these):\n"
            "- Obligation: 'policies shall be reviewed at planned intervals.' "
            "Evidence: 'Review Period: Annually, and additionally after major incidents.' "
            "-> YES. A stated review cadence establishes the requirement; do not demand "
            "minutes of past reviews.\n"
            "- Obligation: 'data subjects may request correction of their personal data.' "
            "Evidence: 'Requests for access, correction, deletion and objection are handled "
            "through the privacy request procedure and tracked to closure.' -> YES. The right "
            "and its handling route are both established; do not demand a fulfilled request log.\n"
            "- Obligation: 'the organization shall maintain contact with relevant authorities.' "
            "Evidence: none in the policy set. -> NO.\n"
            "- Obligation: 'personnel shall be screened prior to employment AND on an ongoing "
            "basis.' Evidence: 'Background checks are performed before employment.' -> PARTIAL. "
            "The pre-employment element is established, the ongoing element is not.\n"
            "- Obligation: 'conflicting duties shall be segregated.' Evidence: 'In a single-person "
            "organization traditional role-based segregation is not possible; instead we implement "
            "compensating controls.' -> PARTIAL, not YES. Where the evidence itself states the "
            "required condition cannot be met and substitutes compensating controls, the objective "
            "is being pursued but the stated requirement is not achieved. Compensating controls "
            "earn PARTIAL; they do not earn YES.\n\n"
            "Output ONLY valid JSON.\n\n"
            "Schema:\n"
            "{\n"
            '  "assessments":[\n'
            '    {\n'
            '      "index": integer,\n'
            '      "status": "YES"|"PARTIAL"|"NO",\n'
            '      "rationale": "Finding-style explanation of the status..."\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            "Do not output explanations outside of the JSON.\n"
            "Do not output markdown."
        )
        if self.rubric == "v2":
            anchor = "Output ONLY valid JSON.\n\n"
            if self.assessor_system_prompt.count(anchor) != 1:
                raise RuntimeError("assessor prompt anchor for the v2 block not found exactly once")
            self.assessor_system_prompt = self.assessor_system_prompt.replace(
                anchor, self.RUBRIC_V2_BLOCK + anchor, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _llm_call(self, system_prompt: str, user_prompt: str,
                  response_format: dict = None, max_retries: int = 6) -> str:
        """Single LLM call with exponential backoff on rate limits.

        Raises LLMCallFailed once retries are exhausted. It must NOT return ""
        on failure: the answer parsers treat an unparseable response as all-NO,
        so a swallowed error silently becomes a confident 'no compliant evidence
        found' verdict for every obligation. That produced 12 fabricated
        Non-Compliant results in an earlier ablation run before this was fixed -
        results that were indistinguishable from real ones.
        """
        kwargs = dict(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            top_p=1.0,
            # Explicit per-request timeout. Without it the SDK default (600s)
            # applies, and with max_retries=6 a single unresponsive endpoint can
            # stall an experiment for the better part of an hour with no output
            # and no error - observed during a variance run, where one request
            # blocked for over ten minutes holding an open socket. A timeout
            # raises a retriable error instead, so the backoff path handles it.
            timeout=180.0,
        )
        if response_format:
            kwargs["response_format"] = response_format

        last_err = None
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(**kwargs)
                raw = completion.choices[0].message.content.strip()
                raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
                if not raw:
                    raise LLMCallFailed("model returned an empty response")
                return raw
            except Exception as e:
                last_err = e
                err = str(e)
                # Connection errors are transient too (observed: four controls in
                # a 20-entry smoke run failed on 'Connection error.' at start-up
                # and were recorded as llm_error without a single retry).
                retriable = ("429" in err or "rate" in err.lower()
                             or "timeout" in err.lower() or "503" in err or "502" in err
                             or "connection" in err.lower())
                if retriable and attempt < max_retries - 1:
                    m = re.search(r'retry.*?(\d+(?:\.\d+)?)s', err, re.IGNORECASE)
                    wait = float(m.group(1)) if m else 2 ** (attempt + 1)
                    wait = min(wait, 60)
                    print(f"    [Rate limit] Waiting {wait:.0f}s "
                          f"(attempt {attempt+1}/{max_retries})", flush=True)
                    time.sleep(wait)   # previously absent: the backoff never slept,
                                       # so all retries burned instantly on a 429
                    continue
                break
        raise LLMCallFailed(f"{self.model_name}: {last_err}")
 
    def _parse_extractor_answers(self, raw, indices):
        result = {i: {"supporting_quotes": [], "contradicting_quotes": []} for i in indices}
        
        repaired = raw.strip()
        try:
            data = json.loads(repaired) if isinstance(repaired, str) else repaired
            for item in data.get("extractions", []):
                idx = int(item.get("index"))
                if idx in result:
                    result[idx]["supporting_quotes"] = item.get("supporting_quotes", [])
                    result[idx]["contradicting_quotes"] = item.get("contradicting_quotes", [])
        except Exception as e:
            print(f"    [LLMAnalyzer] Failed to parse extractor output ({e}): {repaired[:200]}")
        return result

    def _parse_assessor_answers(self, raw, indices):
        result = {i: {"status": "NO", "rationale": "Failed to parse assessor output."} for i in indices}
        
        repaired = raw.strip()
        try:
            data = json.loads(repaired) if isinstance(repaired, str) else repaired
            for item in data.get("assessments", []):
                idx = int(item.get("index"))
                ans = str(item.get("status", "NO")).upper().strip()
                if ans not in ("YES", "PARTIAL", "NO"):
                    ans = "NO"
                if idx in result:
                    result[idx]["status"] = ans
                    result[idx]["rationale"] = str(item.get("rationale", ""))
        except Exception as e:
            print(f"    [LLMAnalyzer] Failed to parse assessor output ({e}): {repaired[:200]}")
        return result

    def _extract_evidence(self, sub_queries: list, combined_clauses: str) -> dict:
        """
        Stage 1: Extracts exact quotes supporting or contradicting the obligations.
        Returns {sq: {"supporting_quotes": [...], "contradicting_quotes": [...]}} dict.
        """
        if not sub_queries:
            return {}
        numbered = "\n".join(f"{i+1}. {sq}" for i, sq in enumerate(sub_queries))
        prompt = (
            f"CLAUSES:\n{combined_clauses}\n\n"
            f"OBLIGATIONS:\n{numbered}\n\n"
            "For each obligation, extract EXACT quotes from the CLAUSES that support OR contradict the obligation.\n"
            "Return empty lists if no evidence exists for that obligation.\n"
            "Do NOT summarize. Extract verbatim."
        )
        raw = self._llm_call(
            system_prompt=self.extractor_system_prompt,
            user_prompt=prompt,
            response_format={"type": "json_object"},
        )
        indices = list(range(1, len(sub_queries) + 1))
        answers = self._parse_extractor_answers(raw, indices)
        return {sub_queries[i-1]: answers[i] for i in indices}

    @staticmethod
    def _locate_quote(quote: str, normed_clauses: list):
        """Finds the clause a verbatim quote was lifted from.

        Exact containment first; otherwise the clause covering >= 80% of the
        quote's words (the extractor occasionally trims punctuation or drops the
        section prefix). Returns None rather than guessing below that.
        """
        q = _norm_quote(quote)
        if len(q) < 12:
            return None
        for cand, text in normed_clauses:
            if q in text:
                return cand
        q_words = set(re.findall(r"\w+", q))
        best, score = None, 0.0
        for cand, text in normed_clauses:
            s = len(q_words & set(re.findall(r"\w+", text))) / len(q_words)
            if s > score:
                best, score = cand, s
        return best if score >= 0.8 else None

    def _with_context(self, quotes: list, normed_clauses: list, shown: set = None) -> list:
        """Appends each quote's context. `shown` collects provisions already
        expanded for this obligation, so several quotes from one provision do
        not repeat it."""
        shown = set() if shown is None else shown
        out = []
        for q in quotes:
            line = q if isinstance(q, str) else str(q)
            src = self._locate_quote(line, normed_clauses) if normed_clauses else None
            if src is not None and self.quote_mode == "full_sentence":
                # The extractor tends to stop at the ':' or ';' where a sentence
                # turns from restating the control to operationalising it
                # ("... is restricted and tightly controlled: use is limited to
                # authorized IT Division personnel, requires a change ticket ...").
                # On dev v2 iteration 1, 11% of located quotes dropped >40 chars of
                # their sentence, and most remaining too-strict errors were such
                # cut-offs. Showing the whole sentence restores what was cut.
                full = _clause_body(src.get("text", ""))
                if len(_norm_quote(full)) - len(_norm_quote(line)) > 20:
                    line += f'\n      [full sentence: "{full}"]'
            if src is not None:
                parts = []
                if src.get("clause_section"):
                    parts.append(f"section: {src['clause_section']}")
                if src.get("clause_lead"):
                    parts.append(f'introduced by: "{src["clause_lead"]}"')
                if parts:
                    line += "\n      [" + "; ".join(parts) + "]"
                if self.context_mode == "provision" and src.get("clause_siblings"):
                    prov = src.get("internal_clause_id", "").rsplit("_Sent_", 1)[0] + "|" + src.get("clause_section", "")
                    if prov in shown:
                        line += "\n      [same provision as quoted above]"
                    else:
                        shown.add(prov)
                        sib = " | ".join(f'"{s}"' for s in src["clause_siblings"])
                        line += f"\n      [rest of the same provision: {sib}]"
            out.append(line)
        return out

    def _assess_compliance(self, sub_queries: list, extractions: dict,
                           map_context: list = None) -> dict:
        """
        Stage 2: Assesses compliance (YES/PARTIAL/NO) and writes finding-style rationale.
        Returns {sq: {"status": "YES"|"PARTIAL"|"NO", "rationale": "..."}} dict.

        Under CONTEXT_MODE=lead each quote carries the section and the
        list-introducing sentence of the clause it came from; under the default
        the prompt is exactly the v1 prompt.
        """
        if not sub_queries:
            return {}
        normed = None
        if self.context_mode in ("lead", "provision") and map_context:
            normed = [(c, _norm_quote(c.get("text", ""))) for c in map_context]
        numbered_parts = []
        for i, sq in enumerate(sub_queries):
            ext = extractions[sq]
            sup_q, con_q = ext["supporting_quotes"], ext["contradicting_quotes"]
            if normed:
                shown = set()
                sup_q = self._with_context(sup_q, normed, shown)
                con_q = self._with_context(con_q, normed, shown)
            sup = "\n    ".join(sup_q) or "None"
            con = "\n    ".join(con_q) or "None"
            numbered_parts.append(
                f"{i+1}. OBLIGATION: {sq}\n"
                f"   SUPPORTING EVIDENCE:\n    {sup}\n"
                f"   CONTRADICTING EVIDENCE:\n    {con}"
            )
        numbered = "\n\n".join(numbered_parts)
        prompt = (
            f"EXTRACTED EVIDENCE:\n{numbered}\n\n"
            "For each obligation, evaluate the status as YES, PARTIAL, or NO based ONLY on the extracted evidence.\n"
            "Write a concise auditor rationale justifying the status."
        )
        if self.rubric == "v2":
            # Rule C as an explicit step: the subject check was skipped when it
            # lived only in the system prompt (a perimeter clause was accepted
            # as office security despite an example naming exactly that pair).
            prompt += ("\nBefore deciding each status, state in the rationale the specific activity "
                       "the obligation requires and which quote establishes THAT activity. Use a role "
                       "or cadence only where it is given for that same activity.")
        raw = self._llm_call(
            system_prompt=self.assessor_system_prompt,
            user_prompt=prompt,
            response_format={"type": "json_object"},
        )
        indices = list(range(1, len(sub_queries) + 1))
        answers = self._parse_assessor_answers(raw, indices)
        return {sub_queries[i-1]: answers[i] for i in indices}
 
    def _verify_subject(self, regulatory_clause: dict, extractions: dict, map_context: list) -> dict:
        """Asks, in a separate narrow call, whether the supporting evidence is about
        this control's own subject rather than a neighbouring control's.

        Motivation: on dev v2 iteration 1, about 8 of 17 consensus too-lenient
        errors accepted evidence for a related control (UPS for redundancy,
        outsourced-development oversight for supplier risk, policy review for
        compliance review) although rubric rule C forbids it - the assessor
        handles many rules at once and drops this one. A failed or unparseable
        check is recorded and never changes the label.
        """
        normed = [(c, _norm_quote(c.get("text", ""))) for c in map_context] if map_context else None
        quotes = []
        for ext in extractions.values():
            sup = ext.get("supporting_quotes", []) or []
            quotes += self._with_context(sup, normed) if normed else [str(q) for q in sup]
        if not quotes:
            return {"addresses_subject": False, "reason": "no supporting evidence was quoted"}
        evidence = "\n".join(f"- {q}" for q in quotes)
        if self.verify_mode == "annexa":
            return self._verify_annexa(regulatory_clause, evidence)
        prompt = (f"CONTROL {regulatory_clause.get('id', '')}: {regulatory_clause['text']}\n\n"
                  f"EVIDENCE THE ASSESSOR RELIED ON:\n{evidence}\n\n"
                  "Is this evidence about this control's own subject matter?")
        try:
            raw = self._llm_call(system_prompt=self.VERIFIER_SYSTEM_PROMPT, user_prompt=prompt,
                                 response_format={"type": "json_object"})
            data = json.loads(raw)
            val = data.get("addresses_subject")
            if not isinstance(val, bool):
                return {"error": f"unparseable verifier output: {raw[:200]}"}
            return {"addresses_subject": val, "reason": str(data.get("reason", ""))[:400]}
        except (LLMCallFailed, json.JSONDecodeError, AttributeError) as e:
            return {"error": str(e)[:200]}

    def _verify_annexa(self, regulatory_clause: dict, evidence: str) -> dict:
        """Contrastive subject check (VERIFY_MODE=annexa).

        Asked "is this about the subject?", the yes/no verifier answered yes for
        neighbouring-control evidence (UPS 'is a form of redundancy'). Here the
        verifier sees all 93 Annex A titles and names the controls the evidence
        directly establishes; the verdict passes iff the target is among them.
        No examples and no tunable neighbour count: the alternatives are the
        whole of Annex A.
        """
        if LLMAnalyzer._annex_titles is None:
            LLMAnalyzer._annex_titles = json.load(open(self.ANNEX_TITLES_PATH, encoding="utf-8"))["titles"]
        titles = LLMAnalyzer._annex_titles
        target = regulatory_clause.get("id", "")
        catalogue = "\n".join(f"{cid}: {t}" for cid, t in titles.items())
        prompt = (f"EVIDENCE:\n{evidence}\n\nANNEX A CONTROLS:\n{catalogue}\n\n"
                  "Which of these controls does this evidence directly establish?")
        try:
            raw = self._llm_call(system_prompt=self.ANNEXA_VERIFIER_SYSTEM_PROMPT, user_prompt=prompt,
                                 response_format={"type": "json_object"})
            data = json.loads(raw)
            est = data.get("establishes")
            if not isinstance(est, list):
                return {"error": f"unparseable verifier output: {raw[:200]}"}
            est = [str(x).strip() for x in est]
            # accept "A.8.14" / "8.14" spellings as well as the ISO_A_ ids
            norm = {("ISO_A_" + re.sub(r"^(ISO_)?A\.?_?", "", x)) for x in est}
            return {"addresses_subject": target in norm, "establishes": sorted(norm)[:15],
                    "reason": str(data.get("reason", ""))[:400], "mode": "annexa"}
        except (LLMCallFailed, json.JSONDecodeError, AttributeError) as e:
            return {"error": str(e)[:200], "mode": "annexa"}

    def _conforms(self, no_count: int, coverage_ratio: float) -> bool:
        """The conformity rule over sub-obligation votes.

        'threshold' (default): coverage >= compliant_threshold - at 1.0 every
        element must be YES. 'no_absent': no element may be absent (NO) and
        coverage must reach NO_ABSENT_MIN_COVERAGE, so a minority of
        partially-stated elements is tolerated but a missing one never is. The
        motivation is that the conjunction runs over a MACHINE decomposition
        that splits more finely than an auditor does (8 elements for A.5.1).
        """
        if self.aggregation == "no_absent":
            return no_count == 0 and coverage_ratio >= self.NO_ABSENT_MIN_COVERAGE
        return coverage_ratio >= self.compliant_threshold

    def _compute_confidence(self, label: str, coverage_ratio: float) -> float:
        if label == "Compliant":
            return round(min(0.95, 0.75 + 0.20 * coverage_ratio), 3)
        if label == "Partially Compliant":
            return round(min(0.90, max(0.55, coverage_ratio)), 3)
        if label in ("Non-Compliant", "Not Compliant"):
            # Under the binary scheme a nonconformity spans everything from
            # "one element short" to "nothing at all", so confidence tracks how
            # far from conformity the control actually is rather than being flat.
            if self.binary_labels:
                return round(max(0.55, 0.90 - 0.35 * coverage_ratio), 3)
            return 0.65
        return 0.30
 
    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
 
    def analyze_gap(self, regulatory_clause: dict, internal_candidates: list,
                    sub_queries: list = None):
 
        if not sub_queries:
            sub_queries = [regulatory_clause["text"]]
 
        # ----------------------------------------------------------------
        # 0. GROUNDING GATE
        # ----------------------------------------------------------------
        max_ce_score = max(
            (c.get("cross_encoder_score", 0.0) for c in internal_candidates),
            default=0.0,
        )
        has_forced = any(
            "forced_by_keyword" in c
            and c.get("cross_encoder_score", 0.0) > self.FORCED_CE_MINIMUM
            for c in internal_candidates
        )
        low_confidence = max_ce_score < self.GROUNDING_THRESHOLD and not has_forced

        # GATE_MODE=assess keeps the signal but stops it deciding the verdict.
        # The cross-encoder's absolute scale does not transfer across document
        # styles: on OCR'd corporate text it scored plainly relevant clauses at
        # 0.01-0.08 ("berpartisipasi dalam keanggotaan forum Keamanan Siber" for
        # A.5.6), so the gate reported 11 controls as unevidenced without ever
        # assessing them - the same degradation-into-confident-verdict defect
        # METHODOLOGY_REVISED.md records for the English-only reranker. Under
        # 'assess' the assessor still has to find quoted evidence for every
        # element, so weak retrieval cannot by itself produce a YES.
        if low_confidence and (self.gate_mode == "ce" or not internal_candidates):
            return ComplianceAnalysis(
                reasoning=(
                    "No policy clause scored above the evidence threshold for this "
                    "control. This may mean the control is genuinely unaddressed, or "
                    "that retrieval/segmentation failed to surface the relevant clause."
                ),
                suggestion=(
                    "Manual review recommended before treating this as a gap. If the "
                    "control is truly unaddressed, draft an explicit policy section for it."
                ),
                # A nonconformity label rather than a separate "Insufficient
                # Evidence" class: both mean "no compliant evidence found", and
                # emitting a class the ground truth never uses made every gated
                # control an automatic scoring miss. The distinction is preserved
                # in evidence_status instead.
                compliance_level=("Not Compliant" if self.binary_labels else "Non-Compliant"),
                evidence_status="insufficient_evidence",
                confidence_score=0.30,
                matched_obligations=[],
                partial_obligations=[],
                missing_obligations=sub_queries,
                low_retrieval_confidence=True,
            )
 
        # ----------------------------------------------------------------
        # Context ordering
        # ----------------------------------------------------------------
        sorted_cands = sorted(
            internal_candidates,
            key=lambda x: x.get("cross_encoder_score", 0.0),
            reverse=True,
        )
        # Truncating to a fixed 11 by CONTROL-level score would undo the
        # reranker's per-sub-obligation evidence recovery: a clause kept
        # specifically because it answers sub-obligation #7 usually scores
        # mid-range against the control as a whole and would be cut here.
        # Keep those explicitly, and let the budget grow with the number of
        # obligations the assessor must actually answer.
        base_budget  = max(11, 3 * len(sub_queries))
        map_context  = sorted_cands[:base_budget]
        in_ctx_ids   = {c["internal_clause_id"] for c in map_context}
        must_keep    = [c for c in internal_candidates
                        if c["internal_clause_id"] not in in_ctx_ids
                        and ("forced_by_keyword" in c
                             or "recovered_for_sub_obligation" in c)]
        map_context  = map_context + must_keep
 
        context_parts = []
        for i, cand in enumerate(map_context):
            forced_tag = (
                f" [FORCED:{cand.get('forced_by_keyword', '')}]"
                if "forced_by_keyword" in cand else ""
            )
            context_parts.append(
                f"Clause {i+1} [{cand['internal_clause_id']}]{forced_tag}: {cand['text']}"
            )
        combined_clauses = "\n\n".join(context_parts)
 
        # ----------------------------------------------------------------
        # 1. MAP PHASE — Two-Stage Auditor Pipeline
        #    Stage 1: Extract evidence (supporting & contradicting)
        #    Stage 2: Assess compliance (YES/PARTIAL/NO)
        # ----------------------------------------------------------------
        try:
            extractions = self._extract_evidence(sub_queries, combined_clauses)
            assessments = self._assess_compliance(sub_queries, extractions, map_context)
        except LLMCallFailed as e:
            # A transport failure is NOT a finding. Return it as an explicit
            # error state so callers can exclude it from scoring instead of
            # recording a fabricated nonconformity.
            print(f"    [{regulatory_clause.get('id','N/A')}] LLM ERROR: {e}")
            return ComplianceAnalysis(
                reasoning=f"Assessment could not be completed: {e}",
                suggestion="Re-run this control; the result below is not a finding.",
                compliance_level=("Not Compliant" if self.binary_labels else "Non-Compliant"),
                evidence_status="llm_error",
                confidence_score=0.0,
                matched_obligations=[],
                partial_obligations=[],
                missing_obligations=[],
                low_retrieval_confidence=low_confidence,
            )
        
        matched_obs = []
        partial_obs = []
        missing_obs = []
        
        for sq, assessment in assessments.items():
            status = assessment["status"]
            rationale = assessment["rationale"]
            formatted_sq = f"{sq}\n   Auditor Finding: {rationale}"
            
            if status == "YES":
                matched_obs.append(formatted_sq)
            elif status == "PARTIAL":
                partial_obs.append(formatted_sq)
            else:
                missing_obs.append(formatted_sq)
 
        total_sq       = len(sub_queries)
        yes_count      = len(matched_obs)
        partial_count  = len(partial_obs)
        no_count       = len(missing_obs)
        
        print(f"    [{regulatory_clause.get('id', 'N/A')}] Sub-assessments: YES={yes_count}, PARTIAL={partial_count}, NO={no_count}, TOTAL={total_sq}")
        
        coverage_ratio = (yes_count + 0.5 * partial_count) / total_sq if total_sq else 0.0

        # Structured audit trail. The label is a deterministic function of these
        # votes, so recording only the label makes a verdict unverifiable: a
        # reviewer cannot tell a retrieval failure from a judgement failure, and
        # an auditor cannot see which sentence a finding rests on.
        trail = []
        for sq, a in assessments.items():
            ex = extractions.get(sq, {}) or {}
            trail.append({
                "obligation": sq,
                "status": a.get("status", "NO"),
                "rationale": a.get("rationale", ""),
                "supporting_quotes": list(ex.get("supporting_quotes", []) or []),
                "contradicting_quotes": list(ex.get("contradicting_quotes", []) or []),
            })
        considered = [{
            "clause_id": c.get("internal_clause_id"),
            "dense_score": round(float(c.get("similarity_score", 0.0)), 4),
            "cross_encoder_score": round(float(c.get("cross_encoder_score", 0.0)), 4),
            "forced_by_keyword": "forced_by_keyword" in c,
            "text": c.get("text", "")[:400],
            **({"lead": c["clause_lead"]} if c.get("clause_lead") else {}),
        } for c in map_context]
 
        evidence_status = "assessed"
        not_compliant = "Not Compliant" if self.binary_labels else "Non-Compliant"
        if total_sq == 0:
            deterministic_label = not_compliant
            evidence_status = "insufficient_evidence"
        elif self._conforms(no_count, coverage_ratio):
            # Default: coverage >= threshold; at 1.0 "every element evidenced".
            deterministic_label = "Compliant"
        elif self.binary_labels:
            # Any unevidenced element is a nonconformity under ISO audit practice.
            # WHICH elements are unevidenced is carried by partial_obligations /
            # missing_obligations and by coverage_ratio, so nothing is lost.
            deterministic_label = not_compliant
        elif yes_count > 0 or partial_count > 0:
            deterministic_label = "Partially Compliant"
        else:
            deterministic_label = not_compliant
 
        # Optional subject check on Compliant verdicts (VERIFY). Run AFTER the
        # deterministic label so 'record' leaves every label untouched and the
        # 'apply' variant can be recomputed exactly from a 'record' run.
        subject_check = None
        if self.verify != "off" and deterministic_label == "Compliant":
            subject_check = self._verify_subject(regulatory_clause, extractions, map_context)
            if self.verify == "apply" and subject_check.get("addresses_subject") is False:
                deterministic_label = not_compliant

        confidence = self._compute_confidence(deterministic_label, coverage_ratio)
 
        # ----------------------------------------------------------------
        # 3. REDUCE PHASE — prose only (1 API call)
        # ----------------------------------------------------------------
        reduce_prompt = (
            f"REGULATORY CONTROL:\n{regulatory_clause['text']}\n\n"
            f"The assessment has already been decided:\n"
            f"  Final Assessment      : {deterministic_label}\n"
            f"  Matched Obligations   : {matched_obs}\n"
            f"  Partial Obligations   : {partial_obs}\n"
            f"  Missing Obligations   : {missing_obs}\n\n"
            "Write ONLY the 'reasoning' and 'suggestion' paragraphs.\n"
            "Your reasoning MUST align with the Final Assessment and the lists above.\n"
            "- If Compliant          : explain how the policy meets all obligations; "
            "do not invent gaps.\n"
            "- If Partially Compliant: acknowledge what is present (matched + partial) "
            "and what is missing; focus suggestion on closing the gaps.\n"
            "- If Non-Compliant      : focus suggestion on the missing obligations.\n"
        )
 
        try:
            schema_json = ComplianceAnalysis.model_json_schema()
            # low_retrieval_confidence is popped too, so the schema the model sees
            # is byte-identical to v1's (it is system-injected, like the others).
            for key in ["compliance_level", "confidence_score",
                        "matched_obligations", "partial_obligations",
                        "missing_obligations", "low_retrieval_confidence",
                        "subject_check"]:
                schema_json["properties"].pop(key, None)
                if key in schema_json.get("required", []):
                    schema_json["required"].remove(key)
 
            json_instructions = (
                "\n\nReturn a raw JSON object adhering to this schema:\n"
                + json.dumps(schema_json)
            )
 
            raw = self._llm_call(
                system_prompt=self.system_prompt + json_instructions,
                user_prompt=reduce_prompt,
                response_format={"type": "json_object"},
            )
 
            parsed = json.loads(raw)
            parsed["compliance_level"]    = deterministic_label
            parsed["evidence_status"]     = evidence_status
            parsed["confidence_score"]    = confidence
            parsed["coverage_ratio"]      = coverage_ratio
            parsed["matched_obligations"] = matched_obs
            parsed["partial_obligations"] = partial_obs
            parsed["missing_obligations"] = missing_obs
            parsed["sub_assessments"]     = trail
            parsed["evidence_considered"] = considered
            parsed["low_retrieval_confidence"] = low_confidence
            parsed["subject_check"] = subject_check
            return ComplianceAnalysis(**parsed)
 
        except Exception as e:
            print(f"  [LLMAnalyzer] Reduce phase failed for "
                  f"'{regulatory_clause['id']}': {e}")
            return ComplianceAnalysis(
                reasoning="Reduce-phase synthesis failed; label derived from map-phase votes.",
                suggestion="Review the partial and missing obligations listed.",
                compliance_level=deterministic_label,
                evidence_status=evidence_status,
                confidence_score=confidence,
                coverage_ratio=coverage_ratio,
                matched_obligations=matched_obs,
                partial_obligations=partial_obs,
                missing_obligations=missing_obs,
                sub_assessments=trail,
                evidence_considered=considered,
                low_retrieval_confidence=low_confidence,
                subject_check=subject_check,
            )


