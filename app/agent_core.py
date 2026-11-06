"""
agent_core.py - the procurement multi-agent system as ONE importable module.

Used by:
  * 11_demo  (notebook with an input box)
  * app.py   (Databricks App chat UI)

Design rules:
  * No Spark here. Everything goes through the Databricks SDK (SQL warehouse,
    Vector Search API, serving endpoints), so it runs in a notebook AND in an App.
  * Facts come from tools (SQL, search, model); the LLM only plans and writes.
  * A deterministic verifier checks every number and citation (fails closed).
  * Every request is appended to procintel.ops.agent_requests (application log).
"""
import datetime
import functools
import json
import math
import os
import re
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem

# MLflow tracing if available (notebook, or App with mlflow installed); otherwise a no-op
try:
    import mlflow
    trace = mlflow.trace
except Exception:                                   # pragma: no cover
    mlflow = None
    def trace(func=None, **kwargs):
        return (lambda f: f) if func is None else func

# ---------------------------------------------------------------------------
# CONFIG (environment variables let the App override defaults without code changes)
# ---------------------------------------------------------------------------
LLM               = os.getenv("PROCINTEL_LLM", "databricks-meta-llama-3-3-70b-instruct")
FORECAST_ENDPOINT = os.getenv("PROCINTEL_FORECAST_ENDPOINT", "procintel-tender-forecast")
INDEX             = os.getenv("PROCINTEL_INDEX", "procintel.ml.rag_focus_index")

w      = WorkspaceClient()                          # notebook: your identity | App: the app's service principal
client = w.serving_endpoints.get_open_ai_client()   # OpenAI-format client for Databricks LLM endpoints


@functools.lru_cache(maxsize=1)
def warehouse_id():
    """The SQL warehouse to run queries on (env var in the App, first warehouse otherwise)."""
    return os.getenv("DATABRICKS_WAREHOUSE_ID") or next(wh.id for wh in w.warehouses.list())


# ---------------------------------------------------------------------------
# SQL helper (Statement Execution API) - works without Spark
# ---------------------------------------------------------------------------
def run_sql(sql, params=None):
    """Run one SQL statement on the warehouse. params = {"name": value} for :name markers."""
    parameters = [StatementParameterListItem(name=k, value=None if v is None else str(v))
                  for k, v in (params or {}).items()] or None
    r = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id(), statement=sql, parameters=parameters, wait_timeout="30s")
    while r.status.state.value in ("PENDING", "RUNNING"):          # long query -> poll
        time.sleep(1)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(r.status.error.message if r.status.error else r.status.state.value)
    if not (r.manifest and r.manifest.schema and r.manifest.schema.columns):
        return []                                                   # e.g. INSERT
    cols = [c.name for c in r.manifest.schema.columns]
    data = (r.result.data_array if r.result else None) or []
    return [dict(zip(cols, row)) for row in data]


# ---------------------------------------------------------------------------
# DATA AGENT: question -> one safe SELECT on Gold -> rows
# ---------------------------------------------------------------------------
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|GRANT|REVOKE|MERGE|TRUNCATE)\b", re.I)


@functools.lru_cache(maxsize=1)
def schema_doc():
    """Gold table descriptions read FROM UNITY CATALOG, so the agent never sees a stale schema."""
    rows = run_sql("""
      SELECT t.table_name, t.comment, concat_ws(', ', collect_list(c.column_name)) AS cols
      FROM procintel.information_schema.tables t
      JOIN procintel.information_schema.columns c
        ON c.table_schema = t.table_schema AND c.table_name = t.table_name
      WHERE t.table_schema = 'gold' AND t.table_name IN ('project_kpis','supplier_kpis','tender_training')
      GROUP BY t.table_name, t.comment""")
    return "\n".join(f"- procintel.gold.{r['table_name']}: {r['comment']}\n  columns: {r['cols']}"
                     for r in rows)


def check_sql(sql):
    """Code-level guardrail. Returns an error message, or None if the SQL is safe."""
    s = sql.strip().rstrip(";")
    if ";" in s:                                    return "multiple statements are not allowed"
    if not re.match(r"^(SELECT|WITH)\b", s, re.I):  return "only SELECT queries are allowed"
    if FORBIDDEN.search(s):                         return "write/DDL keywords are not allowed"
    if any(t.lower() != "gold" for t in re.findall(r"procintel\.(\w+)\.\w+", s, re.I)):
        return "only procintel.gold tables are allowed"
    return None


@trace(name="data_agent")
def data_agent(question, max_attempts=2):
    system = ("You write ONE Databricks SQL SELECT query to answer the question.\n"
              f"Use ONLY these tables (full names):\n{schema_doc()}\n"
              "Rules: SELECT only, LIMIT 50, round money to millions where helpful, "
              "tender_training: filter WHERE is_valid_label. Return ONLY the SQL in a ```sql block.")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    sql, problem = "", None
    for attempt in range(1, max_attempts + 1):
        reply = client.chat.completions.create(model=LLM, messages=messages, temperature=0,
                                               max_tokens=500).choices[0].message.content
        m = re.search(r"```sql\s*(.*?)```", reply, re.S)
        sql = (m.group(1) if m else reply).strip().rstrip(";")
        problem = check_sql(sql)
        if problem is None:
            try:
                return {"agent": "data", "sql": sql, "rows": run_sql(sql)[:50],
                        "error": None, "attempts": attempt}
            except Exception as error:
                problem = f"SQL error: {str(error)[:300]}"
        messages += [{"role": "assistant", "content": reply},                 # SELF-CORRECTION
                     {"role": "user", "content": f"That failed: {problem}. Fix it and return only the SQL."}]
    return {"agent": "data", "sql": sql, "rows": [], "error": problem, "attempts": max_attempts}


# ---------------------------------------------------------------------------
# DOCUMENTS AGENT: hybrid search on the Vector Search index (via the SDK)
# ---------------------------------------------------------------------------
DOC_COLUMNS = ["chunk_id", "doc_family", "authority", "title", "doc_date", "section", "project_id", "text"]


@trace(name="documents_agent")
def documents_agent(query, doc_families=None, project_id=None, k=6):
    filters = {}
    if doc_families: filters["doc_family"] = doc_families
    if project_id:   filters["project_id"] = project_id
    res = w.vector_search_indexes.query_index(
        index_name=INDEX, columns=DOC_COLUMNS, query_text=query, query_type="HYBRID",
        num_results=k, filters_json=json.dumps(filters) if filters else None)
    names = [c.name for c in res.manifest.columns]
    passages = []
    for row in (res.result.data_array or []):
        rec = dict(zip(names, row))
        rec["text"] = str(rec.get("text", ""))[:900]
        passages.append(rec)
    return {"agent": "documents", "query": query, "passages": passages}


# ---------------------------------------------------------------------------
# FORECAST AGENT: description -> features (LLM) -> prediction (served MLflow model)
# ---------------------------------------------------------------------------
FEATURES_CAT = ["procurement_group_code", "procurement_method", "review_type",
                "major_sector", "first_notice_type", "env_category"]


@functools.lru_cache(maxsize=1)
def allowed_values():
    """Categories the model was trained on, read from the data (LLM may only pick these)."""
    return {c: [r[c] for r in run_sql(
        f"SELECT DISTINCT {c} FROM procintel.gold.tender_training WHERE is_valid_label AND {c} IS NOT NULL")]
        for c in FEATURES_CAT}


@functools.lru_cache(maxsize=1)
def latest_governance():
    """Latest governance scores = what a planner knows today."""
    row = run_sql("""SELECT cpia_public_sector, wgi_gov_effectiveness, wgi_regulatory_quality,
                            wgi_control_corruption
                     FROM procintel.gold.tender_training ORDER BY first_notice_date DESC LIMIT 1""")[0]
    return {k: float(v) for k, v in row.items()}


@trace(name="forecast_agent")
def forecast_agent(description):
    allowed = allowed_values()
    system = ("Extract procurement features as JSON with keys: " + ", ".join(FEATURES_CAT) +
              ", amount_usd (number), months_since_approval (number or null).\n"
              "Use ONLY these allowed values (or null if unknown):\n" + json.dumps(allowed) +
              "\nReturn only the JSON object.")
    reply = client.chat.completions.create(model=LLM, temperature=0, max_tokens=400, messages=[
        {"role": "system", "content": system}, {"role": "user", "content": description}]
    ).choices[0].message.content
    feats = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))

    assumptions = []                                                 # be transparent about defaults
    for c in FEATURES_CAT:
        if feats.get(c) not in allowed[c]:
            feats[c] = "UNKNOWN"
            assumptions.append(f"{c} unknown")
    amount = feats.get("amount_usd")
    if not amount:
        amount = 1_000_000
        assumptions.append("amount assumed $1M")
    msa = feats.get("months_since_approval")
    if msa is None:
        msa = 18.0
        assumptions.append("months_since_approval assumed 18")

    record = {**{c: feats[c] for c in FEATURES_CAT},
              "log10_amount_usd": math.log10(max(float(amount), 1.0)),
              "notice_month": float(datetime.date.today().month),
              "months_since_approval": float(msa),
              **latest_governance()}
    pred = w.serving_endpoints.query(name=FORECAST_ENDPOINT, dataframe_records=[record]).predictions[0]
    return {"agent": "forecast", "features": record, "assumptions": assumptions,
            "p50_days": pred["p50_days"], "p90_days": pred["p90_days"],
            "model": "procintel.ml.tender_duration@champion"}


# ---------------------------------------------------------------------------
# SUPERVISOR: LLM tool-calling loop (plan -> act -> observe -> repeat)
# ---------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "query_data",
        "description": "Facts and numbers from Kenya procurement tables: projects, contracts, suppliers, "
                       "tender durations. Use for counts, amounts, rankings, medians.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search World Bank evaluation reports (ICR, IEG reviews, PAD lessons, ISR) for "
                       "reasons, lessons learned and risks. Returns passages with chunk_id citations.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "doc_families": {"type": "array", "items": {"type": "string",
                             "enum": ["ICR", "IEG_ICRR", "IEG_PPAR", "PAD", "ISR"]}},
            "project_id": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "forecast_tender",
        "description": "Predict tender duration (P50 typical days, P90 safe-to-commit days) for a planned "
                       "procurement package described in words.",
        "parameters": {"type": "object", "properties": {"description": {"type": "string"}},
                       "required": ["description"]}}},
]

TOOL_FUNCS = {
    "query_data":       lambda question: data_agent(question),
    "search_documents": lambda query, doc_families=None, project_id=None:
                            documents_agent(query, doc_families, project_id),
    "forecast_tender":  lambda description: forecast_agent(description),
}

SUPERVISOR_PROMPT = """You are a procurement intelligence assistant for World Bank-financed projects in Kenya.
Use the tools to gather evidence, then answer.
RULES:
- SCOPE: only answer questions about World Bank-financed procurement in Kenya.
  For anything else (e.g., GDP, elections, general knowledge), say it is outside your scope and call NO tools.
- Every number must come from a tool result. Never invent numbers.
- Never calculate, convert or combine numbers (no sums, no days-to-months). Report numbers exactly as the tools return them.
- Call each tool at most once per distinct request; reuse earlier results instead of repeating a call.
- Cite DOCUMENT passages with their chunk_id only, one per bracket, e.g. [5594920_0045].
  Numbers from query_data or forecast_tender need no brackets; instead say where they come from
  (e.g. "according to the contract data" or "the model forecasts").
- If sources disagree, prefer higher authority (IEG 0.9 > ICR 0.75 > PAD 0.6 > ISR 0.5) and say so.
- For forecasts, give P50 and P90 and mention any assumptions.
- If the evidence is insufficient, say so plainly."""

# ---------------------------------------------------------------------------
# VERIFIER: deterministic, no LLM, fails closed
# ---------------------------------------------------------------------------
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def numbers_in(text):
    return [float(x.replace(",", "")) for x in NUM.findall(text or "")]


@trace(name="verifier")
def verify(answer, tool_results, question):
    evidence = set(numbers_in(question + " " + json.dumps(tool_results, default=str)))
    evidence |= {x / 1e6 for x in evidence if abs(x) > 1e5} | {x / 1e9 for x in evidence if abs(x) > 1e8}
    retrieved = {p["chunk_id"] for r in tool_results if r.get("agent") == "documents"
                 for p in r.get("passages", [])}
    used_documents = any(r.get("agent") == "documents" for r in tool_results)
    cited = set(re.findall(r"\b(\d+_\d{4})\b", answer or ""))
    clean = re.sub(r"\b\d+_\d{4}\b|\bP\d{6}\b|\[[^\]]*\]", " ", answer or "")
    unsupported = [n for n in numbers_in(clean)
                   if not (abs(n) <= 10 or 1900 <= n <= 2100)
                   and not any(abs(n - e) <= max(1, 0.01 * abs(e)) for e in evidence)]
    bad = sorted(cited - retrieved)
    missing = used_documents and not cited
    ok = not unsupported and not bad and not missing
    return {"verdict": "PASS" if ok else "FLAG", "unsupported_numbers": unsupported,
            "bad_citations": bad, "missing_citations": missing,
            "n_citations": len(cited), "n_verified": len(cited & retrieved)}


@trace(name="supervisor")
def _supervise(question, max_steps=5):
    messages = [{"role": "system", "content": SUPERVISOR_PROMPT},
                {"role": "user", "content": question}]
    tool_results, steps, answer = [], [], None
    for step in range(max_steps):
        msg = client.chat.completions.create(model=LLM, messages=messages, tools=TOOLS,
                                             temperature=0, max_tokens=1200).choices[0].message
        if not msg.tool_calls:
            answer = msg.content
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            try:
                result = TOOL_FUNCS[tc.function.name](**args)
            except Exception as error:
                result = {"agent": tc.function.name, "error": str(error)[:300]}
            tool_results.append(result)
            steps.append({"step": step, "tool": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, default=str)[:6000]})
    if answer is None:
        answer = "I could not finish within the step limit."
    sources = [{"chunk_id": p["chunk_id"], "doc_family": p.get("doc_family"), "title": p.get("title"),
                "section": p.get("section"), "doc_date": p.get("doc_date")}
               for r in tool_results if r.get("agent") == "documents" for p in r.get("passages", [])]
    return {"question": question, "answer": answer, "trace": steps, "tool_results": tool_results,
            "sources": sources, "verification": verify(answer, tool_results, question)}


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT: ask() = supervise + time + log
# ---------------------------------------------------------------------------
def log_request(user, out):
    """Append one row to the application log (same table for notebook and App)."""
    v = out["verification"]
    run_sql("""
      INSERT INTO procintel.ops.agent_requests
        (request_time, `user`, question, agents_used, verdict, n_citations, n_verified,
         unsupported_numbers, latency_ms, trace_id, answer)
      VALUES (current_timestamp(), :user, :question, :agents, :verdict,
              CAST(:nc AS BIGINT), CAST(:nv AS BIGINT), :unsup, CAST(:lat AS BIGINT), :trace_id, :answer)""",
            {"user": user, "question": out["question"],
             "agents": ",".join(t["tool"] for t in out["trace"]),
             "verdict": v["verdict"], "nc": v["n_citations"], "nv": v["n_verified"],
             "unsup": str(v["unsupported_numbers"]), "lat": out["latency_ms"],
             "trace_id": out.get("trace_id"), "answer": out["answer"]})


def ask(question, user="unknown", log=True):
    t0 = time.time()
    out = _supervise(question)
    out["latency_ms"] = int((time.time() - t0) * 1000)
    out["trace_id"] = None
    if mlflow is not None:
        try:
            out["trace_id"] = mlflow.get_last_active_trace_id()
        except Exception:
            pass
    if log:
        try:
            log_request(user, out)
        except Exception as error:                 # logging must never break the answer
            out["log_error"] = str(error)[:200]
    return out