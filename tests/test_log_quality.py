#!/usr/bin/env python3
"""
Log quality diagnostic script (not an assert-based test)

Calls the real LLM, running a lite+agent add and a full+agent add,
printing separators plus request_id-filtered logs for manual quality review.

Usage:
    python3 tests/test_log_quality.py
"""

import logging
import os
import sys

# make sure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    """Run the log quality diagnostics."""
    # -- read API config from the environment; the diagnostic script never carries or echoes credentials. --
    llm_api_key = os.environ.get("MEMORY_LLM_API_KEY")
    embedding_api_key = os.environ.get("MEMORY_EMBEDDER_API_KEY")
    if not llm_api_key or not embedding_api_key:
        raise SystemExit(
            "Set MEMORY_LLM_API_KEY and MEMORY_EMBEDDER_API_KEY before running "
            "this live diagnostic."
        )
    llm_model = os.environ.get("MEMORY_LLM_MODEL", "deepseek-chat")
    llm_base_url = os.environ.get(
        "MEMORY_LLM_BASE_URL",
        "https://api.deepseek.com",
    )
    embedding_model = os.environ.get(
        "MEMORY_EMBEDDER_MODEL",
        "text-embedding-v4",
    )
    embedding_base_url = os.environ.get(
        "MEMORY_EMBEDDER_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    embedding_dims = int(os.environ.get("MEMORY_EMBEDDING_DIMS", "1024"))

    # -- configure logging --
    from hy_memory.utils.log_setup import setup_logging, RequestIdFilter
    setup_logging(level=logging.DEBUG)

    hy_logger = logging.getLogger("hy_memory")

    # drop the stderr StreamHandler (keep the FileHandler) to avoid duplicate output
    for h in list(hy_logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            hy_logger.removeHandler(h)

    # format with exactly the same formatter as hy_memory.log
    _LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s"
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    log_formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # capture all hy_memory logs into a list
    captured_logs: list = []

    class LogCapture(logging.Handler):
        def emit(self, record):
            captured_logs.append(record)

    capture_handler = LogCapture()
    capture_handler.setLevel(logging.DEBUG)
    capture_handler.addFilter(RequestIdFilter())
    hy_logger.addHandler(capture_handler)

    # -- helper functions --

    def pr(msg=""):
        print(msg, flush=True)

    def print_separator(title: str):
        pr()
        pr("=" * 80)
        pr(f"  {title}")
        pr("=" * 80)

    def print_filtered_logs(request_id: str, level_filter: str = "ALL"):
        """Print the logs of one request_id in exactly the hy_memory.log format"""
        filtered = [
            r for r in captured_logs
            if getattr(r, 'request_id', '-') == request_id
        ]
        if level_filter == "INFO+":
            filtered = [r for r in filtered if r.levelno >= logging.INFO]

        pr(f"\n--- request_id={request_id} ({level_filter}, {len(filtered)} lines) ---")
        for record in filtered:
            pr(log_formatter.format(record))
        pr(f"--- End ({len(filtered)} lines) ---")

    def count_logs(request_id: str):
        matched = [r for r in captured_logs if getattr(r, 'request_id', '-') == request_id]
        info_plus = [r for r in matched if r.levelno >= logging.INFO]
        debug_only = [r for r in matched if r.levelno == logging.DEBUG]
        return len(info_plus), len(debug_only)

    def wait_s2_tasks(client_obj, request_id: str, timeout: int = 120):
        """S2 is synchronous now (done when add returns), so there are no background tasks to await."""
        return {"status": "completed", "tasks": []}

    # ================================================================
    # initialize the client
    # ================================================================
    print_separator("Initializing HyMemoryClient")

    from hy_memory.client import HyMemoryClient
    client = HyMemoryClient.from_config({
        "llm": {
            "model": llm_model,
            "api_key": llm_api_key,
            "base_url": llm_base_url,
        },
        "embedder": {
            "model": embedding_model,
            "api_key": embedding_api_key,
            "base_url": embedding_base_url,
            "embedding_dims": embedding_dims,
        },
        "vector_store": {
            "provider": "chroma",
            "embedding_dims": embedding_dims,
        },
        "enable_graph": False,
        "enable_agent": True,
    })

    # print the initialization-phase logs
    init_info = [r for r in captured_logs
                 if getattr(r, 'request_id', '-') == '-' and r.levelno >= logging.INFO]
    init_debug = [r for r in captured_logs
                  if getattr(r, 'request_id', '-') == '-' and r.levelno == logging.DEBUG]
    pr(f"\nInit phase: {len(init_info)} INFO+ lines, {len(init_debug)} DEBUG lines")
    pr("\n--- Init INFO+ logs ---")
    for record in init_info:
        pr(log_formatter.format(record))
    pr(f"--- End ({len(init_info)} lines) ---")

    # -- clear old data --
    pr("\nCleaning up old test data...")
    client.delete_all(user_id="test_log_user")

    # ================================================================
    # Test 1: Lite + Agent add
    # ================================================================
    print_separator("Test 1: Lite + Agent add")

    test_content = (
        "My name is Zhang San, I am 28 years old, and I work as a backend engineer at ByteDance in Beijing."
        "I especially love pour-over Ethiopian Yirgacheffe coffee,"
        "and I often visit independent cafes in Wudaokou on weekends."
        "Lately I have been learning Rust, planning to rewrite an internal tool with it."
    )

    captured_logs.clear()
    result = client.add(
        test_content,
        user_id="test_log_user",
        agent_id="test_agent",
        session_id="test_session",
        pipeline="lite",
        enable_agent=True,
    )
    rid = result.get("request_id", "")

    pr(f"\nResult: success={result.get('success')} memory_id={result.get('memory_id')}")
    if result.get("error_message"):
        pr(f"Error: {result['error_message'][:120]}")

    print_filtered_logs(rid, "INFO+")
    print_filtered_logs(rid, "ALL")

    info_n, debug_n = count_logs(rid)
    pr(f"\n[Summary] INFO+ lines: {info_n}, DEBUG lines: {debug_n}")

    # ================================================================
    # Test 2: Lite search
    # ================================================================
    print_separator("Test 2: Lite search")

    captured_logs.clear()
    search_result = client.search(
        "What coffee does Zhang San like? What is his job?",
        user_id="test_log_user",
        agent_ids=["test_agent"],
    )
    search_rid = search_result.get("request_id", "")

    pr(f"\nSearch result: {len(search_result.get('memories', []))} memories found")

    print_filtered_logs(search_rid, "INFO+")
    print_filtered_logs(search_rid, "ALL")

    info_n, debug_n = count_logs(search_rid)
    pr(f"\n[Summary] INFO+ lines: {info_n}, DEBUG lines: {debug_n}")

    # ================================================================
    # Test 3: Full + Agent add (wait for S2 tasks to finish)
    # ================================================================
    print_separator("Test 3: Full + Agent add")

    test_content_2 = (
        "Today my girlfriend Xiaohong and I went to a newly opened Italian restaurant in Sanlitun,"
        "she is allergic to peanuts so we made a point of asking the chef."
        "After the meal we decided to take a trip to Japan next month,"
        "she wants to see the autumn leaves in Kyoto, and I want to catch a concert in Tokyo."
    )

    captured_logs.clear()
    try:
        result2 = client.add(
            test_content_2,
            user_id="test_log_user",
            agent_id="test_agent",
            session_id="test_session",
            pipeline="pro",
            enable_agent=True,
        )
        rid2 = result2.get("request_id", "")

        pr(f"\nResult: success={result2.get('success')} memory_id={result2.get('memory_id')}")
        if result2.get("error_message"):
            pr(f"Error: {result2['error_message'][:120]}")

        # wait for the S2 background tasks to finish
        pr("\nWaiting for S2 tasks to complete...")
        s2_status = wait_s2_tasks(client, rid2, timeout=120)
        pr(f"S2 status: {s2_status.get('status', 'unknown')}")
        if s2_status.get("tasks"):
            for t in s2_status["tasks"]:
                pr(f"  {t.get('task_type', '?')}: {t.get('status', '?')}")

        print_filtered_logs(rid2, "INFO+")
        print_filtered_logs(rid2, "ALL")

        info_n, debug_n = count_logs(rid2)
        pr(f"\n[Summary] INFO+ lines: {info_n}, DEBUG lines: {debug_n}")
    except Exception as e:
        pr(f"Pro pipeline not available or failed: {e}")
        import traceback
        traceback.print_exc()

    # ================================================================
    # cleanup
    # ================================================================
    print_separator("Cleanup")
    captured_logs.clear()
    client.delete_all(user_id="test_log_user")
    client.close()
    pr("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
