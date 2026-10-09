"""MCP server driven through a real in-process MCP client session (protocol round trip, no subprocess)."""
import anyio
import pytest
from mcp import Client

from closepilot import mcp_server


@pytest.fixture()
def call(con, monkeypatch):
    monkeypatch.setattr(mcp_server, "_con", con)

    def run(steps):
        async def main():
            async with Client(mcp_server.mcp) as c:
                return await steps(c)
        return anyio.run(main)

    return run


def test_tools_listed_and_no_approval_tool(call):
    async def steps(c):
        return {t.name for t in (await c.list_tools()).tools}

    names = call(steps)
    assert names == {"run_close", "get_kpis", "list_queue", "explain_item", "ledger_schema", "query_ledger", "search_policy"}


def test_close_then_explain_item_with_policy_clause(call):
    async def steps(c):
        before = await c.call_tool("get_kpis")
        run = await c.call_tool("run_close", {"use_llm": False})
        queue = await c.call_tool("list_queue", {"agent": "ap", "limit": 3})
        item = queue.structured_content["result"][0]
        return before, run, queue, await c.call_tool("explain_item", {"item_id": item["item_id"]})

    before, run, queue, item = call(steps)
    assert before.is_error and "run_close" in before.content[0].text
    assert run.structured_content["kpis"]["auto_match_rate"] > 0.9
    assert len(queue.structured_content["result"]) == 3
    assert item.structured_content["policy_text"] and item.structured_content["clause"]


def test_query_ledger_is_guarded_and_policy_search_works(call):
    async def steps(c):
        return (await c.call_tool("query_ledger", {"sql": "SELECT count(*) AS n FROM vendors"}),
                await c.call_tool("query_ledger", {"sql": "DROP TABLE vendors"}),
                await c.call_tool("query_ledger", {"sql": "SELECT * FROM labels"}),
                await c.call_tool("search_policy", {"question": "segregation of duties vendor creator approves payment", "k": 1}))

    ok, drop, truth, policy = call(steps)
    assert ok.structured_content["result"] == [{"n": 40}]
    assert drop.is_error and "Only SELECT" in drop.content[0].text
    assert truth.is_error and "forbidden" in truth.content[0].text
    assert policy.structured_content["result"][0]["clause"] == "DOA-3"
