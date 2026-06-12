# mcp_server/client_test.py
import asyncio
from fastmcp import Client


async def test_server():
    """MCPサーバーのツールをPythonから直接テストする"""

    # stdioモードでサーバーに接続
    # "mcp_server/server.py" を実行してサーバーを起動し接続する
    async with Client("mcp_server/server.py") as client:

        # ── 利用可能なツール一覧を確認 ──────────────────────────
        tools = await client.list_tools()
        print("=== 利用可能なツール ===")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:40]}...")

        # ── Resourceの確認 ───────────────────────────────────────
        resources = await client.list_resources()
        print("\n=== 利用可能なリソース ===")
        for resource in resources:
            print(f"  - {resource.uri}")

        # ── ツールの実行テスト ────────────────────────────────────
        print("\n=== list_categories のテスト ===")
        result = await client.call_tool("list_categories", {})
        print(result)

        print("\n=== search_documents のテスト ===")
        result = await client.call_tool(
            "search_documents",
            {"query": "機械学習の評価指標", "top_k": 2}
        )
        print(result)

        print("\n=== search_by_category のテスト ===")
        result = await client.call_tool(
            "search_by_category",
            {"query": "モデル評価", "category": "ML", "top_k": 2}
        )
        print(result)

        # ── Resourceの読み取り ────────────────────────────────────
        print("\n=== Resource の読み取り ===")
        content = await client.read_resource("db://categories")
        print(content)


if __name__ == "__main__":
    asyncio.run(test_server())