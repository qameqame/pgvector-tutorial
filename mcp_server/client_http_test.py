# mcp_server/client_http_test.py
import asyncio
from fastmcp import Client


async def test_http_server():
    """HTTPモードのMCPサーバーをテストする"""

    # stdioモード: async with Client("mcp_server/server.py") as client:
    # HTTPモード:  async with Client("http://localhost:8000/mcp") as client:
    # ← URLを渡すだけで自動的にHTTPトランスポートを使う
    async with Client("http://localhost:8000/mcp") as client:

        # ツール一覧の確認（stdioと全く同じAPI）
        tools = await client.list_tools()
        print("=== 利用可能なツール ===")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:40]}...")

        # ツールの実行テスト
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


if __name__ == "__main__":
    asyncio.run(test_http_server())