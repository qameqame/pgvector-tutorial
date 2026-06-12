# 12_mcp_agent.py
import asyncio
from google import genai
from google.genai import types
from fastmcp import Client
from dotenv import load_dotenv
import os
import time

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def run_agent(task: str):
    """MCPサーバーのツールを使うエージェント"""

    print(f"\nタスク: {task}")
    print("=" * 60)

    # MCPサーバーに接続してツール定義を取得
    async with Client("mcp_server/server.py") as mcp_client:

        # ── MCPからツール定義を自動取得 ──────────────────────────
        # Tool Useでは types.FunctionDeclaration を手書きしていたが
        # MCPではサーバーから自動でスキーマを取得できる
        mcp_tools = await mcp_client.list_tools()

        # MCPのツール定義をGemini用に変換する
        gemini_tools = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            param_name: types.Schema(
                                type=types.Type.STRING
                                if param_schema.get("type") == "string"
                                else types.Type.INTEGER
                                if param_schema.get("type") == "integer"
                                else types.Type.STRING,
                                description=param_schema.get("description", ""),
                            )
                            for param_name, param_schema in
                            (tool.inputSchema.get("properties") or {}).items()
                        },
                        required=tool.inputSchema.get("required", []),
                    ),
                )
                for tool in mcp_tools
            ]
        )

        print(f"MCPサーバーから{len(mcp_tools)}個のツールを取得しました")

        # ── Agenticループ ─────────────────────────────────────────
        contents = [types.Content(role="user", parts=[types.Part(text=task)])]

        for step in range(8):
            print(f"\n[Step {step + 1}]")

            # リトライ処理
            response = None
            for attempt in range(5):
                try:
                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(tools=[gemini_tools]),
                    )
                    break
                except Exception as e:
                    if ("503" in str(e) or "429" in str(e)) and attempt < 4:
                        wait = (attempt + 1) * 10
                        print(f"  サーバー混雑、{wait}秒待ってリトライ...")
                        time.sleep(wait)
                    else:
                        raise

            candidates = response.candidates
            if not candidates or not candidates[0].content or not candidates[0].content.parts:
                return "（回答を取得できませんでした）"

            part = candidates[0].content.parts[0]

            if part.function_call:
                func_name = part.function_call.name
                func_args = dict(part.function_call.args)
                print(f"  → {func_name}({func_args})")

                # Tool Useでは dispatch() で直接関数を呼んでいたが
                # MCPではサーバー経由でツールを実行する
                result = await mcp_client.call_tool(func_name, func_args)
                print(f"  → {len(result) if isinstance(result, list) else result}")

                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(function_call=part.function_call)]
                    )
                )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(
                            function_response=types.FunctionResponse(
                                name=func_name,
                                response={"result": result},
                            )
                        )]
                    )
                )

            else:
                text_parts = [
                    p.text for p in candidates[0].content.parts
                    if hasattr(p, 'text') and p.text
                ]
                print(f"\n[完了] {step + 1}ステップで達成")
                return "\n".join(text_parts)

    return "最大ステップ数に達しました。"


# ── 実行 ────────────────────────────────────────────────────────
async def main():
    result = await run_agent(
        "まずカテゴリを確認して、MLカテゴリの評価指標について詳しく教えてください。"
    )
    print(f"\n最終回答:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())