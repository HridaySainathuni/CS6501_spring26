"""Topic 3 Task 3: Manual tool handling - calculator with geometric functions."""
import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
try:
    from load_secrets import load_secrets
    load_secrets()
except Exception:
    pass
import json
import math
import re
import os

# Geometric and math functions exposed to the calculator
SAFE_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "ceil": math.ceil, "floor": math.floor, "fabs": math.fabs, "pi": math.pi, "e": math.e,
    "radians": math.radians, "degrees": math.degrees,
}

def calculator_impl(expr_str):
    """Evaluate a single expression. Supports numbers, + - * / ** and SAFE_FUNCS."""
    expr_str = (expr_str or "").strip()
    if not expr_str:
        return None
    # Build safe namespace
    ns = {"__builtins__": {}}
    ns.update(SAFE_FUNCS)
    ns.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
    # Replace ^ with **
    expr_str = expr_str.replace("^", "**")
    # Allow only safe chars for eval
    if not re.match(r"^[\d\s+\-*/().%\w]+$", expr_str):
        return None
    try:
        result = eval(expr_str, ns)
        return float(result) if isinstance(result, (int, float)) else result
    except Exception:
        return None

def handle_tool_call(tool_name, arguments_json_str):
    """Dispatch tool by name. Input parsed with json.loads, output with json.dumps."""
    if tool_name == "calculator":
        try:
            args = json.loads(arguments_json_str) if isinstance(arguments_json_str, str) else arguments_json_str
            expr = args.get("expression") or args.get("expr") or args.get("query")
            if expr is None and isinstance(args, dict):
                expr = list(args.values())[0] if args else None
            result = calculator_impl(str(expr))
            if result is None:
                return json.dumps({"error": "Could not evaluate expression", "input": expr})
            return json.dumps({"result": result, "expression": expr})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})
    return json.dumps({"error": f"Unknown function: {tool_name}"})

def run_agent_turn(messages, openai_client):
    """One turn: call model with tools; if tool_calls, execute and append; repeat up to 5."""
    from openai import OpenAI
    client = openai_client or OpenAI()
    tool_defs = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, exp, pi, e, etc.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}},
                "required": ["expression"],
            },
        },
    }]
    for _ in range(5):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tool_defs,
            tool_choice="auto",
            max_tokens=256,
        )
        choice = resp.choices[0]
        msg = choice.message
        if msg.content:
            return msg.content
        if not getattr(msg, "tool_calls", None):
            return None
        messages.append(msg)  # append assistant message once
        for tc in msg.tool_calls:
            name = tc.function.name
            args_str = tc.function.arguments
            result = handle_tool_call(name, args_str)
            tool_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result,
            })
    return "[Max tool turns reached]"

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run.")
        return
    from openai import OpenAI
    client = OpenAI()
    messages = [{"role": "user", "content": "What is sin(0.5) + sqrt(4)? Use the calculator."}]
    reply = run_agent_turn(messages, client)
    print("Assistant:", reply)

if __name__ == "__main__":
    main()
