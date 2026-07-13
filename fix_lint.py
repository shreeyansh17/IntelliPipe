import re

# Fix langchain_agent.py
path1 = "src/agents/langchain_agent.py"
with open(path1, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Comment out lines 583-587ish (query_text block) and 598 (table_name)
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("query_text = (") or stripped.startswith('table_name = alert.get("table_name", "")'):
        lines[i] = line.replace(stripped, "# " + stripped, 1) if not stripped.startswith("#") else line

with open(path1, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Done - now check the file manually around query_text and table_name")
