#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Markdown to HTML with nice styling.
"""

import markdown2
from pathlib import Path


def convert_md_to_html():
    md_path = Path(__file__).parent / "global_partition_paper.md"
    html_path = Path(__file__).parent / "global_partition_paper.html"

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown2.markdown(md_content, extras=[
        'fenced-code-blocks',
        'tables',
        'code-friendly',
        'header-ids'
    ])

    # Add CSS styling
    full_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Global Layer Partition Optimization for DNN Dataflow</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            text-align: center;
        }}
        h2 {{
            color: #2980b9;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        h3 {{
            color: #27ae60;
            margin-top: 25px;
        }}
        pre {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
        }}
        code {{
            background: #f1f2f3;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding: 10px 20px;
            background: #f8f9fa;
            font-style: italic;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f2f2f2;
        }}
        hr {{
            border: none;
            border-top: 2px solid #eee;
            margin: 30px 0;
        }}
        .equation {{
            text-align: center;
            margin: 20px 0;
            font-style: italic;
        }}
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
{html_content}
</body>
</html>'''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"HTML file generated: {html_path}")
    return html_path


if __name__ == "__main__":
    convert_md_to_html()
