"""Render reports/report.md to a polished PDF with the figures embedded.
Uses a lightweight in-house Markdown->HTML pass and Chromium (Playwright) for
print-to-PDF, so no pandoc/LaTeX dependency is needed."""
import base64, os, re

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "reports", "figures")

# Map "Figure N" mentions to the figure files, inserted after their paragraph.
FIGS = {
    1: ("01_severity_hist.png", "Figure 1. Claim severity (log10 EUR): a long, heavy right tail."),
    2: ("02_frequency_lift.png", "Figure 2. Out-of-sample lift: observed vs GLM vs ML frequency by decile."),
    3: ("03_mean_excess.png", "Figure 3. Mean-excess plot: the upward slope indicates a heavy Pareto tail."),
    4: ("04_glm_vs_ml.png", "Figure 4. Frequency GLM vs ML challenger: test Poisson deviance and weighted Gini."),
    5: ("05_aggregate_loss.png", "Figure 5. Simulated annual aggregate loss with expected loss and 99.5% VaR."),
    6: ("06_capital_scenarios.png", "Figure 6. Underwriting-risk capital by tail assumption and per-claim cap (log scale)."),
}


def img_tag(fname, caption):
    with open(os.path.join(FIG, fname), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def md_inline(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w])\*(.+?)\*(?![\w])", r"<em>\1</em>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t


def _figure_numbers(text):
    """Every figure number referenced in a paragraph, handling 'Figure 3',
    'Figures 2 and 4', 'Figures 5 and 6', etc."""
    nums = set()
    for m in re.finditer(r"Figures?\s+(\d+)(?:\s+and\s+(\d+))?", text):
        for g in m.groups():
            if g:
                nums.add(int(g))
    return nums


def md_to_html(md):
    out = []
    lines = md.split("\n")
    inserted = set()
    para = []
    bullets = []

    def flush_para():
        if para:
            text = " ".join(para)
            out.append(f"<p>{md_inline(text)}</p>")
            for n in sorted(_figure_numbers(text)):
                if n in FIGS and n not in inserted:
                    fn, cap = FIGS[n]
                    out.append(img_tag(fn, cap)); inserted.add(n)
            para.clear()

    def flush_bullets():
        if bullets:
            items = "".join(f"<li>{md_inline(b)}</li>" for b in bullets)
            out.append(f"<ul>{items}</ul>")
            bullets.clear()

    def flush():
        flush_para(); flush_bullets()

    for ln in lines:
        s = ln.rstrip()
        if not s:
            flush(); continue
        if s.startswith("---"):
            flush(); out.append("<hr/>"); continue
        m = re.match(r"^(#{1,6})\s+(.*)", s)
        if m:
            flush(); lvl = len(m.group(1))
            out.append(f"<h{lvl}>{md_inline(m.group(2))}</h{lvl}>"); continue
        if s.startswith("> "):                       # blockquote (e.g. the definition)
            flush_para(); flush_bullets()
            out.append(f"<blockquote>{md_inline(s[2:])}</blockquote>"); continue
        if re.match(r"^[-*]\s+", s):                  # unordered list item
            flush_para()
            bullets.append(re.sub(r"^[-*]\s+", "", s)); continue
        if s.startswith("|"):
            flush()
            out.append("<<TABLEROW>>" + s); continue
        flush_bullets()
        para.append(s)
    flush()

    # Post-process consecutive table rows.
    html, buf = [], []
    for chunk in out:
        if chunk.startswith("<<TABLEROW>>"):
            buf.append(chunk.replace("<<TABLEROW>>", "")); continue
        if buf:
            html.append(render_table(buf)); buf = []
        html.append(chunk)
    if buf:
        html.append(render_table(buf))
    return "\n".join(html)


def render_table(rows):
    rows = [r for r in rows if set(r.strip()) != set("|-: ")]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[1:]
    h = "".join(f"<th>{md_inline(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>"
                for r in body)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 19pt; border-bottom: 2px solid #2b6cb0; padding-bottom: 6px;
     color: #1a365d; }
h2 { font-size: 13.5pt; color: #2b6cb0; margin-top: 20px; }
p { text-align: justify; }
strong { color: #1a365d; }
code { font-family: 'Courier New', monospace; font-size: 9.5pt; background:#f2f4f7;
       padding: 0 3px; border-radius: 3px; }
figure { margin: 14px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 92%; border: 1px solid #e2e8f0; border-radius: 4px; }
figcaption { font-size: 9pt; color: #555; font-style: italic; margin-top: 5px; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 12px 0;
        font-family: Helvetica, Arial, sans-serif; page-break-inside: avoid; }
th { background: #2b6cb0; color: #fff; text-align: left; padding: 6px 8px; }
td { border-bottom: 1px solid #e2e8f0; padding: 5px 8px; }
tr:nth-child(even) td { background: #f7fafc; }
hr { border: none; border-top: 1px solid #cbd5e0; margin: 18px 0; }
em { color: #444; }
blockquote { margin: 12px 0; padding: 8px 14px; background: #f7fafc;
             border-left: 4px solid #2b6cb0; font-size: 11pt; }
ul { margin: 8px 0 8px 0; padding-left: 22px; }
li { margin: 4px 0; text-align: justify; }
"""


def main():
    with open(os.path.join(ROOT, "reports", "report.md")) as f:
        md = f.read()
    body = md_to_html(md)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>" \
           f"</head><body>{body}</body></html>"
    html_path = os.path.join(ROOT, "reports", "report.html")
    with open(html_path, "w") as f:
        f.write(html)

    from playwright.sync_api import sync_playwright
    pdf_path = os.path.join(ROOT, "reports", "report.pdf")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + html_path)
        pg.pdf(path=pdf_path, format="A4", print_background=True)
        b.close()
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
