"""Post-process org-exported markdown for LessWrong.

1. Export org to markdown via ox-md (emacsclient)
2. Replace local image paths with GitHub raw URLs
3. Strip Cell Timer lines
4. Fix HTML entities
5. Convert org-generated anchor links (#org...) to LessWrong-style heading slugs

Usage: python scripts/org_to_lesswrong.py [input.org] [output.md]
"""
import re
import subprocess
import sys
import os

REPO_RAW = "https://raw.githubusercontent.com/ElleNajt/Steganography_Wiretapping/main"


def slugify(heading):
    """Convert heading text to LessWrong-style anchor.

    LessWrong's titleToAnchor: keep [a-zA-Z_0-9], replace everything else
    with underscore. Case-preserving. No lowercasing.
    """
    return re.sub(r'[^a-zA-Z0-9_]', '_', heading.strip())


def fix_internal_links(md_text, org_path):
    """Replace (#org...) anchors with LessWrong-style heading slugs.

    ox-md converts [[*Heading][text]] to [text](#org<hash>), but the org IDs
    are ephemeral and not emitted as anchor targets. We parse the org source
    to build a display_text→heading mapping, then slugify the heading.
    """
    org_ids = set(re.findall(r'#(org[a-f0-9]+)', md_text))
    if not org_ids:
        return md_text

    with open(org_path) as f:
        org_text = f.read()

    # [[*Heading][display text]] → display_text maps to slugify(Heading)
    display_to_slug = {}
    for m in re.finditer(r'\[\[\*([^\]]+)\]\[([^\]]+)\]\]', org_text):
        heading = m.group(1).strip()
        display = m.group(2).strip()
        display_to_slug[display] = slugify(heading)

    # [[*Heading]] (no display text) → ox-md renders as section number
    for m in re.finditer(r'\[\[\*([^\]]+)\]\](?!\[)', org_text):
        heading = m.group(1).strip()
        display_to_slug[heading] = slugify(heading)

    def replace_link(m):
        display = m.group(1)
        # Clean HTML entities for matching
        clean = display.replace('&ldquo;', '"').replace('&rdquo;', '"')
        clean = clean.replace('&rsquo;', "'")
        for disp, slug in display_to_slug.items():
            if disp == clean or disp == display:
                return f'[{m.group(1)}](#{slug})'
        return m.group(0)  # leave unchanged if no match

    return re.sub(r'\[([^\]]+)\]\(#(org[a-f0-9]+)\)', replace_link, md_text)


def html_tables_to_markdown(text):
    """Convert HTML <table> blocks to markdown tables."""
    def convert_table(m):
        html = m.group(0)
        # Extract rows: header rows from <thead>, data rows from <tbody>
        header_rows = re.findall(r'<thead>(.*?)</thead>', html, re.DOTALL)
        body_rows = re.findall(r'<tbody>(.*?)</tbody>', html, re.DOTALL)

        def extract_rows(section):
            rows = []
            for tr in re.finditer(r'<tr>(.*?)</tr>', section, re.DOTALL):
                cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', tr.group(1), re.DOTALL)
                cells = [c.strip() for c in cells]
                rows.append(cells)
            return rows

        headers = extract_rows(header_rows[0]) if header_rows else []
        body = extract_rows(body_rows[0]) if body_rows else []

        if not headers and not body:
            return m.group(0)

        lines = []
        if headers:
            for row in headers:
                lines.append('| ' + ' | '.join(row) + ' |')
            lines.append('|' + '|'.join(['---'] * len(headers[0])) + '|')
        if body:
            if not headers:
                # Use first body row as header
                lines.append('| ' + ' | '.join(body[0]) + ' |')
                lines.append('|' + '|'.join(['---'] * len(body[0])) + '|')
                body = body[1:]
            for row in body:
                lines.append('| ' + ' | '.join(row) + ' |')

        return '\n'.join(lines)

    return re.sub(r'<table[^>]*>.*?</table>', convert_table, text, flags=re.DOTALL)


def export_org_to_md(org_path, md_path):
    """Use emacsclient to export org to markdown."""
    abs_org = os.path.abspath(org_path)
    abs_md = os.path.abspath(md_path)
    elisp = f'(with-current-buffer (find-file-noselect "{abs_org}") (org-export-to-file (quote md) "{abs_md}"))'
    subprocess.run(["emacsclient", "--eval", elisp], check=True, capture_output=True)


def postprocess(md_path, org_path):
    with open(md_path) as f:
        text = f.read()

    # Replace local image paths with GitHub raw URLs
    text = re.sub(
        r'!\[img\]\(([^)]+\.png)\)',
        lambda m: f'![img]({REPO_RAW}/{m.group(1)})',
        text
    )

    # Strip Cell Timer lines
    text = re.sub(r'^Cell Timer:.*\n', '', text, flags=re.MULTILINE)

    # Fix HTML entities from org export
    text = text.replace('&amp;', '&')
    text = text.replace('&rsquo;', "'")
    text = text.replace('&ldquo;', '"')
    text = text.replace('&rdquo;', '"')
    text = text.replace('&ndash;', '–')
    text = text.replace('&mdash;', '—')
    text = text.replace('&hellip;', '…')

    # Remove escaped underscores in link text (org export artifact)
    # e.g. Steganography\_Wiretapping -> Steganography_Wiretapping
    text = text.replace('\\_', '_')

    # Convert HTML tables to markdown tables
    text = html_tables_to_markdown(text)

    # Convert org internal links to LessWrong-style heading anchors
    text = fix_internal_links(text, org_path)

    # Clean up excessive blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    with open(md_path, 'w') as f:
        f.write(text)

if __name__ == '__main__':
    org_path = sys.argv[1] if len(sys.argv) > 1 else 'SchellingSteganography.org'
    md_path = sys.argv[2] if len(sys.argv) > 2 else 'post.md'

    print(f"Exporting {org_path} -> {md_path} via org-export...")
    export_org_to_md(org_path, md_path)

    print("Post-processing...")
    postprocess(md_path, org_path)

    # Copy to system clipboard + Emacs kill ring
    with open(md_path) as f:
        content = f.read()
    subprocess.run(["pbcopy"], input=content.encode(), check=True)
    subprocess.run(["emacsclient", "--eval",
                    '(kill-new (gui-get-selection \'CLIPBOARD))'],
                   capture_output=True)

    print(f"Done. Output: {md_path} (copied to kill ring)")
