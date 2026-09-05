"""Function-level CC and CRAP reports (statement coverage, not branch coverage)."""
import argparse
import json
from pathlib import Path

from radon.complexity import cc_visit


def crap_score(complexity, coverage):
    """CRAP = CC² × (1 - covered fraction)³ + CC."""
    if not 0 <= coverage <= 1:
        raise ValueError("coverage must be a fraction between zero and one")
    return complexity ** 2 * (1 - coverage) ** 3 + complexity


def functions(blocks):
    for block in blocks:
        # cc_visit already includes methods alongside class aggregates.
        if not hasattr(block, "methods"):
            yield block
            yield from functions(block.closures)


def report(sources, coverage_files=None):
    rows = []
    for source in sources:
        data = (coverage_files or {}).get(source, {})
        executed = set(data.get("executed_lines", []))
        statements = executed | set(data.get("missing_lines", []))
        for block in functions(cc_visit(Path(source).read_text())):
            lines = set(range(block.lineno, block.endline + 1))
            # Nested functions have their own entries; don't double-count their bodies.
            for child in block.closures:
                lines.difference_update(range(child.lineno + 1, child.endline + 1))
            relevant = statements & lines
            covered = len(executed & relevant) / len(relevant) if relevant else 0.0
            row = dict(file=source, name=block.fullname, line=block.lineno,
                       cc=block.complexity)
            if coverage_files is not None:
                if source not in coverage_files:
                    raise ValueError(f"Missing coverage for {source}")
                row.update(coverage=round(100 * covered, 2),
                           crap=round(crap_score(block.complexity, covered), 2))
            rows.append(row)
    return sorted(rows, key=lambda row: row.get("crap", row["cc"]), reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--coverage")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    coverage_files = None
    if args.coverage:
        coverage_files = json.loads(Path(args.coverage).read_text())["files"]
    rows = report(args.sources, coverage_files)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    text = "CC   Coverage%  CRAP       Function\n" + "\n".join(
        f"{r['cc']:4} {str(r.get('coverage', '-')):>9}  {str(r.get('crap', '-')):>9}  "
        f"{r['file']}:{r['line']} {r['name']}" for r in rows
    ) + "\n"
    output.with_suffix(".txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
