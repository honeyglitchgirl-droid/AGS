#!/usr/bin/env python3
"""Build the inert v3 research equation tier from pinned academic snapshots.

This script extracts display relations only.  It deliberately rejects prose,
proofs, exercises, figures, unresolved source-defined commands, short/trivial
expressions, and normalized/reversed duplicates.  Formula text is never
executed.  External source trees are inputs and are not vendored into AGS.
"""
from __future__ import annotations

import hashlib
import json
import re
import runpy
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "src/ags_sci/knowledge/equations/reference"
_SCHEMA = runpy.run_path(str(ROOT / "src/ags_sci/knowledge/equations/schema.py"))
LEVEL_TOPICS = _SCHEMA["LEVEL_TOPICS"]
TOPIC_CODES = _SCHEMA["TOPIC_CODES"]

DLMF_ROOT = Path("/tmp/dlmf-json/dlmf-chapters-OneJsonObjectPerMathExpr/oneJsonFilePerChapter")
SOURCE_ROOTS = {
    "SCHULLER-GEOMETRIC": Path("/tmp/src-geometric"),
    "SCHULLER-QUANTUM": Path("/tmp/src-qm"),
    "WILTSHIRE-GR": Path("/tmp/src-gr"),
    "UOFT-QFT-JOOT": Path("/tmp/src-qft"),
    "RESTREPO-QFT-SM": Path("/tmp/src-sm"),
    "FITZPATRICK-PLASMA": Path("/tmp/src-plasma"),
    "MSU-STELLAR": Path("/tmp/src-stars"),
    "KEIO-QUANTUM-COMMS": Path("/tmp/src-qinfo"),
    "MIT-NUCLEAR-NOTES": Path("/tmp/src-nuclear"),
    "NUCLEAR-TALENT-MANYBODY": Path("/tmp/src-manybody"),
    "YACHAY-NONLINEAR": Path("/tmp/src-nonlinear"),
    "STACKS-PROJECT": Path("/tmp/src-stacks"),
}

SOURCE_SNAPSHOTS = {
    "SCHULLER-GEOMETRIC": ("sreahw/schuller-geometric", "d324cd4b94d424a2734aea4b13bc3c9139cb8f90"),
    "SCHULLER-QUANTUM": ("sreahw/schuller-quantum", "aabdd400ec8a90d1071c04da388f1e68e5b7053e"),
    "WILTSHIRE-GR": ("Jollywatt/General-Relativity-lectures", "d9451a0285c7251e002fc13757bbd802549c4d17"),
    "UOFT-QFT-JOOT": ("peeterjoot/phy2403-quantum-field-theory", "c6fd595c4e63419c47a7f1c334f3042f5289c2fc"),
    "RESTREPO-QFT-SM": ("restrepo/standard-model-and-beyond", "0fc70d51e23253657ae134eaba5f301ef48c171d"),
    "FITZPATRICK-PLASMA": ("noblepa/380", "f062a7b117ffae2367f715b8f06289c1eb647872"),
    "MSU-STELLAR": ("Open-Astrophysics-Bookshelf/stellar-physics-notes", "afecac238cf47482e8d566bd27b3f634fca4d5cb"),
    "KEIO-QUANTUM-COMMS": ("sfc-aqua/Overview-of-Quantum-Communications-E", "163bdbac1db1e5734db0359c2e936d1bb51b1072"),
    "MIT-NUCLEAR-NOTES": ("lilulu/nuclear-physics-notes", "ca7fff104676f78746e32406041d46fe19640472"),
    "NUCLEAR-TALENT-MANYBODY": ("ManyBodyPhysics/Course2ManyBodyMethods", "b49f8c4afed9145d643c93ef63a6e79fedc33ffe"),
    "YACHAY-NONLINEAR": ("wbandabarragan/nonlinear-dynamics-chaos", "e253ab48c20149af5d2ae67e8257df045c8f441a"),
    "STACKS-PROJECT": ("stacks/stacks-project", "a04446e57ec1fbc252a871afcec7752fb2807b14"),
}

NEW_SOURCES = [
    {
        "id": "SCHULLER-GEOMETRIC", "authority": "Frederic P. Schuller; transcription by Simon Rea",
        "title": "Geometric Anatomy of Theoretical Physics lecture notes",
        "url": "https://github.com/sreahw/schuller-geometric",
        "scope": ["research", "differential_geometry", "topology", "lie_theory", "representation_theory"],
        "snapshot_commit": SOURCE_SNAPSHOTS["SCHULLER-GEOMETRIC"][1], "language": "en",
    },
    {
        "id": "SCHULLER-QUANTUM", "authority": "Frederic P. Schuller; transcription by Simon Rea",
        "title": "Quantum Theory lecture notes",
        "url": "https://github.com/sreahw/schuller-quantum",
        "scope": ["research", "functional_analysis", "operator_theory", "quantum_theory"],
        "snapshot_commit": SOURCE_SNAPSHOTS["SCHULLER-QUANTUM"][1], "language": "en",
    },
    {
        "id": "WILTSHIRE-GR", "authority": "David L. Wiltshire; typesetting repository by Josh Watt",
        "title": "General Relativity lecture notes",
        "url": "https://github.com/Jollywatt/General-Relativity-lectures",
        "scope": ["research", "general_relativity"],
        "snapshot_commit": SOURCE_SNAPSHOTS["WILTSHIRE-GR"][1], "language": "en",
    },
    {
        "id": "UOFT-QFT-JOOT", "authority": "Peeter Joot; University of Toronto PHY2403 course context",
        "title": "Quantum Field Theory lecture notes and worked formalism",
        "url": "https://github.com/peeterjoot/phy2403-quantum-field-theory",
        "scope": ["research", "quantum_field_theory", "gauge_theory"],
        "snapshot_commit": SOURCE_SNAPSHOTS["UOFT-QFT-JOOT"][1], "language": "en",
    },
    {
        "id": "RESTREPO-QFT-SM", "authority": "Diego Restrepo, Universidad de Antioquia",
        "title": "Detailed Lecture Notes in Quantum Field Theory",
        "url": "https://doi.org/10.5281/zenodo.3470180",
        "repository": "https://github.com/restrepo/standard-model-and-beyond",
        "scope": ["research", "quantum_field_theory", "standard_model", "neutrino_physics"],
        "snapshot_commit": SOURCE_SNAPSHOTS["RESTREPO-QFT-SM"][1], "language": "en",
    },
    {
        "id": "FITZPATRICK-PLASMA", "authority": "Richard Fitzpatrick, University of Texas at Austin",
        "title": "Graduate Plasma Physics lecture notes",
        "url": "https://github.com/noblepa/380",
        "scope": ["research", "plasma_physics", "fluid_physics", "magnetohydrodynamics"],
        "snapshot_commit": SOURCE_SNAPSHOTS["FITZPATRICK-PLASMA"][1], "language": "en",
    },
    {
        "id": "MSU-STELLAR", "authority": "Edward F. Brown, Michigan State University",
        "title": "Graduate Stellar Astrophysics notes",
        "url": "https://github.com/Open-Astrophysics-Bookshelf/stellar-physics-notes",
        "scope": ["research", "astrophysics", "stellar_physics"],
        "snapshot_commit": SOURCE_SNAPSHOTS["MSU-STELLAR"][1], "language": "en",
    },
    {
        "id": "KEIO-QUANTUM-COMMS", "authority": "Keio University AQUA group and Japan Q-LEAP Quantum Academy",
        "title": "Overview of Quantum Communications",
        "url": "https://github.com/sfc-aqua/Overview-of-Quantum-Communications-E",
        "scope": ["research", "quantum_information", "quantum_optics"],
        "snapshot_commit": SOURCE_SNAPSHOTS["KEIO-QUANTUM-COMMS"][1], "language": "en",
    },
    {
        "id": "MIT-NUCLEAR-NOTES", "authority": "MIT Department of Nuclear Science and Engineering course-note contributors",
        "title": "MIT Nuclear Physics course materials",
        "url": "https://github.com/lilulu/nuclear-physics-notes",
        "scope": ["research", "nuclear_physics"],
        "snapshot_commit": SOURCE_SNAPSHOTS["MIT-NUCLEAR-NOTES"][1], "language": "en",
    },
    {
        "id": "NUCLEAR-TALENT-MANYBODY", "authority": "Morten Hjorth-Jensen and Nuclear TALENT instructors",
        "title": "Nuclear TALENT Course 2: Many-Body Methods for Nuclear Physics",
        "url": "https://github.com/ManyBodyPhysics/Course2ManyBodyMethods",
        "scope": ["research", "many_body_physics", "nuclear_physics"],
        "snapshot_commit": SOURCE_SNAPSHOTS["NUCLEAR-TALENT-MANYBODY"][1], "language": "en",
    },
    {
        "id": "YACHAY-NONLINEAR", "authority": "Wladimir E. Banda-Barragán, Universidad Yachay Tech",
        "title": "Nonlinear Dynamics and Chaos MSc course materials",
        "url": "https://github.com/wbandabarragan/nonlinear-dynamics-chaos",
        "scope": ["research", "nonlinear_physics", "dynamical_systems"],
        "snapshot_commit": SOURCE_SNAPSHOTS["YACHAY-NONLINEAR"][1], "language": "en",
    },
    {
        "id": "TONG-STRING", "authority": "David Tong, University of Cambridge",
        "title": "Lectures on String Theory",
        "url": "https://davidtong.org/teaching/string-theory/",
        "scope": ["research", "string_theory", "quantum_gravity"], "language": "en",
    },
    {
        "id": "MIT-18966-SYMPLECTIC", "authority": "Denis Auroux and Kartik Venkatram, MIT OpenCourseWare",
        "title": "18.966 Geometry of Manifolds: symplectic geometry lecture notes",
        "url": "https://ocw.mit.edu/courses/18-966-geometry-of-manifolds-spring-2007/pages/lecture-notes/",
        "scope": ["research", "symplectic_geometry", "complex_geometry"], "language": "en",
    },
]

ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|eqnarray\*?|displaymath)\}(.*?)\\end\{\1\}",
    re.S,
)
BRACKET_RE = re.compile(r"\\\[(.*?)\\\]", re.S)
DOLLAR_RE = re.compile(r"(?<!\\)\$\$(.*?)(?<!\\)\$\$", re.S)
CUSTOM_DISPLAY_RE = re.compile(r"\\bse\b(.*?)\\ese\b", re.S)
REL_RE = re.compile(
    r"(?:=|<|>|\\(?:leq?|geq?|sim|simeq|approx|equiv|propto|to|mapsto|in|notin|subset|supset|cong|neq|perp|parallel|rightarrow|leftarrow|Rightarrow|Longrightarrow|Longleftrightarrow|coloneqq|doteq)\b)"
)
SECTION_RE = re.compile(r"\\(?:chapter|section|subsection|subsubsection)\*?(?:\[[^]]*\])?\{([^{}]{1,240})\}")
COMMAND_RE = re.compile(r"\\([A-Za-z@]+)")
COMMAND_DEF_RE = re.compile(r"\\(?:newcommand|renewcommand|providecommand|def)\*?\s*\{?\\([A-Za-z@]+)")
REJECT_SNIPPETS = (
    "includegraphics", "begin{figure", "begin{tikz", "xymatrix", "Qcircuit",
    "begin{tabular", "bibliography", "\\input{", "\\include{", "TODO",
    "todo", "Exercise", "exercise", "solution to", "\\url{", "\\href{",
)


@dataclass(frozen=True)
class Candidate:
    formula: str
    source_id: str
    locator: str
    subtopic: str
    assumptions: tuple[str, ...] = ()


def strip_comments_preserve_lines(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", lambda m: " " * len(m.group(0)), text)


def clean_formula(raw: str, *, wrap_alignment: bool = False) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = re.sub(r"\\(?:label|tag|tag\*)\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:nonumber|notag)\b", "", text)
    text = text.replace("\\ifrac", "\\frac")
    text = text.replace("\\operatorname*{d e t}", "\\det")
    text = text.strip()
    text = re.sub(r"^\$+", "", text)
    text = re.sub(r"\$+[.,;]?$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    # Conservative expansion of widespread, one-argument notation shortcuts.
    text = re.sub(r"\\Norm\{([^{}]+)\}", r"\\lVert \1 \\rVert", text)
    text = re.sub(r"\\Abs\{([^{}]+)\}", r"\\lvert \1 \\rvert", text)
    text = re.sub(r"\\ket\{([^{}]+)\}", r"|\1\\rangle", text)
    text = re.sub(r"\\bra\{([^{}]+)\}", r"\\langle \1|", text)
    text = re.sub(r"\\braket\{([^{}]+)\}\{([^{}]+)\}", r"\\langle \1|\2\\rangle", text)
    text = re.sub(r"\\B([A-Za-z])\b", r"\\boldsymbol{\1}", text)
    if wrap_alignment and ("&" in text or "\\\\" in text) and not text.startswith("\\begin{"):
        text = "\\begin{aligned} " + text + " \\end{aligned}"
    return text


def latex_delimiter_problem(formula: str) -> str | None:
    environments: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([A-Za-z*]+)\}", formula):
        action, name = match.groups()
        if action == "begin":
            environments.append(name)
        elif not environments or environments.pop() != name:
            return "unmatched environment"
    if environments:
        return "unclosed environment"
    stack: list[str] = []
    index = 0
    while index < len(formula):
        if formula[index:index + 2] in {r"\{", r"\}", r"\$", r"\\"}:
            index += 2
            continue
        char = formula[index]
        if char == "{":
            stack.append(char)
        elif char == "}":
            if not stack:
                return "unmatched brace"
            stack.pop()
        index += 1
    return "unclosed brace" if stack else None


def meaningful(formula: str) -> bool:
    if not (30 <= len(formula) <= 3000) or not REL_RE.search(formula):
        return False
    if any(snippet in formula for snippet in REJECT_SNIPPETS):
        return False
    if formula.count("=") > 18 or formula.count("\\\\") > 18:
        return False
    if len(re.findall(r"[A-Za-z\\]", formula)) < 6:
        return False
    return latex_delimiter_problem(formula) is None


def normalized_key(formula: str) -> str:
    value = re.sub(r"\\(?:left|right)", "", formula)
    value = re.sub(r"\s+", "", value).strip(".,;")
    # Reject merely reversed single equalities as count inflation.
    if value.count("=") == 1 and not any(x in value for x in ("<=", ">=", "\\begin")):
        left, right = value.split("=", 1)
        value = "=".join(sorted((left, right)))
    return value


def relation_type(formula: str) -> str:
    if re.search(r"[<>]|\\(?:leq?|geq?)\b", formula):
        return "inequality"
    if re.search(r"\\(?:sim|approx|simeq)\b", formula):
        return "asymptotic_relation"
    if any(token in formula for token in ("\\coloneqq", ":=", "\\equiv")):
        return "definition"
    if "\\begin{cases}" in formula:
        return "differential_system"
    if any(token in formula for token in ("\\partial", "\\nabla", "\\frac{d", "\\dot{")):
        return "evolution_equation"
    if "\\int" in formula:
        return "integral_relation"
    if any(token in formula for token in ("[", "\\commutator")) and "=" in formula:
        return "commutation_relation"
    if any(token in formula for token in ("\\to", "\\mapsto", "\\rightarrow")):
        return "transformation"
    if any(token in formula for token in ("\\sum", "_{n+1}", "_{k+1}")):
        return "recurrence"
    return "equation"


def line_locator(source_id: str, root: Path, path: Path, start: int, end: int, text: str) -> str:
    repo, commit = SOURCE_SNAPSHOTS[source_id]
    relative = path.relative_to(root).as_posix()
    first = text.count("\n", 0, start) + 1
    last = text.count("\n", 0, end) + 1
    return f"https://github.com/{repo}/blob/{commit}/{relative}#L{first}-L{last}"


def defined_commands(root: Path) -> set[str]:
    commands: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".tex", ".sty", ".cls"}:
            commands.update(COMMAND_DEF_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))
    # These wrappers are removed by extraction and are not formula dependencies.
    commands.difference_update({"bse", "ese"})
    return commands


def display_candidates(source_id: str, paths: Iterable[Path], *, reject_custom: bool = True) -> list[Candidate]:
    root = SOURCE_ROOTS[source_id]
    custom = defined_commands(root) if reject_custom else set()
    result: list[Candidate] = []
    for path in sorted(paths):
        original = path.read_text(encoding="utf-8", errors="ignore")
        text = strip_comments_preserve_lines(original)
        sections = list(SECTION_RE.finditer(text))
        matches: list[tuple[int, int, str, bool]] = []
        for match in ENV_RE.finditer(text):
            env, body = match.group(1), match.group(2)
            matches.append((match.start(), match.end(), body, env.startswith(("align", "gather", "eqnarray"))))
        for regex in (BRACKET_RE, DOLLAR_RE, CUSTOM_DISPLAY_RE):
            for match in regex.finditer(text):
                matches.append((match.start(), match.end(), match.group(1), regex is CUSTOM_DISPLAY_RE))
        matches.sort()
        accepted_spans: list[tuple[int, int]] = []
        for start, end, body, wrap in matches:
            if any(a <= start and end <= b for a, b in accepted_spans):
                continue
            formula = clean_formula(body, wrap_alignment=wrap)
            if not meaningful(formula):
                continue
            if reject_custom and (set(COMMAND_RE.findall(formula)) & custom):
                continue
            section = "display_relation"
            for section_match in sections:
                if section_match.start() > start:
                    break
                section = clean_formula(section_match.group(1))[:96]
            result.append(Candidate(
                formula=formula,
                source_id=source_id,
                locator=line_locator(source_id, root, path, start, end, text),
                subtopic=section or "display_relation",
            ))
            accepted_spans.append((start, end))
    return result


def notebook_candidates(source_id: str, paths: Iterable[Path]) -> list[Candidate]:
    root = SOURCE_ROOTS[source_id]
    result: list[Candidate] = []
    repo, commit = SOURCE_SNAPSHOTS[source_id]
    for path in sorted(paths):
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        heading = "notebook_display_relation"
        for cell_number, cell in enumerate(notebook.get("cells", []), 1):
            if cell.get("cell_type") != "markdown":
                continue
            text = "".join(cell.get("source", []))
            headings = re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M)
            if headings:
                heading = clean_formula(headings[-1])[:96]
            for regex in (BRACKET_RE, DOLLAR_RE):
                for match in regex.finditer(text):
                    formula = clean_formula(match.group(1))
                    if not meaningful(formula):
                        continue
                    rel = path.relative_to(root).as_posix()
                    locator = f"https://github.com/{repo}/blob/{commit}/{rel}#cell-{cell_number}"
                    result.append(Candidate(formula, source_id, locator, heading))
    return result


def dlmf_candidates() -> list[tuple[int, Candidate]]:
    result: list[tuple[int, Candidate]] = []
    decoder = json.JSONDecoder()
    for path in sorted(DLMF_ROOT.glob("*.json"), key=lambda item: int(item.stem)):
        chapter = int(path.stem)
        text = path.read_text(encoding="utf-8")
        index = 0
        while True:
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break
            value, index = decoder.raw_decode(text, index)
            equation = value.get("equation")
            if not isinstance(equation, dict):
                continue
            formula = clean_formula(equation.get("tex", ""))
            if not meaningful(formula):
                continue
            permalink = equation.get("permalink", "").replace("http://", "https://")
            if not permalink.startswith("https://dlmf.nist.gov/"):
                continue
            context = equation.get("context-references", {})
            subtopic = (
                context.get("paragraph-title")
                or context.get("subsection-title")
                or context.get("section-title")
                or "numbered_relation"
            )
            assumptions: tuple[str, ...] = ()
            constraint = equation.get("constraints")
            if isinstance(constraint, dict):
                parsed = re.sub(
                    r"\s+", " ", clean_formula(constraint.get("tex", "")).replace("$", "")
                ).strip()
                if parsed and len(parsed) <= 512:
                    assumptions = (parsed,)
            result.append((chapter, Candidate(
                formula=formula, source_id="NIST-DLMF", locator=permalink,
                subtopic=clean_formula(str(subtopic))[:96], assumptions=assumptions,
            )))
    return result


def math_topic_for_dlmf(chapter: int) -> str:
    if chapter == 3:
        return "numerical_mathematics"
    if chapter in {25, 27}:
        return "number_theory"
    if chapter == 32:
        return "dynamical_systems"
    if chapter == 34:
        return "representation_theory"
    if chapter == 35:
        return "operator_theory"
    if chapter in {28, 29, 30, 31, 33, 36}:
        return "mathematical_physics"
    return "special_functions"


def geometric_topic(path_text: str) -> str:
    name = Path(path_text).name
    prefix = int(name[:2]) if name[:2].isdigit() else 0
    if prefix in {4, 5}:
        return "differential_topology" if prefix == 5 else "algebraic_topology"
    if 6 <= prefix <= 12 or 19 <= prefix <= 25:
        return "differential_riemannian_geometry"
    if 13 <= prefix <= 16 or prefix == 18:
        return "lie_theory"
    if prefix == 17:
        return "representation_theory"
    return "differential_riemannian_geometry"


def quantum_math_topic(path_text: str) -> str:
    name = Path(path_text).name
    prefix = int(name[:2]) if name[:2].isdigit() else 0
    if prefix in {4, 7, 8, 10, 11}:
        return "operator_theory"
    if prefix == 5:
        return "probability_stochastic_analysis"
    return "functional_analysis"


MATH_SEEDS: dict[str, list[tuple[str, str, str, str]]] = {
    "partial_differential_equations": [
        (r"-\nabla\!\cdot\!(A(x)\nabla u)+c(x)u=f\quad\text{in }\Omega", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/pdes/ch4.pdf#page=1", "elliptic_operator"),
        (r"\int_\Omega A\nabla u\cdot\nabla v+cuv\,dx=\int_\Omega fv\,dx\quad\forall v\in H_0^1(\Omega)", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/pdes/ch4.pdf#page=4", "weak_elliptic_form"),
        (r"u_t+\nabla\cdot f(u)=0\quad\text{in }\mathbb{R}^n\times(0,\infty)", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/pdes/pde_notes.pdf#page=75", "conservation_law"),
        (r"u_t-\nabla\cdot(a(x,t)\nabla u)=f", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/pdes/pde_notes.pdf#page=156", "parabolic_equation"),
        (r"E(t)=\frac12\int_\Omega\left(|u_t|^2+c^2|\nabla u|^2\right)dx=E(0)", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/pdes/pde_notes.pdf#page=116", "wave_energy"),
        (r"\Delta u=0\ \Longrightarrow\ u(x)=\frac{1}{|\partial B_r|}\int_{\partial B_r(x)}u\,dS", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/pdes/pde_notes.pdf#page=24", "mean_value_property"),
    ],
    "calculus_of_variations": [
        (r"\mathcal{F}[u]=\int_a^b F(x,u,u')\,dx", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/m280_09/ch3.pdf#page=7", "variational_functional"),
        (r"\frac{d}{dx}F_{u'}(x,u,u')-F_u(x,u,u')=0", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/m280_09/ch3.pdf#page=8", "euler_lagrange"),
        (r"F-u'F_{u'}=\mathrm{constant}\quad(F_x=0)", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/m280_09/ch3.pdf#page=10", "beltrami_identity"),
        (r"\nabla\cdot\left(\frac{\nabla u}{\sqrt{1+|\nabla u|^2}}\right)=0", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/m280_09/ch3.pdf#page=28", "minimal_surface_equation"),
        (r"\delta\!\int_{t_0}^{t_1}L(q,\dot q,t)\,dt=0", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/m280_09/ch3.pdf#page=3", "stationary_action"),
        (r"\frac{\partial L}{\partial q^i}-\frac{d}{dt}\frac{\partial L}{\partial\dot q^i}=0", "UCDAVIS-PDE-HUNTER", "https://www.math.ucdavis.edu/~hunter/m280_09/ch3.pdf#page=5", "euler_lagrange_system"),
    ],
    "probability_stochastic_analysis": [
        (r"dX_t=b(X_t,t)\,dt+\sigma(X_t,t)\,dW_t", "MIT-15070-STOCHASTIC", "https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/pages/lecture-notes/", "ito_diffusion"),
        (r"df(t,X_t)=\left(f_t+b f_x+\frac12\sigma^2f_{xx}\right)dt+\sigma f_x\,dW_t", "MIT-15070-STOCHASTIC", "https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/pages/lecture-notes/", "ito_formula"),
        (r"\mathbb{E}\left[\left(\int_0^T H_t\,dW_t\right)^2\right]=\mathbb{E}\int_0^T H_t^2\,dt", "MIT-15070-STOCHASTIC", "https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/pages/lecture-notes/", "ito_isometry"),
        (r"[W]_t=\lim_{|\Pi|\to0}\sum_k(W_{t_{k+1}}-W_{t_k})^2=t", "MIT-15070-STOCHASTIC", "https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/pages/lecture-notes/", "quadratic_variation"),
        (r"\partial_t p=-\partial_x(bp)+\frac12\partial_x^2(\sigma^2p)", "MIT-15070-STOCHASTIC", "https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/pages/lecture-notes/", "fokker_planck"),
        (r"(\mathcal{L}f)(x)=b(x)f'(x)+\frac12\sigma^2(x)f''(x)", "MIT-15070-STOCHASTIC", "https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/pages/lecture-notes/", "diffusion_generator"),
    ],
    "symplectic_complex_geometry": [
        (r"\omega=\sum_{i=1}^n dq^i\wedge dp_i,\qquad d\omega=0", "MIT-18966-SYMPLECTIC", "https://ocw.mit.edu/courses/18-966-geometry-of-manifolds-spring-2007/pages/lecture-notes/", "canonical_symplectic_form"),
        (r"\iota_{X_H}\omega=dH", "MIT-18966-SYMPLECTIC", "https://ocw.mit.edu/courses/18-966-geometry-of-manifolds-spring-2007/pages/lecture-notes/", "hamiltonian_vector_field"),
        (r"\{f,g\}=\omega(X_f,X_g)=X_g(f)", "MIT-18966-SYMPLECTIC", "https://ocw.mit.edu/courses/18-966-geometry-of-manifolds-spring-2007/pages/lecture-notes/", "poisson_bracket"),
        (r"\mathcal{L}_{X_H}\omega=d(\iota_{X_H}\omega)+\iota_{X_H}d\omega=0", "MIT-18966-SYMPLECTIC", "https://ocw.mit.edu/courses/18-966-geometry-of-manifolds-spring-2007/pages/lecture-notes/", "symplectic_flow"),
        (r"\omega(Ju,Jv)=\omega(u,v),\qquad g(u,v)=\omega(u,Jv)", "MIT-18966-SYMPLECTIC", "https://ocw.mit.edu/courses/18-966-geometry-of-manifolds-spring-2007/pages/lecture-notes/", "compatible_triple"),
        (r"\omega^n\neq0\quad\Longleftrightarrow\quad\omega\text{ is nondegenerate}", "MIT-18966-SYMPLECTIC", "https://ocw.mit.edu/courses/18-966-geometry-of-manifolds-spring-2007/pages/lecture-notes/", "nondegeneracy"),
    ],
    "optimization_control": [
        (r"L(x,\lambda,\nu)=f_0(x)+\sum_{i=1}^m\lambda_i f_i(x)+\nu^T(Ax-b)", "STANFORD-CONVEX", "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=244", "lagrangian"),
        (r"g(\lambda,\nu)=\inf_x L(x,\lambda,\nu)", "STANFORD-CONVEX", "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=250", "dual_function"),
        (r"\nabla f_0(x^*)+\sum_i\lambda_i^*\nabla f_i(x^*)+A^T\nu^*=0", "STANFORD-CONVEX", "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=263", "kkt_stationarity"),
        (r"\lambda_i^* f_i(x^*)=0,\qquad \lambda_i^*\ge0", "STANFORD-CONVEX", "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=263", "complementary_slackness"),
        (r"-\dot P=A^TP+PA-PBR^{-1}B^TP+Q", "STANFORD-CONVEX", "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", "riccati_equation"),
        (r"-\partial_t V(x,t)=\inf_u\{\ell(x,u,t)+\nabla V\cdot f(x,u,t)\}", "STANFORD-CONVEX", "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", "hamilton_jacobi_bellman"),
        (r"\dot x^*=\partial_p H(x^*,p^*,u^*),\qquad\dot p^*=-\partial_x H(x^*,p^*,u^*)", "STANFORD-CONVEX", "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", "pontryagin_system"),
        (r"H(x^*,p^*,u^*)=\min_{u\in U}H(x^*,p^*,u)", "STANFORD-CONVEX", "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf", "minimum_principle"),
    ],
}

PHYSICS_SEEDS: dict[str, list[tuple[str, str, str, str, str]]] = {
    "condensed_matter": [
        (r"\psi_{n\mathbf{k}}(\mathbf r)=e^{i\mathbf{k}\cdot\mathbf r}u_{n\mathbf{k}}(\mathbf r)", "TONG-SOLIDSTATE", "https://www.damtp.cam.ac.uk/user/tong/solidstate.html", "bloch_theorem", "established"),
        (r"E(k)=\varepsilon_0-2t\cos(ka)", "TONG-SOLIDSTATE", "https://www.damtp.cam.ac.uk/user/tong/solidstate.html", "tight_binding_band", "model_dependent"),
        (r"\hbar\dot{\mathbf k}=-e(\mathbf E+\dot{\mathbf r}\times\mathbf B)", "TONG-SOLIDSTATE", "https://www.damtp.cam.ac.uk/user/tong/solidstate.html", "semiclassical_crystal_momentum", "model_dependent"),
        (r"\dot{\mathbf r}=\frac{1}{\hbar}\nabla_{\mathbf k}\varepsilon_n(\mathbf k)", "TONG-SOLIDSTATE", "https://www.damtp.cam.ac.uk/user/tong/solidstate.html", "band_velocity", "model_dependent"),
        (r"\Omega_n(\mathbf k)=\nabla_{\mathbf k}\times i\langle u_{n\mathbf k}|\nabla_{\mathbf k}u_{n\mathbf k}\rangle", "TONG-SOLIDSTATE", "https://www.damtp.cam.ac.uk/user/tong/solidstate.html", "berry_curvature", "established"),
        (r"C_n=\frac{1}{2\pi}\int_{\mathrm{BZ}}\Omega_n(\mathbf k)\,d^2k\in\mathbb Z", "TONG-SOLIDSTATE", "https://www.damtp.cam.ac.uk/user/tong/solidstate.html", "chern_number", "established"),
        (r"H_{\mathrm{BCS}}=\sum_{\mathbf k\sigma}\xi_{\mathbf k}c_{\mathbf k\sigma}^\dagger c_{\mathbf k\sigma}-\sum_{\mathbf k}(\Delta c_{\mathbf k\uparrow}^\dagger c_{-\mathbf k\downarrow}^\dagger+\mathrm{h.c.})", "TONG-SOLIDSTATE", "https://www.damtp.cam.ac.uk/user/tong/solidstate.html", "bcs_mean_field_hamiltonian", "model_dependent"),
        (r"E_{\mathbf k}=\sqrt{\xi_{\mathbf k}^2+|\Delta|^2}", "TONG-SOLIDSTATE", "https://www.damtp.cam.ac.uk/user/tong/solidstate.html", "bcs_quasiparticle_spectrum", "model_dependent"),
    ],
    "string_quantum_gravity": [
        (r"\gamma_{\alpha\beta}=\partial_\alpha X^\mu\partial_\beta X^\nu\eta_{\mu\nu}", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string1.pdf#page=9", "induced_worldsheet_metric", "model_dependent"),
        (r"S_{\mathrm{NG}}=-T\int d^2\sigma\sqrt{-\det\gamma}", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string1.pdf#page=9", "nambu_goto_action", "model_dependent"),
        (r"S_{\mathrm P}=-\frac{T}{2}\int d^2\sigma\sqrt{-h}\,h^{\alpha\beta}\partial_\alpha X^\mu\partial_\beta X_\mu", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string1.pdf#page=14", "polyakov_action", "model_dependent"),
        (r"T=\frac{1}{2\pi\alpha'}", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string1.pdf#page=11", "string_tension", "model_dependent"),
        (r"X^\mu(\sigma+2\pi,\tau)=X^\mu(\sigma,\tau)", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string1.pdf#page=8", "closed_string_boundary", "model_dependent"),
        (r"(\partial_\tau^2-\partial_\sigma^2)X^\mu=0", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string1.pdf#page=17", "worldsheet_wave_equation", "model_dependent"),
        (r"L_0-\tilde L_0=0", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string2.pdf", "level_matching", "model_dependent"),
        (r"M^2=\frac{4}{\alpha'}(N-a)=\frac{4}{\alpha'}(\tilde N-a)", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string2.pdf", "closed_string_mass_spectrum", "model_dependent"),
        (r"R\longleftrightarrow\frac{\alpha'}{R},\qquad n\longleftrightarrow m", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string8.pdf#page=7", "t_duality", "model_dependent"),
        (r"p_L=\frac{n}{R}+\frac{mR}{\alpha'},\qquad p_R=\frac{n}{R}-\frac{mR}{\alpha'}", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string8.pdf#page=4", "compactified_momenta", "model_dependent"),
        (r"\beta^G_{\mu\nu}=\alpha'R_{\mu\nu}+O(\alpha'^2)=0", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string7.pdf", "metric_beta_function", "model_dependent"),
        (r"S_{\mathrm{eff}}=\frac{1}{2\kappa^2}\int d^Dx\sqrt{-G}\,e^{-2\Phi}\left(R+4(\nabla\Phi)^2-\frac{1}{12}H^2+\cdots\right)", "TONG-STRING", "https://davidtong.org/pdfs/teaching/string-theory/string7.pdf", "low_energy_effective_action", "model_dependent"),
    ],
}


def ensure_sources() -> None:
    path = REFERENCE / "sources.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    by_id = {source["id"]: source for source in document["sources"]}
    for source in NEW_SOURCES:
        by_id[source["id"]] = source
    document["sources"] = [by_id[key] for key in sorted(by_id)]
    document["metadata_version"] = "3.0.0"
    document["coverage_note"] = (
        "Research coverage is broad and explicitly non-exhaustive. Formula-only records retain pinned source locators; no exposition or derivations are copied."
    )
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def existing_keys() -> set[str]:
    keys: set[str] = set()
    for path in REFERENCE.rglob("equations.jsonl"):
        if "research" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            keys.add(normalized_key(json.loads(line)["formula"]))
    return keys


def candidate_path(locator: str) -> str:
    match = re.search(r"/blob/[0-9a-f]{40}/([^#]+)", locator)
    return match.group(1) if match else locator


def add_record(
    records: dict[tuple[str, str], list[dict]],
    used: set[str],
    domain: str,
    topic: str,
    candidate: Candidate,
    *,
    status: str,
    unit_system: str,
    conventions: tuple[str, ...] = (),
    forced_type: str | None = None,
) -> bool:
    formula = candidate.formula
    key = normalized_key(formula)
    if key in used or not meaningful(formula):
        return False
    used.add(key)
    collection = records[(domain, topic)]
    prefix = "MATH" if domain == "mathematics" else "PHY"
    code = TOPIC_CODES[(domain, topic)]
    record = {
        "id": f"{prefix}-RS-{code}-{len(collection) + 1:06d}",
        "domain": domain,
        "topic": topic,
        "formula": formula,
        "subtopic": candidate.subtopic[:96],
        "source_level": "research",
        "source_ids": [candidate.source_id],
        "relation_type": forced_type or relation_type(formula),
        "status": status,
        "source_locator": candidate.locator,
        "unit_system": unit_system,
    }
    if candidate.assumptions:
        record["assumptions"] = list(candidate.assumptions)
    if conventions:
        record["conventions"] = list(conventions)
    collection.append(record)
    return True


def take(
    candidates: Iterable[Candidate], target: int, records: dict[tuple[str, str], list[dict]],
    used: set[str], domain: str, topic_fn, *, status: str, unit_system: str,
    conventions: tuple[str, ...] = (),
) -> int:
    added = 0
    for candidate in candidates:
        topic = topic_fn(candidate)
        if add_record(records, used, domain, topic, candidate, status=status,
                      unit_system=unit_system, conventions=conventions):
            added += 1
            if added >= target:
                break
    return added


def build() -> dict[tuple[str, str], list[dict]]:
    missing = [str(path) for path in [DLMF_ROOT, *SOURCE_ROOTS.values()] if not path.exists()]
    if missing:
        raise SystemExit("missing pinned extraction inputs: " + ", ".join(missing))
    records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    used = existing_keys()

    # DLMF: take a balanced slice across all 36 chapters, preserving numbered permalinks.
    by_chapter: dict[int, list[Candidate]] = defaultdict(list)
    for chapter, candidate in dlmf_candidates():
        by_chapter[chapter].append(candidate)
    selected: list[tuple[int, Candidate]] = []
    cursor = 0
    while len(selected) < 1800:
        progressed = False
        for chapter in sorted(by_chapter):
            if cursor < len(by_chapter[chapter]):
                selected.append((chapter, by_chapter[chapter][cursor]))
                progressed = True
                if len(selected) >= 1800:
                    break
        if not progressed:
            break
        cursor += 1
    for chapter, candidate in selected:
        add_record(records, used, "mathematics", math_topic_for_dlmf(chapter), candidate,
                   status="established", unit_system="not_applicable")

    # Stacks Project display relations: immutable commit line anchors.
    stacks_root = SOURCE_ROOTS["STACKS-PROJECT"]
    stacks = display_candidates("STACKS-PROJECT", stacks_root.glob("*.tex"))
    stacks_groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in stacks:
        path = candidate_path(candidate.locator).lower()
        if any(word in path for word in ("categor", "homolog", "derived", "simplicial")):
            topic = "category_homological_algebra"
        elif any(word in path for word in ("topolog", "sites", "topology")):
            topic = "algebraic_topology"
        else:
            topic = "algebraic_geometry"
        stacks_groups[topic].append(candidate)
    for topic, quota in (("algebraic_geometry", 360), ("category_homological_algebra", 260), ("algebraic_topology", 130)):
        for candidate in stacks_groups[topic]:
            if len([r for r in records[("mathematics", topic)] if r["source_ids"] == ["STACKS-PROJECT"]]) >= quota:
                break
            add_record(records, used, "mathematics", topic, candidate,
                       status="established", unit_system="not_applicable")

    # Schuller formula displays supply independently located geometry and operator relations.
    geo_root = SOURCE_ROOTS["SCHULLER-GEOMETRIC"]
    geo = display_candidates("SCHULLER-GEOMETRIC", geo_root.joinpath("ga").glob("*.tex"))
    take(geo, 280, records, used, "mathematics",
         lambda candidate: geometric_topic(candidate_path(candidate.locator)),
         status="established", unit_system="not_applicable")
    qm_root = SOURCE_ROOTS["SCHULLER-QUANTUM"]
    qm_math = display_candidates("SCHULLER-QUANTUM", qm_root.joinpath("qt").glob("*.tex"))
    take(qm_math, 180, records, used, "mathematics",
         lambda candidate: quantum_math_topic(candidate_path(candidate.locator)),
         status="established", unit_system="not_applicable")

    for topic, seeds in MATH_SEEDS.items():
        for formula, source_id, locator, subtopic in seeds:
            add_record(records, used, "mathematics", topic,
                       Candidate(formula, source_id, locator, subtopic),
                       status="established", unit_system="not_applicable")

    natural_convention = ("Natural-unit notation and source normalization are retained; no unit conversion was performed.",)
    source_convention = ("Units and sign/index conventions are retained from the cited source; no conversion was performed.",)

    # Physics sources are confined to their academically relevant subfields.
    qft_root = SOURCE_ROOTS["UOFT-QFT-JOOT"]
    qft = display_candidates("UOFT-QFT-JOOT", qft_root.glob("*.tex"))
    qft_counts = Counter()
    for candidate in qft:
        path = candidate_path(candidate.locator).lower()
        topic = "gauge_theory" if any(word in path for word in ("gauge", "yangmills", "faddeev", "ward")) else "quantum_field_theory"
        limits = {"gauge_theory": 130, "quantum_field_theory": 370}
        if qft_counts[topic] >= limits[topic]:
            continue
        if add_record(records, used, "physics", topic, candidate, status="model_dependent",
                      unit_system="natural", conventions=natural_convention):
            qft_counts[topic] += 1

    sm_root = SOURCE_ROOTS["RESTREPO-QFT-SM"]
    sm_paths = [path for path in sm_root.rglob("*.tex") if not any(x in path.name.lower() for x in ("beamer", ".p.", "sl.tex")) and "tikz" not in path.name.lower()]
    sm = display_candidates("RESTREPO-QFT-SM", sm_paths)
    sm_counts = Counter()
    for candidate in sm:
        path = candidate_path(candidate.locator).lower()
        if "leptogenesis" in path:
            topic = "cosmology"
        elif any(word in path for word in ("feynman", "propagator", "smatrix", "quantization")):
            topic = "quantum_field_theory"
        elif any(word in path for word in ("gauge", "cft")):
            topic = "gauge_theory"
        else:
            topic = "standard_model_qcd"
        limits = {"cosmology": 90, "quantum_field_theory": 100, "gauge_theory": 80, "standard_model_qcd": 330}
        if sm_counts[topic] >= limits[topic]:
            continue
        if add_record(records, used, "physics", topic, candidate, status="model_dependent",
                      unit_system="natural", conventions=natural_convention):
            sm_counts[topic] += 1

    plasma_root = SOURCE_ROOTS["FITZPATRICK-PLASMA"]
    plasma = display_candidates("FITZPATRICK-PLASMA", plasma_root.glob("Chapter*/Chapter*.tex"))
    plasma_counts = Counter()
    for candidate in plasma:
        path = candidate_path(candidate.locator)
        topic = "fluid_physics" if any(chapter in path for chapter in ("Chapter03", "Chapter05")) else "plasma_physics"
        limits = {"fluid_physics": 150, "plasma_physics": 300}
        if plasma_counts[topic] >= limits[topic]:
            continue
        if add_record(records, used, "physics", topic, candidate, status="model_dependent",
                      unit_system="source_defined", conventions=source_convention):
            plasma_counts[topic] += 1

    stars_root = SOURCE_ROOTS["MSU-STELLAR"]
    star_paths = [path for path in stars_root.rglob("*.tex") if "mesa-" not in path.name and path.name not in {"stellar-notes.tex", "sample-handout.tex", "symbols.tex"}]
    stars = display_candidates("MSU-STELLAR", star_paths)
    star_counts = Counter()
    for candidate in stars:
        path = candidate_path(candidate.locator).lower()
        if "/nuclear/" in path:
            topic = "nuclear_physics"
        elif any(word in path for word in ("/eos/", "thermodynamics", "/convection/")):
            topic = "statistical_mechanics"
        elif "/plasma/" in path:
            topic = "plasma_physics"
        else:
            topic = "astrophysics"
        limits = {"nuclear_physics": 70, "statistical_mechanics": 110, "plasma_physics": 40, "astrophysics": 300}
        if star_counts[topic] >= limits[topic]:
            continue
        if add_record(records, used, "physics", topic, candidate, status="model_dependent",
                      unit_system="source_defined", conventions=source_convention):
            star_counts[topic] += 1

    qinfo_root = SOURCE_ROOTS["KEIO-QUANTUM-COMMS"]
    qinfo_paths = [path for path in qinfo_root.glob("CH*.tex")]
    qinfo = display_candidates("KEIO-QUANTUM-COMMS", qinfo_paths)
    qinfo_counts = Counter()
    for candidate in qinfo:
        path = Path(candidate_path(candidate.locator)).name
        topic = "quantum_mechanics" if path.startswith(("CH02", "CH03")) else "quantum_information_optics"
        limits = {"quantum_mechanics": 80, "quantum_information_optics": 190}
        if qinfo_counts[topic] >= limits[topic]:
            continue
        if add_record(records, used, "physics", topic, candidate, status="established",
                      unit_system="source_defined", conventions=source_convention):
            qinfo_counts[topic] += 1

    nuclear_root = SOURCE_ROOTS["MIT-NUCLEAR-NOTES"]
    nuclear = display_candidates("MIT-NUCLEAR-NOTES", nuclear_root.rglob("*.tex"))
    take(nuclear, 240, records, used, "physics", lambda _: "nuclear_physics",
         status="model_dependent", unit_system="source_defined", conventions=source_convention)

    many_root = SOURCE_ROOTS["NUCLEAR-TALENT-MANYBODY"]
    many_paths = list(many_root.glob("doc/src/*/*-plain-print.tex"))
    many = display_candidates("NUCLEAR-TALENT-MANYBODY", many_paths)
    take(many, 400, records, used, "physics", lambda _: "many_body_physics",
         status="model_dependent", unit_system="source_defined", conventions=source_convention)

    gr_root = SOURCE_ROOTS["WILTSHIRE-GR"]
    gr_paths = list(gr_root.glob("chapters/*.tex"))
    gr = display_candidates("WILTSHIRE-GR", gr_paths)
    take(gr, 35, records, used, "physics", lambda _: "general_relativity",
         status="model_dependent", unit_system="source_defined", conventions=source_convention)

    nonlinear_root = SOURCE_ROOTS["YACHAY-NONLINEAR"]
    nonlinear = notebook_candidates("YACHAY-NONLINEAR", nonlinear_root.rglob("*.ipynb"))
    take(nonlinear, 50, records, used, "physics", lambda _: "nonlinear_physics",
         status="model_dependent", unit_system="dimensionless",
         conventions=("Dimensionless variables and scaling are retained from the cited notebook.",))

    for topic, seeds in PHYSICS_SEEDS.items():
        for formula, source_id, locator, subtopic, status in seeds:
            unit = "natural" if topic == "string_quantum_gravity" else "source_defined"
            convention = natural_convention if unit == "natural" else source_convention
            add_record(records, used, "physics", topic,
                       Candidate(formula, source_id, locator, subtopic), status=status,
                       unit_system=unit, conventions=convention)

    return records


def write_records(records: dict[tuple[str, str], list[dict]]) -> None:
    research_root = REFERENCE / "research"
    if research_root.exists():
        for path in research_root.rglob("equations.jsonl"):
            path.unlink()
    for domain, topics in LEVEL_TOPICS["research"].items():
        for topic in sorted(topics):
            rows = records[(domain, topic)]
            path = research_root / domain / topic / "equations.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def write_coverage(records: dict[tuple[str, str], list[dict]]) -> None:
    domains: dict[str, dict[str, dict]] = {}
    for domain, topics in LEVEL_TOPICS["research"].items():
        domains[domain] = {}
        for topic in sorted(topics):
            rows = records[(domain, topic)]
            source_ids = sorted({source for row in rows for source in row["source_ids"]})
            # Automated formula-level extraction gives broad but not exhaustive research coverage.
            status = "covered" if len(rows) >= 25 else "partial" if rows else "not covered"
            domains[domain][topic] = {
                "records": len(rows), "source_count": len(source_ids),
                "source_ids": source_ids, "status": status,
            }
    document = {
        "coverage_version": "1.0.0", "corpus_version": "3.0.0",
        "generated_on": "2026-10-05",
        "claim": "Broad formula-reference coverage only; not exhaustive research coverage.",
        "status_policy": {"covered": "at least 25 distinct relations", "partial": "1-24 distinct relations", "not covered": "zero relations"},
        "domains": domains,
    }
    (REFERENCE / "research_coverage.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    ensure_sources()
    records = build()
    write_records(records)
    write_coverage(records)
    counts = Counter({f"{domain}/{topic}": len(rows) for (domain, topic), rows in records.items()})
    domains = Counter()
    for (domain, _), rows in records.items():
        domains[domain] += len(rows)
    print(json.dumps({"domains": domains, "total": sum(domains.values()), "topics": counts}, indent=2, sort_keys=True))
    uncovered = [key for key, value in counts.items() if value == 0]
    if uncovered:
        raise SystemExit("uncovered research topics: " + ", ".join(uncovered))
    if domains["mathematics"] < 2500 or domains["physics"] < 2500:
        raise SystemExit(f"research target not met: {dict(domains)}")


if __name__ == "__main__":
    main()
