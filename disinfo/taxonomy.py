"""Figures 1-6 of the survey as data, with renderers.

All six figures are diagrams, not data plots: Fig. 1 is the four-stage pipeline,
Fig. 3 a GNN schematic, and Figs. 2, 4, 5 and 6 are taxonomy trees. Encoding
them as structures rather than redrawing them by hand has two payoffs beyond
reproduction: the trees become navigable (``describe``, ``leaves``, ``path_to``)
so the notebook can look concepts up in them, and Figs. 5 and 6 carry the
per-category reference counts that ``plots.py`` turns into the histogram the
survey never draws.

In Figs. 5 and 6 each leaf entry is printed as ``[i]-j`` where ``i`` is the
survey's reference number and ``j`` the publication year; those are stored as
``(i, j)`` pairs. A trailing ``...`` box in the figure means the list is
truncated by the authors, recorded as ``truncated=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Node", "FIG1_STAGES", "FIG2_FALSE_INFORMATION", "FIG3_EXAMPLE",
           "FIG4_FEATURES", "FIG5_APPROACHES", "FIG6_ALGORITHMS", "FIGURES",
           "leaves", "describe", "path_to", "ref_counts"]


@dataclass
class Node:
    """One box in a taxonomy tree."""
    label: str
    children: list["Node"] = field(default_factory=list)
    refs: list[tuple[int, int]] = field(default_factory=list)
    truncated: bool = False          # the figure's trailing "..." box

    def __iter__(self):
        yield self
        for c in self.children:
            yield from c


def _n(label, *children, refs=None, truncated=False):
    return Node(label, list(children), refs or [], truncated)


# --------------------------------------------------------------------------
# Fig. 1 -- the general framework
# --------------------------------------------------------------------------
# "This process can be broadly categorized into four key stages" (Sect. 1).
# The section each stage is elaborated in is the survey's own cross-reference;
# `pipeline.py` implements the four in this order.

FIG1_STAGES = [
    dict(name="Feature Extraction", section="4",
         detail="Initial feature vectors: textual, visual, and so on, via "
                "linguistic models and CNNs."),
    dict(name="Graph Construction", section="5.3",
         detail="Similarity graph, propagation graph, or heterogeneous graph."),
    dict(name="GNN", section="3",
         detail="An embedding per node, capturing local structure and content."),
    dict(name="Classification", section="5.1",
         detail="Embeddings into a classifier; output binary (fake/real) or "
                "multiclass (true/false/unverified/non-rumor)."),
]


# --------------------------------------------------------------------------
# Fig. 2 -- types of false information
# --------------------------------------------------------------------------
# Definitions are quoted from Sect. 2 and attached so the notebook can print
# the taxonomy and its glossary from one structure.

FIG2_DEFINITIONS = {
    "False Information":
        "content that contains false and untrue information, written for "
        "different purposes.",
    "MisInformation":
        "false, mistaken, or misleading information that is often considered "
        "honest mistake... The purpose of misinformation is not to deceive the "
        "audience.",
    "DisInformation":
        "false information that is spread deliberately to mislead and deceive.",
    "Satire":
        "an article that incorporates cleverly crafted allusions. It does not "
        "have a harmful intention, but there is a potential for deception.",
    "Fake News":
        "news articles that are intentionally and verifiably false.",
    "Rumor":
        "information that has not yet been confirmed by official resources... "
        "not necessarily incorrect.",
    "Hoax":
        "messages sent to a wide range of people to persuade or manipulate "
        "them, primarily by using a threat or deception.",
    "Clickbait":
        "using misleading headlines to encourage users to click on a link.",
    "Conspiracy Theory":
        "beliefs primarily dismissed by society, attributing an event to a "
        "group of agents with illegal, hidden, and malicious intent.",
    "Opinion Spam":
        "fake or intentionally biased reviews or comments about products.",
    "Propaganda":
        "biased and misleading information spread following a preconceived "
        "strategy to strengthen a particular viewpoint or political agenda.",
}

FIG2_FALSE_INFORMATION = _n(
    "False Information",
    _n("MisInformation",
       _n("Satire")),
    _n("DisInformation",
       _n("Fake News"),
       _n("Rumor"),
       _n("Other",
          _n("Hoax"), _n("Clickbait"), _n("Conspiracy Theory"),
          _n("Opinion Spam"), _n("Propaganda"))),
)


# --------------------------------------------------------------------------
# Fig. 3 -- the GNN schematic
# --------------------------------------------------------------------------
# Left: a 5-node graph with N(a) circled. Right: the 3-layer computation tree
# for node a. `plots.py` draws the tree by unrolling this graph, so the figure
# is generated from the adjacency rather than transcribed -- if the unrolling
# is wrong, the picture is visibly wrong.

FIG3_EXAMPLE = dict(
    nodes=["a", "b", "c", "d", "e"],
    edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "e"), ("d", "e")],
    root="a",
    layers=3,
    caption="Each rhombus presents a function that consists of a linear "
            "transformation, an aggregation function (sum, mean etc), and an "
            "activation function (ReLU, sigmoid etc).",
)


# --------------------------------------------------------------------------
# Fig. 4 -- types of features
# --------------------------------------------------------------------------
# NOTE: the published figure spells the leaf "Commnet". Corrected to "Comment"
# here; the typo is recorded in docs/DISCREPANCIES_SURVEY.md.

FIG4_FEATURES = _n(
    "Features",
    _n("Content-based",
       _n("Linguistic", _n("Lexical"), _n("Syntactic")),
       _n("Visual"),
       _n("Semantic")),
    _n("Context-based",
       _n("User-based", _n("Comment"), _n("Profile")),
       _n("Network-based", _n("Propagation"), _n("Temporal"), _n("Structural"))),
)


# --------------------------------------------------------------------------
# Fig. 5 -- combating disinformation approaches
# --------------------------------------------------------------------------

FIG5_APPROACHES = _n(
    "Combating Disinformation Approaches",
    _n("Detection",
       _n("Content-based",
          _n("Style-based",
             refs=[(45, 2005), (46, 2009), (49, 2017), (51, 2022), (52, 2023)],
             truncated=True),
          _n("Knowledge-based",
             refs=[(43, 2014), (41, 2015), (42, 2016), (40, 2020), (44, 2023)],
             truncated=True)),
       _n("Context-based",
          _n("Stance-based",
             refs=[(53, 2011), (54, 2016), (56, 2018), (57, 2022), (58, 2023)],
             truncated=True),
          _n("Propagation-based",
             refs=[(59, 2012), (60, 2014), (61, 2018), (62, 2022), (63, 2023)],
             truncated=True)),
       _n("Hybrid",
          refs=[(65, 2021), (64, 2022), (66, 2022), (67, 2022), (68, 2023)],
          truncated=True)),
    _n("Intervention",
       _n("Source-based",
          refs=[(69, 2012), (71, 2015), (72, 2018), (73, 2022), (74, 2023)],
          truncated=True),
       _n("User-based",
          refs=[(75, 2013), (77, 2018), (78, 2020), (79, 2022), (80, 2023)],
          truncated=True)),
)


# --------------------------------------------------------------------------
# Fig. 6 -- algorithm-based categorization
# --------------------------------------------------------------------------

FIG6_ALGORITHMS = _n(
    "Disinformation Detection Approaches",
    _n("ML-based",
       _n("Traditional ML-based",
          _n("SVM", refs=[(87, 2012), (89, 2016), (90, 2022), (91, 2023)],
             truncated=True),
          _n("RF", refs=[(88, 2014), (89, 2016), (90, 2022), (92, 2023)],
             truncated=True),
          _n("DT", refs=[(85, 2011), (89, 2016), (93, 2023)], truncated=True),
          _n("LR", refs=[(89, 2016), (90, 2022), (93, 2023)], truncated=True),
          _n("...")),
       _n("DL-based",
          _n("Non GNN",
             _n("CNN", refs=[(94, 2017), (95, 2022), (96, 2023)], truncated=True),
             _n("RNN", refs=[(97, 2016), (95, 2022), (98, 2023)], truncated=True),
             _n("Transformers",
                refs=[(99, 2019), (101, 2021), (103, 2022), (104, 2023)],
                truncated=True),
             _n("...")),
          _n("GNN",
             _n("Homogeneous Graph",
                _n("Similarity Graph",
                   refs=[(105, 2019), (40, 2020), (106, 2022), (107, 2023)],
                   truncated=True),
                _n("Propagation Graph",
                   refs=[(108, 2019), (109, 2021), (110, 2022)],
                   truncated=True)),
             _n("Heterogeneous Graph",
                refs=[(111, 2020), (67, 2022), (112, 2022), (113, 2023)],
                truncated=True)))),
    _n("Non-ML-based",
       _n("Cognitive Psychology", refs=[(81, 2014), (84, 2018)], truncated=True),
       _n("Rhetorical Approach", refs=[(82, 2015), (83, 2017)], truncated=True)),
)


FIGURES = {
    2: ("Categorization of various types of false information",
        FIG2_FALSE_INFORMATION),
    4: ("Different types of features used in the literature for disinformation "
        "detection", FIG4_FEATURES),
    5: ("A categorization of combating disinformation approaches",
        FIG5_APPROACHES),
    6: ("Algorithm-based categorization of disinformation detection approaches",
        FIG6_ALGORITHMS),
}


# --------------------------------------------------------------------------
# navigation helpers
# --------------------------------------------------------------------------

def leaves(root: Node) -> list[Node]:
    """Every node with no children."""
    return [n for n in root if not n.children]


def path_to(root: Node, label: str) -> list[str] | None:
    """Labels from ``root`` down to ``label``, or None if absent.

    Lets the notebook answer "where does GraphSAGE sit in Fig. 6?" without the
    reader tracing boxes by eye.
    """
    if root.label == label:
        return [root.label]
    for c in root.children:
        sub = path_to(c, label)
        if sub is not None:
            return [root.label] + sub
    return None


def describe(root: Node, indent: str = "") -> str:
    """Render a tree as indented text -- the accessible form of the figure."""
    out = [f"{indent}{root.label}"]
    if root.refs:
        shown = ", ".join(f"[{i}]-{y}" for i, y in root.refs)
        out[-1] += f"   ({shown}{', ...' if root.truncated else ''})"
    for c in root.children:
        out.append(describe(c, indent + "    "))
    return "\n".join(out)


def ref_counts(root: Node) -> dict[str, int]:
    """Number of references cited under each leaf category of Figs. 5-6.

    Counts only what the figure prints. Every list ends in "..." in the
    original, so these are lower bounds on the true literature and are labelled
    as such wherever they are plotted.
    """
    return {n.label: len(n.refs) for n in root if n.refs}
