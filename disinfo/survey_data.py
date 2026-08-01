"""Tables 1-4 of the survey, transcribed as data, plus the claims they support.

Lakzaei, Haghir Chehreghani & Bagheri (2024), *Artificial Intelligence Review*
57:52.

The survey reports no experiments of its own. Its empirical content is a
meta-analysis: Tables 1-2 record what 34 other papers reported, Table 3
characterises 10 datasets, and Sect. 5.3.2 draws four conclusions from Tables
1-2 in prose without ever plotting them. ``PAPER_CLAIMS`` states those four
conclusions as testable predicates; ``verify_claims`` evaluates them against the
transcribed table. That is the sense in which this survey has reproducible
results, and ``plots.py`` draws the figures the survey implies but omits.

Transcription notes -- places where the printed table is ambiguous or wrong are
collected in ``TRANSCRIPTION_NOTES`` and reproduced in
``docs/DISCREPANCIES_SURVEY.md``. Values are otherwise entered exactly as
printed, including the year column's disagreements with the citation year.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["METHODS", "DATASETS", "EXAMPLES", "PAPER_CLAIMS",
           "TRANSCRIPTION_NOTES", "methods_table", "datasets_table",
           "examples_table", "long_results", "verify_claims", "verify_table3"]


# --------------------------------------------------------------------------
# Tables 1 and 2 -- GNN-based disinformation detection methods
# --------------------------------------------------------------------------
# Columns follow the printed header exactly:
#   ref, year, gnn, graph, features, approach, setting, disinfo_type,
#   results = [(dataset, metric, value), ...]
#
# `year` is the survey's own Year column. It disagrees with the citation year
# for three rows (Autef 2020/2019, Bai 2021/2020, Zhiyuan 2020/2020); both are
# kept so neither reading is lost. `table` records which of the two printed
# tables the row came from.

_M = [
    dict(ref="Benamira et al.", cite_year=2019, year=2019, table=1,
         gnn=["GCN", "GAT"], graph="Similarity", features=["Textual"],
         approach="Content-based", setting="Semi-supervised",
         disinfo_type="Fake news",
         results=[("Custom DS", "ACC", 0.849)]),
    dict(ref="Hu et al.", cite_year=2019, year=2019, table=1,
         gnn=["GCN"], graph="Similarity", features=["Textual", "Profile"],
         approach="Content-based", setting="Semi-supervised",
         disinfo_type="Fake news",
         results=[("LIAR", "ACC", 0.492)]),
    dict(ref="Autef et al.", cite_year=2020, year=2019, table=1,
         gnn=["GCN", "GAT", "GraphSage"], graph="Propagation",
         features=["Textual", "Profile"], approach="Context-based",
         setting="Supervised", disinfo_type="Fake news",
         results=[("Twitter15", "ACC", 0.690), ("Twitter16", "ACC", 0.750)]),
    dict(ref="Huang et al.", cite_year=2019, year=2019, table=1,
         gnn=["GCN"], graph="Propagation", features=["Textual", "Profile"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter15", "ACC", 0.752), ("Twitter16", "ACC", 0.773)]),
    dict(ref="Han et al.", cite_year=2020, year=2020, table=1,
         gnn=["GraphSage"], graph="Propagation", features=["Profile", "Temporal"],
         approach="Context-based", setting="Supervised", disinfo_type="Fake news",
         results=[("PolitiFact", "ACC", 0.803), ("GossipCop", "ACC", 0.825)]),
    dict(ref="Bian et al.", cite_year=2020, year=2020, table=1,
         gnn=["GCN"], graph="Propagation", features=["Textual", "Profile"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Weibo", "ACC", 0.961), ("Twitter15", "ACC", 0.886),
                  ("Twitter16", "ACC", 0.880)]),
    dict(ref="Wang et al.", cite_year=2020, year=2020, table=1,
         gnn=["GCN"], graph="Similarity",
         features=["Textual", "Visual", "Semantic"],
         approach="Content-based", setting="Supervised", disinfo_type="Fake news",
         results=[("Weibo", "ACC", 0.886), ("PHEME", "ACC", 0.876)]),
    dict(ref="Shang et al.", cite_year=2020, year=2020, table=1,
         gnn=["GCN"], graph="Similarity", features=["Comments"],
         approach="Context-based", setting="Supervised", disinfo_type="Faux",
         results=[("Reddit", "ACC", 0.754), ("Twitter", "ACC", 0.711)]),
    dict(ref="Zhiyuan et al.", cite_year=2020, year=2020, table=1,
         gnn=["GAT"], graph="Propagation", features=["Textual"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("PHEME", "Macro-F1", 0.753)]),
    dict(ref="Bai et al.", cite_year=2021, year=2020, table=1,
         gnn=["GCN"], graph="Propagation", features=["Textual"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("PHEME", "ACC", 0.841)]),
    dict(ref="Malhotra et al.", cite_year=2020, year=2020, table=1,
         gnn=["GCN"], graph="Propagation", features=["Textual", "Profile"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter15", "ACC", 0.866), ("Twitter16", "ACC", 0.865)]),
    dict(ref="Lin et al.", cite_year=2020, year=2020, table=1,
         gnn=["GCN"], graph="Propagation", features=["Textual"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter15", "ACC", 0.856), ("Twitter16", "ACC", 0.881)]),
    dict(ref="Ren et al.", cite_year=2020, year=2020, table=1,
         gnn=["GAT"], graph="Heterogeneous", features=["Textual", "Profile"],
         approach="Content-based", setting="Semi-supervised",
         disinfo_type="Fake news",
         results=[("PolitiFact", "ACC", 0.615), ("BuzzFeed", "ACC", 0.735)]),
    dict(ref="Huang et al.", cite_year=2020, year=2020, table=1,
         gnn=["GAT"], graph="Heterogeneous",
         features=["Textual", "Profile", "Temporal"],
         approach="Content-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter15", "ACC", 0.911), ("Twitter16", "ACC", 0.924)]),
    dict(ref="Ke et al.", cite_year=2020, year=2020, table=1,
         gnn=["GCN"], graph="Propagation", features=["Textual", "Profile"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Weibo", "ACC", 0.969)]),
    dict(ref="Nguyen et al.", cite_year=2020, year=2020, table=1,
         gnn=["GraphSage"], graph="Heterogeneous", features=["Textual", "Profile"],
         approach="Hybrid", setting="Supervised", disinfo_type="Fake news",
         results=[("Twitter", "AUC", 0.752)]),
    dict(ref="Lu and Li", cite_year=2020, year=2020, table=1,
         gnn=["GCN"], graph="Similarity", features=["Textual", "Profile"],
         approach="Hybrid", setting="Supervised", disinfo_type="Fake news",
         results=[("Twitter15", "ACC", 0.877), ("Twitter16", "ACC", 0.908)]),
    dict(ref="Song et al.", cite_year=2021, year=2021, table=1,
         gnn=["GAT", "GCN"], graph="Propagation", features=["Textual", "Temporal"],
         approach="Context-based", setting="Supervised", disinfo_type="Fake news",
         results=[("Weibo", "ACC", 0.968), ("FakeNewsNet", "ACC", 0.935)]),
    dict(ref="Yuan et al.", cite_year=2021, year=2021, table=1,
         gnn=["GAT"], graph="Similarity", features=["Textual", "Visual"],
         approach="Content-based", setting="Supervised", disinfo_type="Fake news",
         results=[("Twitter", "Macro-F1", 0.883), ("Weibo", "Macro-F1", 0.952)]),
    dict(ref="Wei et al.", cite_year=2021, year=2021, table=1,
         gnn=["GCN"], graph="Propagation", features=["Textual"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter15", "ACC", 0.892), ("Twitter16", "ACC", 0.915),
                  ("PHEME", "ACC", 0.715)]),
    dict(ref="Jiang et al.", cite_year=2021, year=2021, table=1,
         gnn=["GAT"], graph="Propagation", features=["Textual"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter15", "ACC", 0.851), ("Twitter16", "ACC", 0.883)]),
    dict(ref="Choi et al.", cite_year=2021, year=2021, table=1,
         gnn=["GCN"], graph="Propagation", features=["Textual"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter15", "ACC", 0.827), ("Twitter16", "ACC", 0.836),
                  ("Weibo", "ACC", 0.936)]),

    # ---- Table 2 ----
    dict(ref="Ran et al.", cite_year=2022, year=2022, table=2,
         gnn=["GAT"], graph="Heterogeneous",
         features=["Textual", "Profile", "Temporal"],
         approach="Hybrid", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter15", "ACC", 0.908), ("Twitter16", "ACC", 0.916)]),
    dict(ref="Song et al.", cite_year=2022, year=2022, table=2,
         gnn=["GAT"], graph="Propagation", features=["Textual"],
         approach="Context-based", setting="Supervised", disinfo_type="Fake news",
         results=[("Weibo", "ACC", 0.957), ("FakeNewsNet", "ACC", 0.922),
                  ("Twitter", "ACC", 0.899)]),
    dict(ref="Liu et al.", cite_year=2022, year=2022, table=2,
         gnn=["GAT"], graph="Heterogeneous",
         features=["Textual", "Profile", "Comments"],
         approach="Hybrid", setting="Supervised", disinfo_type="Rumor",
         results=[("Weibo", "ACC", 0.944), ("Twitter15", "ACC", 0.905),
                  ("Twitter16", "ACC", 0.902)]),
    dict(ref="Wei et al.", cite_year=2022, year=2022, table=2,
         gnn=["GCN"], graph="Propagation", features=["Textual"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter15", "ACC", 0.901), ("Twitter16", "ACC", 0.908),
                  ("PHEME", "ACC", 0.694)]),
    dict(ref="Zheng et al.", cite_year=2022, year=2022, table=2,
         gnn=["GAT"], graph="Heterogeneous",
         features=["Textual", "Visual", "Comments"],
         approach="Hybrid", setting="Supervised", disinfo_type="Rumor",
         results=[("PHEME", "ACC", 0.887), ("Weibo", "ACC", 0.889)]),
    dict(ref="Inan", cite_year=2022, year=2022, table=2,
         gnn=["GAT"], graph="Heterogeneous",
         features=["Textual", "Profile", "Comments"],
         approach="Content-based", setting="Supervised", disinfo_type="Fake news",
         results=[("PolitiFact", "ACC", 0.874), ("GossipCop", "ACC", 0.802)]),
    dict(ref="Weizhi et al.", cite_year=2022, year=2022, table=2,
         gnn=["GCN"], graph="Similarity", features=["Textual", "Profile"],
         approach="Content-based", setting="Supervised", disinfo_type="Fake news",
         results=[("Snopes", "Macro-F1", 0.800),
                  ("PolitiFact", "Macro-F1", 0.691)]),
    dict(ref="Paraschiv et al.", cite_year=2022, year=2022, table=2,
         gnn=["GCN", "GAT", "GraphSage"], graph="Propagation",
         features=["Textual", "Profile", "Comments"],
         approach="Hybrid", setting="Supervised", disinfo_type="Fake news",
         results=[("US Elections dataset", "ACC", 0.867)]),
    dict(ref="Xu et al.", cite_year=2022, year=2022, table=2,
         gnn=["GCN"], graph="Propagation", features=["Textual"],
         approach="Hybrid", setting="Supervised", disinfo_type="Rumor",
         results=[("Weibo", "ACC", 0.957), ("CED", "ACC", 0.882)]),
    dict(ref="Yang et al.", cite_year=2023, year=2023, table=2,
         gnn=["GAT"], graph="Propagation", features=["Textual", "Temporal"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("PHEME", "ACC", 0.882), ("Weibo", "ACC", 0.972)]),
    dict(ref="Cui et al.", cite_year=2023, year=2023, table=2,
         gnn=["GGNN"], graph="Similarity", features=["Textual", "Semantic"],
         approach="Content-based", setting="Supervised", disinfo_type="Fake news",
         results=[("LIAR", "ACC", 0.868), ("Constraint", "ACC", 0.918),
                  ("Twitter15", "ACC", 0.946), ("Twitter16", "ACC", 0.968)]),
    dict(ref="Thota et al.", cite_year=2023, year=2023, table=2,
         gnn=["GCN"], graph="Propagation", features=["Textual", "Profile"],
         approach="Context-based", setting="Supervised", disinfo_type="Rumor",
         results=[("Twitter", "ACC", 0.800), ("Weibo", "ACC", 0.911)]),
]

METHODS = _M


# --------------------------------------------------------------------------
# Table 3 -- datasets
# --------------------------------------------------------------------------
# `size` and `n_labels` are the survey's printed values. `verify_table3` checks
# them against what the download scripts actually produce.

DATASETS = [
    dict(name="LIAR", domain=["politics"], content=["text"], size=12836,
         platform="politifact.com",
         labels=["true", "mostly-true", "half-true", "barely-true", "false",
                 "pants-fire"]),
    dict(name="BuzzFeed News", domain=["politics"], content=["text"], size=2282,
         platform="facebook",
         labels=["mostly true", "mixture of true and false", "mostly false",
                 "no factual content"]),
    dict(name="Yelp", domain=["technology"], content=["text"], size=45954,
         platform="yelp.com", labels=["fake", "real"]),
    dict(name="Reddit", domain=["society"], content=["text"], size=2780,
         platform="reddit.com", labels=["fake", "real"]),
    dict(name="Fakeddit", domain=["politics", "society"],
         content=["text", "image", "videos"], size=795108, platform="reddit.com",
         labels=["True", "Satire/Parody", "Misleading Content",
                 "Imposter Content", "False Connection", "Manipulated Content"]),
    dict(name="PHEME", domain=["politics", "society"],
         content=["text", "propagation graph"], size=6425, platform="twitter",
         labels=["True", "False", "Unverified", "non-rumor"]),
    dict(name="Twitter15", domain=["society"],
         content=["text", "propagation graph"], size=1490, platform="twitter",
         labels=["True", "False", "Unverified", "non-rumor"]),
    dict(name="Twitter16", domain=["society"],
         content=["text", "propagation graph"], size=818, platform="twitter",
         labels=["True", "False", "Unverified", "non-rumor"]),
    dict(name="Sina Weibo", domain=["society"],
         content=["text", "images", "propagation graph"], size=4664,
         platform="weibo", labels=["fake", "real"]),
    dict(name="FakeNewsNet", domain=["politics", "celebrities"],
         content=["text", "images", "propagation graph"], size=23901,
         platform="politifact.com / gossipcop.com", labels=["fake", "real"]),
]


# --------------------------------------------------------------------------
# Table 4 -- example records
# --------------------------------------------------------------------------

EXAMPLES = [
    ("LIAR", "Walking Dead: In the case of a catastrophic event, the "
             "Atlanta-area offices of the Centers for Disease Control and "
             "Prevention will self-destruct.", "pants-fire"),
    ("LIAR", "Donald Trump: “NATO is opening up a major terror division. "
             "...Im sure Im not going to get credit for it, but that was "
             "largely because of what I was saying and my criticism of "
             "NATO.”", "false"),
    ("LIAR", "Robin Vos: “The Chicago Bears have had more starting "
             "quarterbacks in the last 10 years than the total number of "
             "tenured (UW) faculty fired during the last two decades.”",
     "true"),
    ("LIAR", "Republican Party Texas: “Jim Dunnam has not lived in the "
             "district he represents for years now.”", "barely-true"),
    ("LIAR", "Scott Surovell: “When did the decline of coal start? It "
             "started when natural gas took off that started to begin in "
             "(President George W.) Bushs administration.”", "half-true"),
    ("LIAR", "Barack Obama: “Hillary Clinton agrees with John McCain "
             "“by voting to give George Bush the benefit of the doubt on "
             "Iran.”", "mostly-true"),
    ("Twitter15", "rip elly may clampett: so sad to learn #beverlyhillbillies "
                  "star donna douglas has passed away", "true"),
    ("Twitter15", "seriously? racist mcdonald's sign is obviously a hoax.",
     "false"),
    ("Twitter15", "an open letter to trump voters from his top "
                  "strategist-turned-defector URL via @xojanedotcom",
     "unverified"),
    ("Twitter15", "brandon marshall visits and offers advice, support to "
                  "brother of fallen hero zaevion dobson", "non-rumor"),
]


# --------------------------------------------------------------------------
# The four conclusions of Sect. 5.3.2, as testable predicates
# --------------------------------------------------------------------------

PAPER_CLAIMS = {
    "first_year_2019": dict(
        quote="GNN is a novel technique for disinformation detection, with its "
              "first research being presented in 2019.",
        section="5.3.2"),
    "gcn_gat_dominate": dict(
        quote="GCN and GAT stand out as the most widely utilized graph neural "
              "networks.",
        section="5.3.2"),
    "propagation_majority": dict(
        quote="The majority of methods employ the propagation graph, which is a "
              "homogeneous graph, placing them in the category of context-based "
              "methods.",
        section="5.3.2"),
    "textual_majority": dict(
        quote="The majority of methods rely on textual features, and user "
              "profile features are also commonly utilized in many approaches. "
              "However, features such as comments, semantic characteristics and "
              "temporal aspects have been given less consideration.",
        section="5.3.2"),
    "supervised_majority": dict(
        quote="Most methods are based on supervised learning, where only a few "
              "methods harness the potential of GNNs in semi-supervised "
              "learning.",
        section="5.3.2"),
    "multiclass_below_50": dict(
        quote="only a limited number of researchers have dedicated their efforts "
              "to multiclass classification and as a result, in this setting "
              "existing algorithms suffer from relatively low accuracy rates, "
              "typically below 50%.",
        section="7"),
}


TRANSCRIPTION_NOTES = [
    "Table 1, Autef et al.: the Year column says 2019 but the reference is "
    "dated 2020. Both are kept, as `year` and `cite_year`.",
    "Table 1, Bai et al.: Year column says 2020, reference dated 2021.",
    "Table 1 row 'Song et al. (2021)' prints the dataset as 'FakeNewsNe', "
    "truncated in the PDF's column. Read as FakeNewsNet, matching the Table 2 "
    "row for the same authors.",
    "Table 2, Cui et al.: LIAR ACC 0.868 is irreconcilable with Sect. 7's claim "
    "that multiclass accuracy is 'typically below 50%' and with the LIAR row of "
    "Table 1 (Hu et al., ACC 0.492). 0.868 is a plausible *binary* LIAR number "
    "and an implausible 6-class one. Flagged by `verify_claims`; see "
    "docs/DISCREPANCIES_SURVEY.md.",
    "The 'Approach' column is the survey's content/context/hybrid axis of Fig. "
    "5, not the graph type. Ren et al. and Huang et al. (2020) are labelled "
    "'Content-based' while building heterogeneous graphs from user and "
    "propagation structure, which by the survey's own Sect. 4.2 definition is "
    "context. Transcribed as printed.",
    "Performance is a single number per dataset with no standard deviation, "
    "no split protocol and mixed metrics (ACC, Macro-F1, AUC). Cross-row "
    "comparison is therefore indicative only -- see `long_results`.",
]


# --------------------------------------------------------------------------
# DataFrame views
# --------------------------------------------------------------------------

def methods_table() -> pd.DataFrame:
    """Tables 1 and 2 as one wide DataFrame, one row per method."""
    rows = []
    for m in METHODS:
        rows.append({
            "ref": m["ref"], "year": m["year"], "cite_year": m["cite_year"],
            "table": m["table"], "gnn": ", ".join(m["gnn"]),
            "graph": m["graph"], "features": ", ".join(m["features"]),
            "approach": m["approach"], "setting": m["setting"],
            "disinfo_type": m["disinfo_type"],
            "datasets": ", ".join(d for d, _, _ in m["results"]),
            "performance": "; ".join(f"{d} {k}: {v:.3f}"
                                     for d, k, v in m["results"]),
        })
    return pd.DataFrame(rows)


def long_results() -> pd.DataFrame:
    """One row per (method, dataset) result -- the tidy form the plots need."""
    rows = []
    for i, m in enumerate(METHODS):
        for dataset, metric, value in m["results"]:
            rows.append({
                "method_id": i, "ref": m["ref"], "year": m["year"],
                "gnn": ", ".join(m["gnn"]), "graph": m["graph"],
                "approach": m["approach"], "setting": m["setting"],
                "disinfo_type": m["disinfo_type"], "dataset": dataset,
                "metric": metric, "value": value,
            })
    return pd.DataFrame(rows)


def datasets_table() -> pd.DataFrame:
    """Table 3."""
    return pd.DataFrame([{
        "dataset": d["name"], "domain": ", ".join(d["domain"]),
        "content": ", ".join(d["content"]), "n_labels": len(d["labels"]),
        "labels": ", ".join(d["labels"]), "size": d["size"],
        "platform": d["platform"],
        "has_propagation": "propagation graph" in d["content"],
    } for d in DATASETS])


def examples_table() -> pd.DataFrame:
    """Table 4."""
    return pd.DataFrame(EXAMPLES, columns=["dataset", "sample", "label"])


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _explode(column: str) -> pd.Series:
    """Count occurrences of a list-valued field across methods."""
    vals: list[str] = []
    for m in METHODS:
        vals.extend(m[column])
    return pd.Series(vals).value_counts()


def verify_claims() -> pd.DataFrame:
    """Evaluate each claim of ``PAPER_CLAIMS`` against the transcribed tables.

    Returns a DataFrame with the claim, the quantity computed from the table,
    and whether the survey's prose is supported. This is the survey's only
    reproducible empirical content, so it is checked rather than assumed.
    """
    df = methods_table()
    n = len(df)
    out = []

    def add(key, verdict, evidence):
        out.append({"claim": key, "section": PAPER_CLAIMS[key]["section"],
                    "quote": PAPER_CLAIMS[key]["quote"],
                    "supported": verdict, "evidence": evidence})

    # 1. First GNN work in 2019.
    first = int(df["year"].min())
    add("first_year_2019", first == 2019,
        f"earliest Year in Tables 1-2 is {first} "
        f"({', '.join(sorted(df.loc[df['year'] == first, 'ref']))})")

    # 2. GCN and GAT most widely used. Counted per mention, since rows such as
    #    Autef et al. list three architectures.
    gnn = _explode("gnn")
    top2 = set(gnn.index[:2])
    add("gcn_gat_dominate", top2 == {"GCN", "GAT"},
        "mentions: " + ", ".join(f"{k}={v}" for k, v in gnn.items()))

    # 3. Propagation graph is the majority.
    graph = df["graph"].value_counts()
    prop = int(graph.get("Propagation", 0))
    add("propagation_majority", prop > n / 2,
        f"Propagation={prop}/{n} ({prop / n:.0%}); " +
        ", ".join(f"{k}={v}" for k, v in graph.items()))

    # 4. Textual dominant; comments/semantic/temporal neglected.
    feat = _explode("features")
    textual = int(feat.get("Textual", 0))
    rare = {k: int(feat.get(k, 0)) for k in ("Comments", "Semantic", "Temporal")}
    add("textual_majority",
        textual > n / 2 and all(v < textual / 2 for v in rare.values()),
        f"Textual={textual}/{n}, Profile={int(feat.get('Profile', 0))}; " +
        ", ".join(f"{k}={v}" for k, v in rare.items()))

    # 5. Mostly supervised.
    setting = df["setting"].value_counts()
    sup = int(setting.get("Supervised", 0))
    add("supervised_majority", sup > n / 2,
        f"Supervised={sup}/{n}; " + ", ".join(f"{k}={v}" for k, v in setting.items()))

    # 6. Multiclass accuracy below 50%. The survey's own tables contradict this:
    #    every 4-class Twitter15/16 and PHEME number is far above 0.5.
    lr = long_results()
    multiclass = {d["name"] for d in DATASETS if len(d["labels"]) > 2}
    mc = lr[lr["dataset"].isin(multiclass) & (lr["metric"] == "ACC")]
    below = (mc["value"] < 0.5).sum()
    add("multiclass_below_50", below > len(mc) / 2,
        f"of {len(mc)} accuracies on multiclass datasets "
        f"({', '.join(sorted(multiclass & set(lr['dataset'])))}), "
        f"{below} are below 0.5; median is {mc['value'].median():.3f}")

    return pd.DataFrame(out)


def verify_table3(counts: dict[str, int]) -> pd.DataFrame:
    """Compare Table 3's printed sizes with what the downloads actually contain.

    ``counts`` maps dataset name to the number of items found on disk, as
    produced by ``disinfo.data.dataset_sizes``.
    """
    rows = []
    for d in DATASETS:
        got = counts.get(d["name"])
        rows.append({
            "dataset": d["name"], "table3_size": d["size"],
            "downloaded": got,
            "match": None if got is None else got == d["size"],
            "delta": None if got is None else got - d["size"],
        })
    return pd.DataFrame(rows)
