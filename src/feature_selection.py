"""
Feature-selection report: recomputes EVERYTHING from ai_trust_by_country.csv and writes a PDF.
The report exists in two languages produced by the same code (identical numbers):

  python feature_selection_report.py              -> Chinese  -> report_output/feature_selection_report.pdf
  python feature_selection_report.py --lang en    -> English  -> report_output_en/feature_selection_report.pdf

Reads : ai_trust_by_country.csv  (nothing else)
Writes: <output folder>/*.pdf and 01..11 *.csv  <- every table/number shown in the PDF, for checking
Set REPORT_PNG=1 to also export each PDF page as a PNG.

Sections of the code (search for the "# ==== STEP" banners)
  STEP 0  settings, variable roles, meaning-based themes
  STEP 1  pairwise Spearman correlations between the candidate features
  STEP 2  data-driven grouping  (hierarchical clustering on 1-|rho|, cut at 0.25)
  STEP 3  within-group vs between-group correlation (+ permutation test for the themes)
  STEP 4  selection inside each group (4 tests -> votes -> best member per group)
  STEP 5  redundancy check (VIF) and cross-validated performance
  STEP 6  build the PDF
"""
import os
import re
import sys
import unicodedata
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# STEP 0  settings, variable roles, themes
# =============================================================================
LANG = os.environ.get("REPORT_LANG", "zh").lower()
if "--lang" in sys.argv:
    LANG = sys.argv[sys.argv.index("--lang") + 1].lower()
EN = LANG == "en"


def tx(zh, en):
    """Pick the Chinese or English version of a user-visible string."""
    return en if EN else zh


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "datasets"
OUT = HERE / ("report_output_en" if EN else "report_output")
OUT.mkdir(exist_ok=True)
TARGET = "trust_ai_pct"
SEED = 42

# ---- knobs (change these to test how sensitive the result is) --------------
CUT_DISTANCE = 0.25        # dendrogram cut: distance = 1 - |rho|; 0.25 ~ groups whose members average |rho| >= 0.75
SPEARMAN_MIN = 0.5         # test 1: |rho| >= this ...
SPEARMAN_P = 0.01          #         ... and p < this
MI_QUANTILE = 0.5          # test 2: mutual information >= this quantile of all candidates (median)
LASSO_FREQ_MIN = 0.5       # test 3: chosen by LassoCV in >= 50% of bootstrap resamples
LASSO_BOOT = 200
RF_QUANTILE = 0.75         # test 4: RF permutation importance in the top quartile
VOTES_NEEDED = 3           # a variable needs >= 3 of the 4 tests to be selected
N_PERM = 5000              # permutations for the theme-vs-random test

# ---- fonts -----------------------------------------------------------------
_have = {f.name for f in fm.fontManager.ttflist}
if EN:
    FONT = "DejaVu Sans"
else:
    FONT = next((n for n in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC", "DengXian"] if n in _have), None)
    if FONT is None:
        print("WARNING: no CJK font found, Chinese text will show as boxes")
plt.rcParams["font.sans-serif"] = [FONT or "DejaVu Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ---- colours (colour-blind safe) -------------------------------------------
C_POS, C_NEG, C_KEEP, C_PASS, C_FAIL, C_INK = "#0072B2", "#D55E00", "#009E73", "#B8E0D2", "#EDEDED", "#222222"

df = pd.read_csv(DATA_DIR / "ai_trust_by_country.csv", encoding="utf-8")
df["log_gdp"] = np.log(df["gdp_per_capita_usd"])
df["log_pop"] = np.log(df["population"])
df["economy_advanced"] = (df["economy_type"] == "advanced").astype(int)
y = df[TARGET].values

# columns that are NOT candidate features, and why
EXCLUDED = {
    "country": tx("标识符", "identifier"),
    "iso3": tx("标识符", "identifier"),
    "accept_ai_pct": tx("与 trust 几乎是同一概念(接受度)，属于泄漏",
                        "almost the same construct as trust (acceptance); leakage"),
    "risks_outweigh_benefits_pct": tx("对 AI 的总体评价，是 trust 的结果而非原因",
                                      "overall verdict on AI, an outcome of trust rather than a cause"),
    "sentiment": tx("按整体态度打的国家标签", "country label based on the overall attitude"),
    "gdp_per_capita_usd": tx("改用 log_gdp(分布偏斜)", "replaced by log_gdp (skewed)"),
    "population": tx("改用 log_pop(分布极度偏斜)", "replaced by log_pop (extremely skewed)"),
    "region": tx("9 个类别对应 47 行，过于稀疏", "9 categories for 47 rows, too sparse"),
    "economy_type": tx("改用 0/1 变量 economy_advanced", "replaced by the 0/1 variable economy_advanced"),
}
candidates = [c for c in df.columns if c not in set(EXCLUDED) | {TARGET}
              and c not in ("log_gdp", "log_pop", "economy_advanced")] + ["log_gdp", "log_pop", "economy_advanced"]
short = lambda s: s.replace("_pct", "")
SEP = tx("、", ", ")

# meaning-based grouping, decided by hand from what each survey item asks (NOT from the data)
THEMES = {
    tx("T1 经济背景", "T1 Economic context"):
        ["log_gdp", "log_pop", "economy_advanced"],
    tx("T2a 现有保障与知晓度", "T2a Safeguards & awareness"):
        ["safeguards_sufficient_pct", "aware_regulations_pct"],
    tx("T2b 监管需求与信息环境担忧", "T2b Rules demand & info worry"):
        ["regulation_required_pct", "unsure_online_content_pct", "concerned_elections_pct", "want_misinfo_laws_pct"],
    tx("T3 感知收益", "T3 Perceived benefits"):
        ["expect_benefits_pct", "experienced_benefits_pct", "top_benefit_pct"],
    tx("T4a 对 AI 风险的担忧", "T4a Worry about AI risks"):
        ["concerned_negative_pct", "top_risk_pct"],
    tx("T4b 实际遭遇负面结果", "T4b Experienced harm"):
        ["experienced_negative_pct"],
    tx("T5 AI 素养与培训", "T5 AI literacy & training"):
        ["literacy_skills_pct", "ai_training_pct", "use_ai_effectively_pct"],
    tx("T6 个人使用与依赖", "T6 Personal use & dependence"):
        ["use_ai_work_regularly_pct", "couldnt_work_without_ai_pct", "relied_not_learning_pct", "concerned_left_behind_pct"],
    tx("T7 不当/风险行为", "T7 Improper / risky behaviour"):
        ["inappropriate_use_pct", "mistakes_due_ai_pct", "contravene_policies_pct", "relied_no_eval_pct",
         "presented_ai_as_own_pct"],
    tx("T8 组织采用", "T8 Organisational adoption"):
        ["org_uses_ai_pct", "org_ai_strategy_pct", "org_provides_training_pct", "org_ai_policies_pct"],
    tx("T9 职场影响", "T9 Workplace impact"):
        ["work_performable_by_ai_pct", "ai_raised_compliance_risk_pct", "increased_efficiency_pct",
         "increased_revenue_pct", "increased_workload_pct"],
}
THEME_OF = {f: t for t, fs in THEMES.items() for f in fs}
assert sorted(THEME_OF) == sorted(candidates), "every candidate must be in exactly one theme"
K = len(candidates)
print(f"language = {LANG} | n = {len(df)} countries, {K} candidate features")

# =============================================================================
# STEP 1  pairwise Spearman correlations
# =============================================================================
corr = df[candidates].corr(method="spearman")          # signed
A = corr.abs().values                                   # |rho| matrix used everywhere below
iu = np.triu_indices(K, 1)
pair_abs = A[iu]                                        # K*(K-1)/2 pair values
overall = {
    "n_pairs": len(pair_abs),
    "median_abs_rho": float(np.median(pair_abs)),
    "share_ge_0.5": float((pair_abs >= 0.5).mean()),
    "share_ge_0.75": float((pair_abs >= 0.75).mean()),
    "share_lt_0.3": float((pair_abs < 0.3).mean()),
}
corr.to_csv(OUT / "01_spearman_matrix_36x36.csv", encoding="utf-8-sig")
print("overall:", {k: round(v, 3) for k, v in overall.items()})

# =============================================================================
# STEP 2  data-driven grouping: average-linkage clustering on distance = 1 - |rho|
# =============================================================================
dist = 1 - corr.abs()
Z = linkage(squareform(dist.values, checks=False), method="average")
cl = pd.Series(fcluster(Z, t=CUT_DISTANCE, criterion="distance"), index=candidates, name="cluster")
dn = dendrogram(Z, no_plot=True, labels=candidates)
leaf_order = [candidates.index(l) for l in dn["ivl"]]   # order of variables along the dendrogram
csize = cl.value_counts()
multi_clusters = [k for k in cl.iloc[leaf_order].drop_duplicates() if csize[k] > 1]
n_single = int((csize == 1).sum())
print(f"clusters: {cl.nunique()} total, {len(multi_clusters)} with >1 member, {n_single} singletons")

# =============================================================================
# STEP 3  within-group vs between-group correlation
# =============================================================================


def group_summary(labels):
    """Mean |rho| inside each group, between the group and everything else, plus pooled numbers."""
    labs = np.asarray(labels)
    rows = []
    for g in pd.unique(labs):
        idx, oth = np.where(labs == g)[0], np.where(labs != g)[0]
        w = A[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean() if len(idx) > 1 else np.nan
        b = A[np.ix_(idx, oth)].mean()
        rows.append((g, len(idx), w, b))
    tab = pd.DataFrame(rows, columns=["group", "n", "within", "between"])
    same = labs[:, None] == labs[None, :]
    pooled_w = A[iu][same[iu]].mean()
    pooled_b = A[iu][~same[iu]].mean()
    return tab, pooled_w, pooled_b


def block_matrix(labels, order):
    labs = np.asarray(labels)
    M = np.full((len(order), len(order)), np.nan)
    for i, g in enumerate(order):
        for j, h in enumerate(order):
            gi, hj = np.where(labs == g)[0], np.where(labs == h)[0]
            if i == j:
                if len(gi) > 1:
                    M[i, j] = A[np.ix_(gi, gi)][np.triu_indices(len(gi), 1)].mean()
            else:
                M[i, j] = A[np.ix_(gi, hj)].mean()
    return M


cl_tab, cl_w, cl_b = group_summary(cl.values)
th_lab = np.array([THEME_OF[c] for c in candidates])
th_tab, th_w, th_b = group_summary(th_lab)

# permutation test (themes only): keep the theme sizes, shuffle which feature gets which theme label
rng = np.random.default_rng(SEED)
null = np.empty(N_PERM)
codes = pd.factorize(th_lab)[0]
for i in range(N_PERM):
    p = rng.permutation(codes)
    same = p[:, None] == p[None, :]
    null[i] = A[iu][same[iu]].mean()
perm_p = (np.sum(null >= th_w) + 1) / (N_PERM + 1)
print(f"clusters: within {cl_w:.3f} vs between {cl_b:.3f} | themes: within {th_w:.3f} vs between {th_b:.3f} | "
      f"random-label within {null.mean():.3f} (p={perm_p:.4f})")

cl_tab.assign(group=lambda d: "C" + d.group.astype(str)).to_csv(OUT / "02_cluster_within_between.csv", index=False, encoding="utf-8-sig")
th_tab.to_csv(OUT / "03_theme_within_between.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"grouping": ["cluster", "theme", "random labels (mean of %d perms)" % N_PERM],
              "pooled_within_abs_rho": [cl_w, th_w, null.mean()],
              "pooled_between_abs_rho": [cl_b, th_b, np.nan],
              "permutation_p": [np.nan, perm_p, np.nan]}).to_csv(OUT / "04_pooled_summary.csv", index=False, encoding="utf-8-sig")

# =============================================================================
# STEP 4  selection inside each group: 4 tests -> votes -> best member of the cluster
# =============================================================================
res = pd.DataFrame(index=candidates)
res["cluster"], res["theme"] = cl, pd.Series(THEME_OF)
res["spearman"], res["p_value"] = zip(*[spearmanr(df[c], y) for c in candidates])
res["abs_rho"] = res["spearman"].abs()
res["mutual_info"] = mutual_info_regression(df[candidates], y, random_state=SEED, n_neighbors=3)

Xz = pd.DataFrame(StandardScaler().fit_transform(df[candidates]), columns=candidates)
rng = np.random.default_rng(SEED)
hits = pd.Series(0.0, index=candidates)
for _ in range(LASSO_BOOT):                                           # test 3: Lasso bootstrap selection frequency
    idx = rng.choice(len(df), len(df), replace=True)
    m = LassoCV(cv=5, random_state=SEED, max_iter=20000).fit(Xz.iloc[idx], y[idx])
    hits += np.abs(m.coef_) > 1e-8
res["lasso_freq"] = hits / LASSO_BOOT

Xdf = df[candidates]                                                   # test 4: RF permutation importance, held-out folds
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=1)
perm = pd.Series(0.0, index=candidates)
n_fold = 0
for tr, te in RepeatedKFold(n_splits=5, n_repeats=4, random_state=SEED).split(Xdf):
    rf.fit(Xdf.iloc[tr], y[tr])
    perm += permutation_importance(rf, Xdf.iloc[te], y[te], n_repeats=10, random_state=SEED, n_jobs=-1).importances_mean
    n_fold += 1
res["rf_perm_imp"] = perm / n_fold

res["t1_spearman"] = (res["abs_rho"] >= SPEARMAN_MIN) & (res["p_value"] < SPEARMAN_P)
res["t2_mi"] = res["mutual_info"] >= res["mutual_info"].quantile(MI_QUANTILE)
res["t3_lasso"] = res["lasso_freq"] >= LASSO_FREQ_MIN
res["t4_rf"] = res["rf_perm_imp"] >= res["rf_perm_imp"].quantile(RF_QUANTILE)
res["votes"] = res[["t1_spearman", "t2_mi", "t3_lasso", "t4_rf"]].sum(axis=1)
res["cluster_best"] = res.groupby("cluster")["abs_rho"].transform("max") == res["abs_rho"]   # highest |rho| in its cluster
res["selected"] = (res["votes"] >= VOTES_NEEDED) & res["cluster_best"]


def verdict(r):
    if r["selected"]:
        return "KEEP"
    if r["votes"] >= VOTES_NEEDED:
        return "redundant"      # passes the tests but a cluster-mate has a higher |rho|
    return "borderline" if r["votes"] == 2 else "drop"


res["verdict"] = res.apply(verdict, axis=1)
CONTROL = "log_gdp"             # economic context: added by hand as a control variable
selected = res.index[res["selected"]].tolist()
final_set = selected + ([CONTROL] if CONTROL not in selected else [])
print("selected by rule:", selected, "| final set:", final_set)
res.sort_values(["selected", "votes", "abs_rho"], ascending=False).to_csv(OUT / "05_feature_table.csv", encoding="utf-8-sig")

# same 4 tests applied to the meaning-based themes: best member per theme (same rule, votes then |rho|)
tmp = res.sort_values(["votes", "abs_rho"], ascending=False)
bt = tmp.groupby("theme").head(1).reset_index().set_index("theme")          # 'index' = variable name
th_pick = pd.DataFrame({"n_members": [len(v) for v in THEMES.values()]}, index=list(THEMES))
th_pick["best_member"] = [bt.loc[t, "index"] for t in THEMES]
th_pick["rho_with_trust"] = [bt.loc[t, "spearman"] for t in THEMES]
th_pick["votes"] = [int(bt.loc[t, "votes"]) for t in THEMES]
th_pick["within_abs_rho"] = th_tab.set_index("group").loc[list(THEMES), "within"].values
th_pick["passes_rule"] = th_pick["votes"] >= VOTES_NEEDED
th_pick.to_csv(OUT / "06_theme_best_members.csv", encoding="utf-8-sig")

# =============================================================================
# STEP 5  redundancy (VIF) and cross-validated performance
# =============================================================================


def vif(cols):
    Xc = StandardScaler().fit_transform(df[cols])
    return pd.Series([variance_inflation_factor(Xc, i) for i in range(len(cols))], index=cols)


vif_all, vif_final = vif(candidates), vif(final_set)
pd.concat([vif_all.rename("VIF_all_candidates"), vif_final.rename("VIF_final_set")], axis=1).to_csv(
    OUT / "07_vif.csv", encoding="utf-8-sig")

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
lasso = lambda: make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=SEED, max_iter=20000))
forest = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=1)
kbest = lambda k: make_pipeline(StandardScaler(), SelectKBest(f_regression, k=k), RidgeCV(alphas=np.logspace(-2, 3, 30)))
NF = len(final_set)
NOTE_OPT = tx("偏乐观(用全部数据选特征)", "optimistic (selected on all data)")
experiments = [
    (tx("基线：预测均值", "Baseline: predict the mean"), DummyRegressor(), final_set, "-"),
    (tx(f"全部 {K} 个特征 - Ridge", f"All {K} features - Ridge"), ridge(), candidates, tx("无偏", "unbiased")),
    (tx(f"全部 {K} 个特征 - Lasso", f"All {K} features - Lasso"), lasso(), candidates, tx("无偏", "unbiased")),
    (tx(f"折内选 {NF} 个(SelectKBest) - Ridge", f"In-fold selection of {NF} + Ridge"), kbest(NF), candidates,
     tx("无偏(选择在每折内完成)", "unbiased (in-fold selection)")),
    (tx(f"最终 {NF} 个特征 - Ridge", f"Final {NF} features - Ridge"), ridge(), final_set, NOTE_OPT),
    (tx(f"最终 {NF} 个特征 - 随机森林", f"Final {NF} features - Random forest"), forest(), final_set, NOTE_OPT),
]
perf_rows = []
for name, model, cols, note in experiments:
    r = cross_validate(model, df[cols], y, cv=cv, n_jobs=-1, scoring=["r2", "neg_root_mean_squared_error"])
    perf_rows.append((name, 0 if isinstance(model, DummyRegressor) else len(cols), r["test_r2"].mean(),
                      r["test_r2"].std(), -r["test_neg_root_mean_squared_error"].mean(), note))
perf = pd.DataFrame(perf_rows, columns=["scheme", "n_features", "cv_r2_mean", "cv_r2_sd", "cv_rmse", "note"])
perf.to_csv(OUT / "08_cv_performance.csv", index=False, encoding="utf-8-sig")
print(perf.round(3).to_string(index=False))

# block matrices (mean |rho| between groups; diagonal = within-group mean)
cl_order = [k for k in cl.iloc[leaf_order].drop_duplicates()]
M_cl = block_matrix(cl.values, cl_order)
th_order = list(THEMES)
M_th = block_matrix(th_lab, th_order)
pd.DataFrame(M_cl, index=[f"C{k}" for k in cl_order], columns=[f"C{k}" for k in cl_order]).to_csv(
    OUT / "09_block_matrix_clusters.csv", encoding="utf-8-sig")
pd.DataFrame(M_th, index=th_order, columns=th_order).to_csv(OUT / "10_block_matrix_themes.csv", encoding="utf-8-sig")
pd.DataFrame({"variable": candidates, "cluster": [f"C{k}" for k in cl], "theme": [THEME_OF[c] for c in candidates]}).to_csv(
    OUT / "11_group_assignments.csv", index=False, encoding="utf-8-sig")

# =============================================================================
# STEP 6  PDF
# =============================================================================
PORTRAIT, LAND = (8.27, 11.69), (11.69, 8.27)
pdf = PdfPages(OUT / "feature_selection_report.pdf")

# optional: also export every page as PNG (set REPORT_PNG=1) for quick visual checking
_orig_savefig, _page = pdf.savefig, [0]


def _savefig(fig=None, **kw):
    _page[0] += 1
    if os.environ.get("REPORT_PNG"):
        (OUT / "pages").mkdir(exist_ok=True)
        fig.savefig(OUT / "pages" / f"page_{_page[0]:02d}.png", dpi=80)
    _orig_savefig(fig, **kw)


pdf.savefig = _savefig
TOK = re.compile(r"[　-鿿＀-￯]|[^\s　-鿿＀-￯]+|\s+")


def wrap(text, width):
    """Wrap mixed Chinese/English text to `width` half-width columns without splitting English words."""
    lines = []
    for para in text.split("\n"):
        cur, w = "", 0
        for tok in TOK.findall(para):
            tw = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in tok)
            if tok.isspace():
                if cur:
                    cur, w = cur + " ", w + 1
                continue
            if w + tw > width and cur:
                lines.append(cur.rstrip())
                cur, w = "", 0
            cur, w = cur + tok, w + tw
        lines.append(cur.rstrip())
    return lines


class TextPages:
    """Simple flowing text: (style, text) blocks, styles h1 h2 p b(ullet) code."""
    SIZE = {"h1": 17, "h2": 12.5, "p": 9.6, "b": 9.6, "code": 8.6}

    def __init__(self):
        self.fig = None

    def _new(self):
        self.fig = plt.figure(figsize=PORTRAIT)
        self.y = 0.955

    def _flush(self):
        if self.fig is not None:
            pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None

    def add(self, blocks):
        for style, text in blocks:
            size = self.SIZE[style]
            width = int((90 if EN else 96) * 9.6 / size) - (3 if style == "b" else 0)
            lines = wrap(text, width)
            lh = size * 1.62 / 72 / PORTRAIT[1]
            need = lh * len(lines) + (0.012 if style in ("h1", "h2") else 0.006)
            if style in ("h1", "h2"):
                need += 0.09          # keep a heading together with the text that follows it
            if self.fig is None or self.y - need < 0.04:
                self._flush()
                self._new()
            if style in ("h1", "h2"):
                self.y -= 0.008
            for i, ln in enumerate(lines):
                x = 0.07 + (0.02 if style == "b" else 0)
                if style == "b" and i == 0:
                    self.fig.text(0.07, self.y, "•", fontsize=size, va="top", color=C_INK)
                self.fig.text(x, self.y, ln, fontsize=size, va="top", color=C_INK,
                              fontweight="bold" if style in ("h1", "h2") else "normal",
                              family="monospace" if style == "code" else None)  # code blocks: ASCII only
                self.y -= lh
            self.y -= 0.006

    def close(self):
        self._flush()


def caption(fig, text, y_top=0.145, size=8.6, width=158):
    lines = wrap(text, width)
    lh = size * 1.6 / 72 / fig.get_size_inches()[1]
    for i, ln in enumerate(lines):
        fig.text(0.05, y_top - i * lh, ln, fontsize=size, va="top", color=C_INK)


def fig_title(fig, text, sub=None):
    fig.text(0.05, 0.975, text, fontsize=14, fontweight="bold", va="top", color=C_INK)
    if sub:
        fig.text(0.05, 0.93, sub, fontsize=9, va="top", color="#555555")


pct = lambda v: f"{100 * v:.0f}%"
n_multi = len(multi_clusters)
tp = TextPages()

# ---------- page: summary ---------------------------------------------------
sel_txt = SEP.join(short(s) for s in selected)
tp.add([
    ("h1", tx("影响 AI 信任 (trust_ai_pct) 的因素：变量分组与特征选择",
              "Factors behind trust in AI (trust_ai_pct): variable grouping and feature selection")),
    ("p", tx(f"数据：ai_trust_by_country.csv，{len(df)} 个国家，{K} 个候选特征(已排除泄漏变量，见方法页)。"
             f"本报告中的每个数字和图表，都可以用同目录下的 feature_selection_report.py 从原始 CSV 重新算出；"
             f"所有中间表格另存在 {OUT.name}/*.csv。",
             f"Data: ai_trust_by_country.csv, {len(df)} countries, {K} candidate features (leakage variables already "
             f"removed, see the Method page). Every number and chart in this report can be recomputed from the raw CSV "
             f"with feature_selection_report.py in the same folder; the intermediate tables are saved in {OUT.name}/*.csv.")),
    ("h2", tx("核心发现", "Key findings")),
    ("b", tx(f"1. 总体上变量之间关系紧密。{K} 个候选特征两两之间共 {overall['n_pairs']} 对，"
             f"|Spearman ρ| 的中位数为 {overall['median_abs_rho']:.2f}；{pct(overall['share_ge_0.5'])} 的变量对 |ρ|≥0.5，"
             f"{pct(overall['share_ge_0.75'])} 的变量对 |ρ|≥0.75。信息大量重复，不能把它们全部放进模型(图 1)。",
             f"1. Overall, the variables are tightly related. The {K} candidate features form {overall['n_pairs']} pairs; "
             f"the median |Spearman ρ| is {overall['median_abs_rho']:.2f}; {pct(overall['share_ge_0.5'])} of the pairs have "
             f"|ρ|≥0.5 and {pct(overall['share_ge_0.75'])} have |ρ|≥0.75. Much of the information is duplicated, so the "
             f"variables cannot all go into one model (Fig. 1).")),
    ("b", tx(f"2. 分组之后，组内变量高度相关。按相关性聚类(数据驱动)得到 {cl.nunique()} 组，其中 {n_multi} 组有多个成员，"
             f"这些组内变量的平均 |ρ| = {cl_w:.2f}(图 2、图 3)。按题目含义手工分的 {len(THEMES)} 个主题，"
             f"组内平均 |ρ| = {th_w:.2f}，随机打乱标签只有 {null.mean():.2f}(置换检验 p = {perm_p:.4f})。",
             f"2. After grouping, variables inside a group are highly correlated. Correlation clustering (data-driven) gives "
             f"{cl.nunique()} groups, {n_multi} of them with more than one member; the mean |ρ| inside these groups is "
             f"{cl_w:.2f} (Fig. 2, 3). For the {len(THEMES)} meaning-based themes I defined by hand, the within-theme mean "
             f"|ρ| is {th_w:.2f}, versus only {null.mean():.2f} when the theme labels are shuffled (permutation test "
             f"p = {perm_p:.4f}).")),
    ("b", tx(f"3. 分组之后，组间差距变大。数据驱动分组的组间平均 |ρ| 只有 {cl_b:.2f}(组内 {cl_w:.2f})；"
             f"主题分组的组间为 {th_b:.2f}(组内 {th_w:.2f})(图 3、图 4)。因此每组只需保留少数代表变量。",
             f"3. After grouping, the groups are more distinct from each other. The between-group mean |ρ| is only "
             f"{cl_b:.2f} for the data-driven groups (within: {cl_w:.2f}) and {th_b:.2f} for the themes (within: "
             f"{th_w:.2f}) (Fig. 3, 4). So only a few representatives per group are needed.")),
    ("b", tx(f"4. 每组内部的选择：用 4 项检验投票，得票≥{VOTES_NEEDED} 且在所属聚类中与 trust 的 |ρ| 最大者入选(图 5)。"
             f"规则选出 {len(selected)} 个：{sel_txt}；另手工加入 log_gdp 作为经济背景的控制变量，最终 {len(final_set)} 个特征。",
             f"4. Selection inside each group: 4 tests vote; a variable needs ≥{VOTES_NEEDED} votes and must have the largest "
             f"|ρ| with trust in its cluster (Fig. 5). The rule selects {len(selected)}: {sel_txt}. I add log_gdp by hand as "
             f"a control for economic context, giving {len(final_set)} final features.")),
    ("b", tx("5. 预测表现：4 个特征的交叉验证 R² 与 36 个特征的全量模型相当甚至更好，且共线性大幅降低(图 7)。"
             "注意：特征是用全部 47 行数据选出的，所以该 R² 偏乐观，见最后一页的局限说明。",
             "5. Predictive performance: the cross-validated R² with the 4 features is comparable to, or better than, a model "
             "using all 36 features, and collinearity is far lower (Fig. 7). Caveat: the features were selected using all "
             "47 rows, so this R² is optimistic; see the limitations on the last page.")),
    ("h2", tx("本报告页面", "Pages in this report")),
    ("p", tx("图 1 总体相关性 | 图 2 聚类树与重排热图 | 图 3 组内 vs 组间 | 图 4 组间相关矩阵 | "
             "图 5 每组变量对 trust 的影响与选择结果 | 表 1 主题分组的选择 | 图 6 最终特征与 trust 的关系 | "
             "图 7 共线性与预测表现 | 局限说明",
             "Fig. 1 overall correlation | Fig. 2 dendrogram and reordered heatmap | Fig. 3 within vs between groups | "
             "Fig. 4 between-group correlation matrix | Fig. 5 effect of each variable on trust and selection result | "
             "Table 1 selection in the meaning-based themes | Fig. 6 final features vs trust | Fig. 7 collinearity and "
             "performance | Limitations")),
])
tp.close()

# ---------- page: method (answers "what criterion / what method") ----------------
tp = TextPages()
tp.add([
    ("h1", tx("方法：如何分组，如何在组内选变量", "Method: how variables are grouped and selected within groups")),
    ("h2", tx("0. 先剔除的列", "0. Columns removed first")),
    ("p", tx("identifier 与泄漏类：" + "；".join(f"{k}({v})" for k, v in EXCLUDED.items())
             + "。剩余 36 个候选特征 = 原有 33 个调查指标 + log_gdp、log_pop、economy_advanced。",
             "Identifiers and leakage: " + "; ".join(f"{k} ({v})" for k, v in EXCLUDED.items())
             + ". The remaining 36 candidates = 33 original survey indicators + log_gdp, log_pop, economy_advanced.")),
    ("h2", tx("1. 分组标准(两套，互相对照)", "1. Grouping criteria (two schemes, compared with each other)")),
    ("b", tx(f"数据驱动分组(主分组)：只看数值，不看含义。相似度 = 两变量 Spearman 秩相关的绝对值 |ρ|，距离 = 1 − |ρ|；"
             f"用平均连接(average linkage)层次聚类，在距离 {CUT_DISTANCE} 处切树。因为平均连接看的是簇内平均"
             f"距离，所以“组内平均 |ρ| 约≥{1 - CUT_DISTANCE:.2f}”，并不要求每一对都超过该值。{CUT_DISTANCE} 是我设定的经验阈值，"
             f"可在脚本里改 CUT_DISTANCE。",
             f"Data-driven grouping (main): looks only at the numbers, not at meaning. Similarity = the absolute Spearman "
             f"rank correlation |ρ| between two variables; distance = 1 − |ρ|. Average-linkage hierarchical clustering, cut "
             f"at distance {CUT_DISTANCE}. Because average linkage looks at the mean distance inside a cluster, this means "
             f"'the mean |ρ| inside a group is about ≥{1 - CUT_DISTANCE:.2f}'; it does not require every single pair to "
             f"exceed that value. {CUT_DISTANCE} is an empirical threshold I chose; change CUT_DISTANCE in the script.")),
    ("b", tx("含义分组(对照)：我根据每个调查题目的含义手工分为 11 个主题(如“AI 素养与培训”“组织采用”)，不使用任何数据。"
             "用它来检验：数据驱动的分组，是否与“意思相近的题目”一致；以及组内相关性高，不是聚类算法自己造出来的。",
             "Meaning-based grouping (control): I split the variables by hand into 11 themes (e.g. 'AI literacy & training', "
             "'Organisational adoption') from the wording of each survey item, using no data at all. It checks that the "
             "data-driven groups agree with 'items that mean similar things', and that the high within-group correlation "
             "is not just an artefact of the clustering algorithm.")),
    ("h2", tx("2. 组内选变量的方法：四项检验投票 + 每个聚类只留一个",
              "2. Selecting within groups: four voting tests + one variable per cluster")),
    ("b", tx(f"检验 1 单变量相关：|Spearman ρ| ≥ {SPEARMAN_MIN} 且 p < {SPEARMAN_P}。",
             f"Test 1, univariate correlation: |Spearman ρ| ≥ {SPEARMAN_MIN} and p < {SPEARMAN_P}.")),
    ("b", tx(f"检验 2 互信息(3 近邻估计)：不低于 {K} 个候选特征的中位数，能捕捉非线性关系。",
             f"Test 2, mutual information (3-nearest-neighbour estimate): at least the median of the {K} candidates; "
             f"captures non-linear relationships.")),
    ("b", tx(f"检验 3 Lasso 稳定性：对样本做 {LASSO_BOOT} 次自助重采样，每次用 5 折交叉验证的 LassoCV 拟合，"
             f"系数非零的频率 ≥ {LASSO_FREQ_MIN:.0%}。",
             f"Test 3, Lasso stability: bootstrap the sample {LASSO_BOOT} times and fit a 5-fold-CV LassoCV each time; the "
             f"coefficient must be non-zero in ≥{LASSO_FREQ_MIN:.0%} of the resamples.")),
    ("b", tx(f"检验 4 随机森林置换重要性：5 折×4 次重复，在留出折上打乱该变量，看预测误差增加多少；取平均后排在前 {1 - RF_QUANTILE:.0%}。",
             f"Test 4, random-forest permutation importance: 5 folds × 4 repeats; shuffle the variable on the held-out fold "
             f"and measure the increase in prediction error; the average must be in the top {1 - RF_QUANTILE:.0%}.")),
    ("b", tx(f"选择规则：得票≥{VOTES_NEEDED}(共 4 票)，并且在所属聚类中与 trust 的 |ρ| 最大。这样每个聚类最多入选一个，"
             "避免把信息重复的变量同时放进模型。未通过的聚类不入选。",
             f"Selection rule: ≥{VOTES_NEEDED} of the 4 votes AND the largest |ρ| with trust within its cluster. Each cluster "
             f"therefore contributes at most one variable, so duplicated information is not fed to the model twice. "
             f"Clusters in which no member passes contribute nothing.")),
    ("b", tx(f"人工决定：log_gdp 未通过投票(票数 {int(res.loc['log_gdp', 'votes'])})，但因为经济发展水平是关键背景，作为控制变量保留；"
             "economy_advanced 与它高度共线，所以只留连续的 log_gdp。",
             f"Manual decision: log_gdp fails the vote ({int(res.loc['log_gdp', 'votes'])} votes) but is kept as a control "
             f"because economic development is key background; economy_advanced is highly collinear with it, so only the "
             f"continuous log_gdp is used.")),
    ("h2", tx("3. 需要注意的点", "3. Points to keep in mind")),
    ("p", tx("· 聚类是按相关性分的，所以“组内相关性高”在数据驱动分组里部分是构造出来的；真正独立的证据是含义分组的置换检验(图 3 右)。\n"
             "· 选择规则与阈值(0.5、中位数、50%、前 25%、3 票)都是经验值，改动会改变名单；"
             "safeguards_sufficient 与 couldnt_work_without_ai 在各种设置下都很稳，increased_efficiency 相对不稳。\n"
             "· 国家层面 47 个样本，结论是预测性的，不代表因果，也不能推到个人层面。",
             "· Clusters are formed from correlations, so 'high correlation within a group' is partly built in for the "
             "data-driven grouping; the truly independent evidence is the permutation test on the meaning-based themes "
             "(Fig. 3, right).\n"
             "· The rules and thresholds (0.5, median, 50%, top 25%, 3 votes) are empirical; changing them changes the list. "
             "safeguards_sufficient and couldnt_work_without_ai are robust across settings, increased_efficiency is less so.\n"
             "· With 47 countries the conclusions are predictive, not causal, and cannot be transferred to individuals.")),
])
tp.close()

# ---------- Fig 1: overall correlation --------------------------------------
fig = plt.figure(figsize=LAND)
fig_title(fig, tx("图 1  总体上，候选变量之间关系紧密", "Fig. 1  Overall, the candidate variables are tightly related"),
          tx(f"{K} 个候选特征两两 Spearman 相关(原始列顺序)",
             f"Pairwise Spearman correlation of the {K} candidate features (original column order)"))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], left=0.16, right=0.96, top=0.88, bottom=0.30, wspace=0.28)
ax = fig.add_subplot(gs[0])
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(K), [short(c) for c in candidates], rotation=90, fontsize=5.5)
ax.set_yticks(range(K), [short(c) for c in candidates], fontsize=5.5)
cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("Spearman ρ", fontsize=8)
ax = fig.add_subplot(gs[1])
ax.hist(pair_abs, bins=np.linspace(0, 1, 21), color=C_POS, edgecolor="white")
for v, lab in ((0.5, "0.5"), (0.75, "0.75")):
    ax.axvline(v, color=C_NEG, ls="--", lw=1)
    ax.text(v + 0.01, ax.get_ylim()[1] * 0.93, lab, color=C_NEG, fontsize=8)
ax.set(xlabel=tx("变量对的 |Spearman ρ|", "|Spearman ρ| of a variable pair"),
       ylabel=tx(f"变量对数量(共 {overall['n_pairs']} 对)", f"Number of pairs (total {overall['n_pairs']})"))
ax.set_title(tx(f"中位数 {overall['median_abs_rho']:.2f}", f"Median {overall['median_abs_rho']:.2f}"), fontsize=10)
caption(fig, tx(f"左：{K}×{K} 的相关矩阵，红色为正相关、蓝色为负相关，大片深红说明多数变量朝同一方向变化。"
                f"右：把 {overall['n_pairs']} 个变量对的 |ρ| 画成直方图：{pct(overall['share_ge_0.5'])} 的变量对 |ρ|≥0.5，"
                f"{pct(overall['share_ge_0.75'])} 的 |ρ|≥0.75，只有 {pct(overall['share_lt_0.3'])} 低于 0.3。"
                "这意味着许多变量测的是同一件事(各国 AI 采用成熟度)，直接全部放进模型会出现严重共线性，需要先分组去冗余。"
                "少数变量(如 concerned_elections、want_misinfo_laws、log_pop)与其他变量相关性弱，位于图中的浅色带。",
                f"Left: the {K}×{K} correlation matrix; red = positive, blue = negative; the large dark-red areas show that most "
                f"variables move together. Right: histogram of |ρ| over the {overall['n_pairs']} variable pairs: "
                f"{pct(overall['share_ge_0.5'])} of the pairs have |ρ|≥0.5, {pct(overall['share_ge_0.75'])} have |ρ|≥0.75, and "
                f"only {pct(overall['share_lt_0.3'])} are below 0.3. Many variables therefore measure the same thing (each "
                f"country's AI-adoption maturity); putting them all into one model causes severe collinearity, so groups must "
                f"be formed and the redundancy removed first. A few variables (e.g. concerned_elections, want_misinfo_laws, "
                f"log_pop) are weakly related to the rest and show up as the pale bands."))
pdf.savefig(fig)
plt.close(fig)

# ---------- Fig 2: dendrogram + ordered heatmap -----------------------------
fig = plt.figure(figsize=LAND)
fig_title(fig, tx("图 2  按相关性聚类后，出现清晰的“块”", "Fig. 2  After clustering by correlation, clear 'blocks' appear"),
          tx(f"距离 = 1 − |ρ|，平均连接，在 {CUT_DISTANCE} 处切分 → {cl.nunique()} 组",
             f"distance = 1 − |ρ|, average linkage, cut at {CUT_DISTANCE} → {cl.nunique()} groups"))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.12, right=0.97, top=0.88, bottom=0.26, wspace=0.05)
ax = fig.add_subplot(gs[0])
dendrogram(Z, labels=[short(c) for c in candidates], orientation="right", color_threshold=CUT_DISTANCE,
           ax=ax, leaf_font_size=6.5, above_threshold_color="#888888")
ax.axvline(CUT_DISTANCE, color=C_NEG, ls="--", lw=1)
ax.text(CUT_DISTANCE + 0.01, ax.get_ylim()[1] * 1.005, tx(f"切分线 {CUT_DISTANCE}", f"cut {CUT_DISTANCE}"),
        color=C_NEG, fontsize=8, va="bottom")
ax.set_xlabel(tx("距离 = 1 − |ρ|", "distance = 1 − |ρ|"))
ax = fig.add_subplot(gs[1])
ordc = [candidates[i] for i in leaf_order]
ax.imshow(corr.loc[ordc, ordc].values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks([]); ax.set_yticks([])
starts = {}
for i, name in enumerate(ordc):
    starts.setdefault(cl[name], []).append(i)
for k, idxs in starts.items():
    if len(idxs) > 1:
        lo, hi = min(idxs), max(idxs)
        ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1, fill=False, ec="black", lw=1.4))
        ax.text(hi + 0.9, (lo + hi) / 2, f"C{k}", fontsize=8, va="center", fontweight="bold")
ax.set_title(tx("按聚类顺序重排的相关矩阵(黑框 = 多成员组)",
                "Correlation matrix reordered by cluster (black boxes = multi-member groups)"), fontsize=9)
members_txt = "  ".join(f"C{k}[{len(cl[cl == k])}]" for k in multi_clusters)
caption(fig, tx(f"左：树状图，两个变量在越靠左的位置合并，说明越相似；橙色虚线为切分线，线以左连在一起的变量归为一组。"
                f"右：用同一顺序重排相关矩阵后，相似变量聚在对角线上形成深红色的块，黑框标出 {n_multi} 个多成员组"
                f"(方括号内为成员数：{members_txt})。其余 {n_single} 个变量与别的变量都不够相似，各自单独成组。"
                f"组的编号 C* 沿用脚本里 fcluster 的输出，成员名单见 {OUT.name}/11_group_assignments.csv。",
                f"Left: dendrogram; two variables that merge further to the left are more similar; the orange dashed line is "
                f"the cut, and variables joined to the left of it form one group. Right: reordering the correlation matrix the "
                f"same way puts similar variables into dark-red blocks along the diagonal; the black boxes mark the {n_multi} "
                f"multi-member groups (member counts in brackets: {members_txt}). The other {n_single} variables are not "
                f"similar enough to any other and form groups of their own. Group numbers C* follow the fcluster output of "
                f"the script; the member lists are in {OUT.name}/11_group_assignments.csv."))
pdf.savefig(fig)
plt.close(fig)

# ---------- Fig 3: within vs between ----------------------------------------
fig = plt.figure(figsize=LAND)
fig_title(fig, tx("图 3  分组之后：组内相关性高，组间相关性低",
                  "Fig. 3  After grouping: high correlation inside groups, low between groups"),
          tx("每个多成员组的组内平均 |ρ|(蓝) vs 该组与其余所有变量的平均 |ρ|(橙)",
             "Mean |ρ| inside each multi-member group (blue) vs between the group and all other variables (orange)"))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 0.9], left=0.07, right=0.97, top=0.86, bottom=0.30, wspace=0.62 if EN else 0.42)


def wb_panel(ax, tab, labels, title):
    t = tab[tab["n"] > 1].copy()
    t["lab"] = [labels(g) for g in t["group"]]
    t = t.sort_values("within")
    yy = np.arange(len(t))
    ax.barh(yy + 0.2, t["within"], 0.38, color=C_POS, label=tx("组内", "within group"))
    ax.barh(yy - 0.2, t["between"], 0.38, color=C_NEG, label=tx("组间(与其余变量)", "between (vs. all others)"))
    ax.set_yticks(yy, t["lab"], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel(tx("平均 |Spearman ρ|", "Mean |Spearman ρ|"))
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=False)


wb_panel(fig.add_subplot(gs[0]), cl_tab, lambda g: f"C{g} ({int(csize[g])})",
         tx(f"数据驱动分组(组内 {cl_w:.2f} / 组间 {cl_b:.2f})", f"Data-driven groups (within {cl_w:.2f} / between {cl_b:.2f})"))
wb_panel(fig.add_subplot(gs[1]), th_tab, lambda g: g,
         tx(f"含义分组(组内 {th_w:.2f} / 组间 {th_b:.2f})", f"Meaning-based themes (within {th_w:.2f} / between {th_b:.2f})"))
ax = fig.add_subplot(gs[2])
ax.hist(null, bins=40, color="#BBBBBB")
ax.axvline(th_w, color=C_POS, lw=2)
ax.set_xlim(null.min() - 0.01, th_w + 0.05)
ax.text(th_w - 0.005, ax.get_ylim()[1] * 0.92, tx(f"实际 {th_w:.2f}", f"actual {th_w:.2f}"), color=C_POS, fontsize=8, ha="right")
ax.set(xlabel=tx("随机打乱主题标签后的组内平均 |ρ|", "Within-theme mean |ρ| (labels shuffled)"),
       ylabel=tx("次数", "Count"))
ax.set_title(tx(f"置换检验：p = {perm_p:.4f}", f"Permutation test: p = {perm_p:.4f}"), fontsize=10)
weak = th_tab[(th_tab["n"] > 1) & (th_tab["within"] < 0.4)]["group"].tolist()
_mc = cl_tab[cl_tab["n"] > 1]
_all = (_mc["within"] > _mc["between"]).all()
caption(fig, tx(f"左：数据驱动分组中，{'全部' if _all else '多数'} {len(_mc)} 个多成员组的组内相关(蓝)高于组间(橙)。这部分有“构造”成分——聚类本来就是把相关性高的放在一起。"
                f"中：更有说服力的是按含义分的主题，它没有用到数据，但组内平均 |ρ| 仍是 {th_w:.2f}，高于组间的 {th_b:.2f}。"
                f"右：把 {K} 个变量随机分到相同大小的组里重复 {N_PERM} 次，组内均值只有 {null.mean():.2f} 左右，"
                f"实际值 {th_w:.2f} 远高于随机水平(p = {perm_p:.4f})，说明“意思相近的题目，数据上也相近”。"
                f"例外：{'、'.join(weak) if weak else '无'} 的组内相关很弱，说明这些题目虽然主题相近，但测的其实是不同的东西。",
                f"Left: in the data-driven grouping, {'all' if _all else 'most of the'} {len(_mc)} multi-member groups have a "
                f"within-group correlation (blue) higher than the between-group one (orange). This is partly built in: "
                f"clustering puts highly correlated variables together. Middle: stronger evidence comes from the meaning-based "
                f"themes, which did not use the data; their within-theme mean |ρ| is still {th_w:.2f}, above the "
                f"between-theme {th_b:.2f}. Right: randomly assigning the {K} variables to groups of the same sizes {N_PERM} "
                f"times gives a within-group mean of only about {null.mean():.2f}; the actual {th_w:.2f} is far above chance "
                f"(p = {perm_p:.4f}), so 'items that mean similar things are also similar in the data'. Exceptions: "
                f"{', '.join(weak) if weak else 'none'} have a very weak within-group correlation, i.e. similar topics but "
                f"they actually measure different things."))
pdf.savefig(fig)
plt.close(fig)

# ---------- Fig 4: block matrices -------------------------------------------
fig = plt.figure(figsize=LAND)
fig_title(fig, tx("图 4  组间差距：对角线(组内)明显亮于非对角线(组间)",
                  "Fig. 4  Between-group gap: within (diagonal) is brighter than between"),
          tx("单元格 = 两组成员之间的平均 |ρ|；对角线 = 组内平均 |ρ|(单成员组为空)",
             "Cell = mean |ρ| between the members of two groups; diagonal = within-group mean |ρ| (empty for single-member groups)"))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], left=0.09, right=0.95, top=0.88, bottom=0.24, wspace=0.35)
for i, (M, order, lab, ttl) in enumerate((
        (M_cl, cl_order, [f"C{k}" + ("" if csize[k] > 1 else " ·") for k in cl_order],
         tx(f"数据驱动分组({len(cl_order)} 组，· = 单成员)", f"Data-driven groups ({len(cl_order)} groups, · = single member)")),
        (M_th, th_order, [t.split(" ", 1)[0] for t in th_order],
         tx(f"含义分组({len(th_order)} 个主题)", f"Meaning-based themes ({len(th_order)})")))):
    ax = fig.add_subplot(gs[i])
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#EEEEEE")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), lab, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), lab, fontsize=7)
    for a_ in range(len(order)):
        for b_ in range(len(order)):
            if not np.isnan(M[a_, b_]):
                ax.text(b_, a_, f"{M[a_, b_]:.1f}"[1:] if M[a_, b_] < 1 else "1", ha="center", va="center", fontsize=5.2,
                        color="white" if M[a_, b_] < 0.55 else "black")
    ax.set_title(ttl, fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label(tx("平均 |ρ|", "mean |ρ|"), fontsize=8)
offd_cl = M_cl[~np.eye(len(cl_order), dtype=bool)]
offd_th = M_th[~np.eye(len(th_order), dtype=bool)]
caption(fig, tx(f"每个格子是“行组”与“列组”成员之间所有变量对的平均 |ρ|，数字省略了小数点前的 0(.8 表示 0.8)。"
                f"数据驱动分组：对角线(组内)各组均值 {np.nanmean(np.diag(M_cl)):.2f}，非对角线(组间)均值 {np.nanmean(offd_cl):.2f}。"
                f"含义分组：对角线 {np.nanmean(np.diag(M_th)):.2f}，非对角线 {np.nanmean(offd_th):.2f}"
                "(这里是各组简单平均，与图 3 的“全部变量对合并平均”略有差异)。"
                "注意：组间并非无关——例如“AI 素养”“组织采用”“职场影响”之间的组间相关仍然较高，这些变量共同反映各国 AI 采用成熟度；"
                "而“监管需求”“风险担忧”等主题与其余主题的相关较低。所以每个大类里保留一个代表，而不是每个聚类之间都彼此独立。",
                f"Each cell is the mean |ρ| over all variable pairs between the row group and the column group; the leading 0 "
                f"is dropped (.8 means 0.8). Data-driven groups: mean of the diagonal (within) {np.nanmean(np.diag(M_cl)):.2f}, "
                f"mean of the off-diagonal (between) {np.nanmean(offd_cl):.2f}. Meaning-based themes: diagonal "
                f"{np.nanmean(np.diag(M_th)):.2f}, off-diagonal {np.nanmean(offd_th):.2f} (simple averages over groups here, so "
                f"slightly different from the pooled numbers in Fig. 3). Note that groups are not unrelated: for example "
                f"'AI literacy', 'organisational adoption' and 'workplace impact' remain fairly highly correlated with one "
                f"another, because they jointly reflect each country's AI-adoption maturity, whereas themes such as 'demand "
                f"for regulation' and 'worry about risks' have a low correlation with the rest. That is why one representative "
                f"is kept per broad area rather than assuming that every cluster is independent."))
pdf.savefig(fig)
plt.close(fig)

# ---------- Fig 5: selection inside each group (portrait) ---------------------
sing = [k for k in cl.iloc[leaf_order].drop_duplicates() if csize[k] == 1]
mc_sorted = sorted(multi_clusters, key=lambda k: -res.loc[res.cluster == k, "abs_rho"].max())
rows = []
for k in mc_sorted:
    rows.append(("hdr", tx(f"C{k}  ({int(csize[k])} 个变量，组内互相高度相关)",
                           f"C{k}  ({int(csize[k])} variables, highly inter-correlated)"), None))
    rows += [("var", short(f), f) for f in res[res.cluster == k].sort_values("abs_rho", ascending=False).index]
sv = res[res.cluster.isin(sing)].sort_values("abs_rho", ascending=False)
rows.append(("hdr", tx(f"单成员组 ({len(sv)} 个变量，没有足够相似的伙伴，各自独立)",
                       f"Single-member groups ({len(sv)})"), None))
rows += [("var", short(f), f) for f in sv.index]
nrow = len(rows)
fig = plt.figure(figsize=PORTRAIT)
fig_title(fig, tx("图 5  每个组内的变量对 trust 的影响，以及选择结果",
                  "Fig. 5  Effect on trust within each group, and the selection"),
          tx("条形 = 与 trust 的 Spearman ρ；绿格 = 通过该项检验；绿色底纹 = 最终入选",
             "Bar = Spearman ρ with trust; green cell = passes the test; green shading = finally selected"))
gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.75, 0.95], left=0.27, right=0.985, top=0.865, bottom=0.165, wspace=0.03)
ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
for a_ in (ax1, ax2, ax3):
    a_.set_ylim(nrow - 0.5, -0.5)
lab_v = {"KEEP": tx("入选", "KEEP"), "redundant": tx("被同组替代", "redundant"), "borderline": tx("边缘", "borderline"),
         "drop": tx("剔除", "drop")}
for i, (kind, lab, f) in enumerate(rows):
    if kind == "hdr":
        for a_ in (ax1, ax2, ax3):
            a_.axhspan(i - 0.5, i + 0.5, color="#E4E9EF", zorder=0)
        continue
    r = res.loc[f]
    if r["selected"]:
        for a_ in (ax1, ax2, ax3):
            a_.axhspan(i - 0.5, i + 0.5, color="#DDF2E9", zorder=0)
    ax1.barh(i, r["spearman"], 0.62, color=C_POS if r["spearman"] > 0 else C_NEG, zorder=2,
             alpha=1.0 if r["p_value"] < 0.05 else 0.35)
    cells = [(r["t1_spearman"], "p<.001" if r["p_value"] < 0.001 else f"p={r['p_value']:.3f}"),
             (r["t2_mi"], f"{r['mutual_info']:.2f}"), (r["t3_lasso"], f"{r['lasso_freq']:.2f}"),
             (r["t4_rf"], f"{r['rf_perm_imp']:.3f}")]
    for j, (ok, txt) in enumerate(cells):
        ax2.add_patch(Rectangle((j - 0.46, i - 0.4), 0.92, 0.8, color=C_PASS if ok else C_FAIL, zorder=1))
        ax2.text(j, i, txt, ha="center", va="center", fontsize=5.6, color=C_INK)
    ax3.text(0.05, i, f"{int(r['votes'])}/4  {lab_v[r['verdict']]}", va="center", fontsize=6.6,
             fontweight="bold" if r["selected"] else "normal", color=C_KEEP if r["selected"] else C_INK)
ax1.set_yticks(range(nrow), [lab for kind, lab, _ in rows], fontsize=6.6)
for tl, (kind, _, _) in zip(ax1.get_yticklabels(), rows):
    if kind == "hdr":
        tl.set_fontweight("bold")
ax1.set_xlim(-1, 1); ax1.axvline(0, color="#555555", lw=0.8)
ax1.set_xlabel(tx("与 trust 的 Spearman ρ\n(浅色 = p ≥ 0.05)", "Spearman ρ with trust\n(pale = p ≥ 0.05)"), fontsize=7)
ax1.tick_params(axis="x", labelsize=6.5); ax1.grid(axis="x", color="#DDDDDD", lw=0.5, zorder=0)
ax2.set_xlim(-0.5, 3.5)
ax2.set_xticks(range(4), tx(["① 相关\n|ρ|≥.5\np<.01", "② 互信息\n≥中位数", "③ Lasso\n频率≥50%", "④ RF重要性\n前25%"],
                            ["(1) Correlation\n|ρ|≥.5\np<.01", "(2) Mutual info\n≥ median", "(3) Lasso\nfreq ≥50%",
                             "(4) RF import.\ntop 25%"]), fontsize=5.8)
ax2.xaxis.tick_top(); ax2.tick_params(length=0); ax2.set_yticks([])
ax3.set_xlim(0, 1); ax3.set_xticks([0.5], [tx("得票 / 结论", "votes / result")], fontsize=6.5)
ax3.xaxis.tick_top(); ax3.tick_params(length=0); ax3.set_yticks([])
for a_ in (ax2, ax3):
    for s in a_.spines.values():
        s.set_visible(False)
neg_vars = SEP.join(short(c) for c in res.index[res["spearman"] < 0])
caption(fig, tx(f"读法：每组是一簇互相高度相关的变量(灰色标题行)。组内各变量与 trust 的相关方向一致、强度接近，因此只需选一个：得票≥{VOTES_NEEDED} 且 |ρ| 最大者入选(绿底)。"
                f"“被同组替代”表示通过了检验，但同组有 |ρ| 更高的变量。与 trust 呈负相关的只有 {neg_vars}"
                "(经济越发达、越强调监管，信任越低)，其余变量都是正相关。注意 safeguards_sufficient 等变量与 trust 的高相关，未必意味着因果。",
                f"How to read: each group is a cluster of highly inter-correlated variables (grey rows). Members of a group "
                f"relate to trust in the same direction and with similar strength, so one is enough: the one with "
                f"≥{VOTES_NEEDED} votes and the largest |ρ| is selected (green). 'redundant' = passed the tests, but a "
                f"group-mate has a higher |ρ|. The only negative relationships with trust are {neg_vars}; all other "
                f"variables are positive. A high correlation does not necessarily imply causation."),
        y_top=0.08, size=7.4, width=tx(125, 122))
pdf.savefig(fig)
plt.close(fig)

# ---------- Table 1: themes -------------------------------------------------
fig = plt.figure(figsize=LAND)
fig_title(fig, tx("表 1  含义分组：每个主题的代表变量(同样规则：票数最多，并列取 |ρ| 最大)",
                  "Table 1  Meaning-based themes: representative of each theme"),
          tx(None, "Same rule as before: most votes, ties broken by the largest |ρ|"))
ax = fig.add_axes([0.04, 0.40, 0.92, 0.5]); ax.axis("off")
cells = [[t, int(r.n_members), "-" if np.isnan(r.within_abs_rho) else f"{r.within_abs_rho:.2f}", short(r.best_member),
          f"{r.rho_with_trust:+.2f}", f"{r.votes}/4",
          tx(f"通过(≥{VOTES_NEEDED} 票)", f"passes (≥{VOTES_NEEDED} votes)") if r.passes_rule else tx("未通过", "fails")]
         for t, r in th_pick.iterrows()]
tb = ax.table(cellText=cells,
              colLabels=tx(["主题", "成员数", "组内平均|ρ|", "代表变量", "与 trust 的 ρ", "得票", "是否入选"],
                           ["Theme", "Members", "Within mean |ρ|", "Representative", "ρ with trust", "Votes", "Selected?"]),
              loc="upper center", cellLoc="center", colWidths=[0.24, 0.07, 0.10, 0.24, 0.10, 0.07, 0.14])
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.55)
for (r_, c_), cell in tb.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if r_ == 0:
        cell.set_facecolor("#E4E9EF"); cell.set_text_props(fontweight="bold")
    elif th_pick["passes_rule"].iloc[r_ - 1]:
        cell.set_facecolor("#DDF2E9")
tset = [short(th_pick.loc[t, "best_member"]) for t in THEMES if th_pick.loc[t, "passes_rule"]]
r_pair = corr.loc['org_provides_training_pct', 'increased_revenue_pct']
caption(fig, tx(f"两套分组选出的变量对比：数据驱动 → {'、'.join(short(f) for f in final_set)}；含义分组 → {'、'.join(tset)}(另加 log_gdp)。"
                "含义分组的代表更多，因为它把强相关但含义不同的变量(如培训、素养、收益)分在不同主题里；"
                f"但它选出的 org_provides_training 与 increased_revenue 相关系数高达 {r_pair:.2f}，"
                "放进同一个线性模型会共线。“T2b”“T4a”这类组内平均 |ρ| 很低的主题，说明分组本身就不成立(题目含义相近，但数据上无关)，"
                "它们没有变量通过检验，也符合预期。",
                f"Comparison of the two schemes: data-driven → {', '.join(short(f) for f in final_set)}; meaning-based → "
                f"{', '.join(tset)} (plus log_gdp). The meaning-based scheme has more representatives because it places strongly "
                f"correlated but differently worded variables (training, literacy, benefits) into different themes; but its "
                f"org_provides_training and increased_revenue correlate at {r_pair:.2f}, so they would be collinear in one "
                f"linear model. Themes with a very low within-theme mean |ρ|, such as 'T2b' and 'T4a', show that the grouping "
                f"itself does not hold (similar topic, unrelated in the data); it is expected that no variable in them passes."),
        y_top=0.32)
pdf.savefig(fig)
plt.close(fig)

# ---------- Fig 6: final features vs trust ----------------------------------
fig = plt.figure(figsize=LAND)
fig_title(fig, tx("图 6  最终入选的特征与 trust 的关系", "Fig. 6  Final selected features vs trust"),
          tx(f"最终 {len(final_set)} 个：{'、'.join(short(f) for f in final_set)}",
             f"Final {len(final_set)}: {', '.join(short(f) for f in final_set)}"))
gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.88, bottom=0.30, hspace=0.42, wspace=0.18)
for i, f in enumerate(final_set):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    for typ, col in (("advanced", C_POS), ("emerging", C_NEG)):
        m_ = df["economy_type"] == typ
        ax.scatter(df.loc[m_, f], y[m_], s=16, color=col, alpha=0.85,
                   label=tx("发达经济体" if typ == "advanced" else "新兴经济体",
                            "Advanced economies" if typ == "advanced" else "Emerging economies"))
    for j in df.index:
        ax.annotate(df.loc[j, "iso3"], (df.loc[j, f], y[j]), fontsize=4.8, xytext=(2, 2), textcoords="offset points", color="#555555")
    b1, b0 = np.polyfit(df[f], y, 1)
    xs = np.linspace(df[f].min(), df[f].max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="black", lw=1)
    r_ = res.loc[f]
    ctrl = tx("(控制变量)", " (control)") if f == CONTROL and not r_["selected"] else ""
    ax.set_title(tx(f"{short(f)}   ρ = {r_['spearman']:+.2f}，{int(r_['votes'])}/4 票{ctrl}",
                    f"{short(f)}   ρ = {r_['spearman']:+.2f}, {int(r_['votes'])}/4 votes{ctrl}"), fontsize=9)
    ax.set_xlabel(short(f) + (" (%)" if f.endswith("_pct") else ""), fontsize=8)
    ax.set_ylabel("trust_ai_pct (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7, frameon=False, loc="upper left")
corr_pairs = corr.loc[final_set, final_set].abs().values[np.triu_indices(len(final_set), 1)]
caption(fig, tx("每个点是一个国家，蓝 = 发达经济体、橙 = 新兴经济体，黑线为最小二乘拟合线。前三个变量与 trust 都是清晰的正向关系；"
                "log_gdp 是负相关(人均 GDP 越高，信任越低)，且新兴/发达经济体在图上明显分成两簇，这也是要把它作为控制变量保留的原因。"
                f"入选变量之间的 |ρ| 在 {corr_pairs.min():.2f}–{corr_pairs.max():.2f}，共线性可控(最大 VIF = {vif_final.max():.1f}，见下页)。"
                "这些是国家层面的相关关系，不能解释为因果。",
                "Each point is a country; blue = advanced, orange = emerging economy; the black line is the least-squares fit. "
                "The first three variables have a clear positive relationship with trust; log_gdp is negative (higher GDP per "
                "capita, lower trust), and advanced and emerging economies split into two clusters, which is why it is kept as "
                f"a control variable. |ρ| among the selected variables ranges from {corr_pairs.min():.2f} to "
                f"{corr_pairs.max():.2f}, so collinearity is manageable (max VIF = {vif_final.max():.1f}, next page). These are "
                f"country-level correlations and must not be read as causal."))
pdf.savefig(fig)
plt.close(fig)

# ---------- Fig 7: VIF + performance ----------------------------------------
fig = plt.figure(figsize=LAND)
fig_title(fig, tx("图 7  选择的效果：共线性大幅降低，预测能力没有明显损失",
                  "Fig. 7  Effect of the selection: much lower collinearity, no clear loss of predictive power"))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.10, right=0.97, top=0.88, bottom=0.36, wspace=0.42)
ax = fig.add_subplot(gs[0])
va = vif_all.sort_values()
ax.barh(range(K), va.values, color=[C_KEEP if c in final_set else "#BBBBBB" for c in va.index])
ax.set_yticks(range(K), [short(c) for c in va.index], fontsize=5.5)
ax.set_xscale("log"); ax.axvline(10, color=C_NEG, ls="--", lw=1)
ax.text(10.5, 0, "VIF = 10", color=C_NEG, fontsize=7)
ax.set_xlabel(tx("VIF(对数轴；绿色 = 最终入选)", "VIF (log axis; green = finally selected)"), fontsize=8)
ax.set_title(tx(f"{K} 个候选特征 VIF 最大 {vif_all.max():.0f}；最终 {len(final_set)} 个最大 {vif_final.max():.1f}",
                f"VIF of the {K} candidates: max {vif_all.max():.0f}; final {len(final_set)}: max {vif_final.max():.1f}"), fontsize=9)
ax = fig.add_subplot(gs[1]); ax.axis("off")
cells = [[r["scheme"], int(r["n_features"]), f"{r['cv_r2_mean']:.2f} ± {r['cv_r2_sd']:.2f}", f"{r['cv_rmse']:.1f}", r["note"]]
         for _, r in perf.iterrows()]
tb = ax.table(cellText=cells,
              colLabels=tx(["方案", "特征数", "CV R²(均值±SD)", "RMSE", "备注"],
                           ["Model", "Features", "CV R² ± SD", "RMSE", "Note"]),
              loc="upper center", cellLoc="left", colWidths=[0.36, 0.09, 0.17, 0.08, 0.30])
tb.auto_set_font_size(False); tb.set_fontsize(6.6 if not EN else 6.0); tb.scale(1, 2.0)
for (r_, c_), cell in tb.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if r_ == 0:
        cell.set_facecolor("#E4E9EF"); cell.set_text_props(fontweight="bold")
n_over = int((vif_all > 10).sum())
caption(fig, tx(f"左：VIF 衡量一个变量能被其他变量线性解释的程度，>10 通常认为共线性严重。{K} 个候选特征中有 {n_over} 个超过 10；"
                f"分组选择后只剩 {len(final_set)} 个，最大 VIF 为 {vif_final.max():.1f}。"
                f"右：重复 5 折交叉验证(10 次重复)。最终 {len(final_set)} 个特征的 R² 与 {K} 个特征的全量模型相当，甚至更高；"
                "“折内选择”一行把选特征放进每一折里完成，是最无偏的基准。标为“偏乐观”的两行，其特征是用全部 47 行选出来的，测试折的信息"
                "已经参与了选择，实际泛化表现应更接近“折内选择”一行。R² 的标准差较大(每折只有约 9 个国家)，方案之间的小差异不应过度解读。",
                f"Left: VIF measures how well a variable is explained linearly by the others; >10 is usually taken as severe "
                f"collinearity. {n_over} of the {K} candidates exceed 10; after group-wise selection only {len(final_set)} "
                f"remain, with a max VIF of {vif_final.max():.1f}. Right: repeated 5-fold cross-validation (10 repeats). The R² "
                f"with the final {len(final_set)} features is comparable to, or higher than, the full {K}-feature models; the "
                f"'in-fold selection' row does the feature selection inside every fold and is the least biased benchmark. The "
                f"two rows marked 'optimistic' chose their features on all 47 rows, so information from the test folds already "
                f"influenced the selection; real generalisation should be closer to the 'in-fold' row. The R² standard "
                f"deviation is large (only about 9 countries per fold), so small differences between rows should not be "
                f"over-interpreted."),
        y_top=0.30)
pdf.savefig(fig)
plt.close(fig)

# ---------- limitations -----------------------------------------------------
tp = TextPages()
tp.add([
    ("h1", tx("局限与需要你判断的地方", "Limitations and points for your judgement")),
    ("b", tx("特征是用全部 47 行选出的，随后评估用的还是这批数据，所以“最终特征”的 CV R² 偏乐观；无偏的对照是“折内选择”。"
             "如果之后要做 70/30 的训练/测试划分，严格做法应只用训练集做特征选择。",
             "The features were selected on all 47 rows and then evaluated on the same data, so the CV R² of the final "
             "features is optimistic; the unbiased comparison is the 'in-fold selection' row. If a 70/30 train/test split is "
             "used later, strictly the feature selection should use the training set only.")),
    ("b", tx(f"样本只有 {len(df)} 个国家，而候选特征有 {K} 个。相关系数、Lasso 频率、重要性排名都有较大不确定性，"
             "阈值稍微变化，名单就可能变化(建议改脚本顶部的参数看敏感性)。",
             f"There are only {len(df)} countries but {K} candidate features. The correlations, Lasso frequencies and "
             f"importance rankings are all quite uncertain, and slightly different thresholds can change the list (change the "
             f"parameters at the top of the script to check the sensitivity).")),
    ("b", tx("数据驱动分组的“组内高相关”部分是聚类造成的；含义分组的置换检验才是独立证据。含义分组由我根据题目文字手工划分，"
             "你可以在脚本的 THEMES 字典里修改，所有数字会随之重算。",
             "The 'high correlation within groups' of the data-driven grouping is partly produced by the clustering; the "
             "permutation test on the meaning-based grouping is the independent evidence. The meaning-based grouping is my "
             "hand-made split from the item wording; you can edit the THEMES dictionary in the script and all numbers are "
             "recomputed.")),
    ("b", tx("分组之间也不是独立的：AI 素养、组织采用、职场影响、个人使用等很可能都受同一个潜在因素(各国 AI 采用成熟度)驱动，"
             "入选变量“预测”了 trust，不等于它“导致”了 trust。",
             "Groups are not independent of each other: AI literacy, organisational adoption, workplace impact and personal "
             "use are probably all driven by one latent factor (each country's AI-adoption maturity). A selected variable "
             "'predicting' trust does not mean it 'causes' trust.")),
    ("b", tx("不当使用类变量(mistakes_due_ai、inappropriate_use 等)与 trust 正相关，很可能是“使用越多，出现问题的比例越高”的暴露效应，"
             "不宜解读成“犯错让人更信任”。",
             "The improper-use variables (mistakes_due_ai, inappropriate_use, ...) are positively related to trust, most "
             "likely an exposure effect ('the more people use AI, the higher the share reporting problems'); do not read it "
             "as 'making mistakes increases trust'.")),
    ("b", tx("人工决定共有三处：剔除泄漏变量(accept_ai_pct 等)；聚类阈值 0.25；保留 log_gdp 作为控制变量。它们都在脚本顶部或注释中标明。",
             "There are three manual decisions: removing the leakage variables (accept_ai_pct etc.); the clustering threshold "
             "0.25; keeping log_gdp as a control variable. All are marked at the top of the script or in comments.")),
    ("h2", tx("复现方式", "How to reproduce")),
    ("code", "python feature_selection_report.py             # Chinese\npython feature_selection_report.py --lang en    # English"),
    ("p", tx(f"输出 {OUT.name}/feature_selection_report.pdf，以及 01~11 号 csv：01 相关矩阵；02/03 组内与组间相关；"
             "04 汇总；05 特征表(全部检验结果)；06 主题代表；07 VIF；08 交叉验证表现；09/10 组间矩阵；11 分组名单。"
             "设置环境变量 REPORT_PNG=1 还会把每页另存为 PNG。",
             f"Output: {OUT.name}/feature_selection_report.pdf and the csv files 01-11: 01 correlation matrix; 02/03 within- "
             f"and between-group correlation; 04 pooled summary; 05 feature table (all test results); 06 theme "
             f"representatives; 07 VIF; 08 cross-validated performance; 09/10 between-group matrices; 11 group assignments. "
             f"Set the environment variable REPORT_PNG=1 to also save every page as a PNG.")),
])
tp.close()
pdf.close()
print("PDF written:", OUT / "feature_selection_report.pdf")
