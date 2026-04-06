import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Package Price Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stAppViewContainer"] > .main { background-color: #0d1117; }
[data-testid="stHeader"] { background: #0d1117; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #c9d1d9 !important; }
h1, h2, h3, h4 { color: #e6edf3 !important; }
p, li, .stMarkdown { color: #c9d1d9; }

[data-baseweb="tab-list"] {
    background: #161b22; border-radius: 10px;
    padding: 4px; gap: 4px; border: 1px solid #30363d;
}
[data-baseweb="tab"] {
    background: transparent !important; color: #8b949e !important;
    border-radius: 7px !important; font-size: 13px !important;
    font-weight: 500 !important; padding: 6px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #21262d !important; color: #e6edf3 !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 22px !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #21262d !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 8px !important;
}
[data-testid="stRadio"] label { color: #c9d1d9 !important; font-size: 13px !important; }
[data-testid="stExpander"] {
    border: 1px solid #30363d !important;
    border-radius: 10px !important; background: #161b22 !important;
}

.opt-badge {
    border: 2px solid #3fb950; border-radius: 12px;
    padding: 16px 22px; text-align: center; margin: 8px auto;
    max-width: 400px;
}
.opt-lbl { font-size: 12px; color: #7ee787; }
.opt-v   { font-size: 28px; font-weight: 700; color: #3fb950; }

.top-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center;
}
.top-card.best { border-color: #58a6ff; border-width: 2px; }
.top-name   { font-size: 13px; font-weight: 700; color: #e6edf3; }
.top-detail { font-size: 11px; color: #8b949e; margin-top: 2px; }
.badge { font-size: 11px; padding: 3px 12px; border-radius: 20px; font-weight: 600; }
.bi { background: #1f3a5f; color: #58a6ff; }
.bs { background: #1a3a2a; color: #3fb950; }
.bg { background: #21262d; color: #8b949e; }

.ins {
    border-left: 3px solid #58a6ff; border-radius: 0 8px 8px 0;
    padding: 10px 14px; margin: 10px 0 14px;
    font-size: 13px; color: #c9d1d9; line-height: 1.7;
}
.sec {
    font-size: 14px; font-weight: 600; color: #e6edf3;
    border-bottom: 1px solid #30363d; padding-bottom: 8px;
    margin: 1.25rem 0 .75rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", on_bad_lines="skip", engine="python")
    df.columns = [
        c.strip().replace("\xa0", "").replace("\ufeff", "").replace("\r", "").replace("\n", "")
        for c in df.columns
    ]

    col_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    def find(target):
        key = target.lower().replace(" ", "").replace("_", "")
        if key in col_map:
            return col_map[key]
        for actual in df.columns:
            if all(w in actual.lower() for w in target.lower().split()):
                return actual
        raise KeyError(f"'{target}' tidak ditemukan. Kolom: {df.columns.tolist()}")

    df = df.rename(columns={
        find("Total Revenue (Net)"): "Revenue",
        find("package_base_price"):  "Price",
        find("Total Paying Users"):  "Users",
        find("Source_App"):          "App",
        find("Year"):                "Year",
        find("Category_Name"):       "Category",
        find("Content_Name"):        "Package",
        find("Content_Type"):        "ContentType",
    })

    for c in ["Revenue", "Price", "Users"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Revenue", "Price", "Users"])
    df = df[df["Price"] > 1000]
    df["ARPU"] = df["Revenue"] / df["Users"].replace(0, np.nan)
    df["Price_Bucket"] = pd.cut(
        df["Price"],
        bins=[0, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, 100_000_001],
        labels=["<100K", "100-250K", "250-500K", "500K-1M", "1M-5M", ">5M"],
    )
    return df

df = load_data()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def rp(n, short=False):
    if pd.isna(n) or n == 0: return "—"
    if short:
        if n >= 1e9:  return f"Rp {n/1e9:.2f}M"
        if n >= 1e6:  return f"Rp {n/1e6:.1f}jt"
        if n >= 1e3:  return f"Rp {n/1e3:.0f}K"
    return f"Rp {int(n):,}".replace(",", ".")

def num(n): return f"{int(n):,}".replace(",", ".")

THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font_color="#c9d1d9", font_size=12)
GRID  = dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")

def ft(fig, h=360):
    fig.update_layout(**THEME, height=h, margin=dict(l=8, r=8, t=32, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11))
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig

def fit_model(pp, col, degree):
    pts = pp[pp[col] > 0].copy()
    d   = min(degree, max(1, len(pts) - 1))
    m   = Pipeline([("poly", PolynomialFeatures(d)), ("lr", LinearRegression())])
    m.fit(pts[["price"]], pts[col])
    return m

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Filter & Input")
    st.markdown("---")

    all_apps  = sorted(df["App"].unique())
    all_years = sorted(df["Year"].unique())
    all_types = sorted(df["ContentType"].unique())

    sel_apps  = st.multiselect("📱 Aplikasi",    all_apps,  default=all_apps[:5])
    sel_years = st.multiselect("📅 Tahun",       all_years, default=[y for y in all_years if y <= 2025])
    sel_types = st.multiselect("📂 Tipe Konten", all_types, default=all_types)

    st.markdown("---")
    st.markdown("### ⚙️ Model Optimasi")
    goal = st.radio("Optimasi berdasarkan:",
                    ["Revenue Total", "Jumlah Pengguna", "ARPU (Revenue/User)"], index=0)
    poly_degree = st.slider("Derajat Polynomial", 1, 4, 3)

    st.markdown("---")
    st.markdown("### 💡 Simulasi")
    sim_price    = st.number_input("Harga Paket (Rp)", min_value=10_000, max_value=50_000_000,
                                   value=300_000, step=10_000, format="%d")
    target_users = st.number_input("Target Pengguna", min_value=1, max_value=100_000,
                                   value=500, step=10)

# ─── Filter ──────────────────────────────────────────────────────────────────
mask = (
    df["App"].isin(sel_apps  if sel_apps  else all_apps)
    & df["Year"].isin(sel_years if sel_years else all_years)
    & df["ContentType"].isin(sel_types if sel_types else all_types)
)
fdf = df[mask].copy()
if fdf.empty:
    st.error("⚠️ Tidak ada data untuk filter ini. Perluas pilihan di sidebar.")
    st.stop()

# ─── Aggregate ───────────────────────────────────────────────────────────────
pp = (
    fdf.groupby("Price", as_index=False)
    .agg(total_revenue=("Revenue","sum"), total_users=("Users","sum"),
         avg_arpu=("ARPU","median"), count=("Package","count"))
    .rename(columns={"Price":"price"}).sort_values("price")
)

goal_col_map = {
    "Revenue Total":       "total_revenue",
    "Jumlah Pengguna":     "total_users",
    "ARPU (Revenue/User)": "avg_arpu",
}
goal_col  = goal_col_map[goal]
model     = fit_model(pp, goal_col, poly_degree)
p_range   = np.linspace(fdf["Price"].quantile(0.05), fdf["Price"].quantile(0.95), 500)
y_pred    = np.maximum(0, model.predict(p_range.reshape(-1, 1)))
opt_px    = p_range[np.argmax(y_pred)]
opt_val   = y_pred[np.argmax(y_pred)]
sim_pred  = float(np.maximum(0, model.predict([[sim_price]])[0]))
diff_pct  = ((sim_pred - opt_val) / abs(opt_val) * 100) if opt_val != 0 else 0

# ─── Header + KPIs ───────────────────────────────────────────────────────────
st.markdown("# 📦 Package Price Optimizer Dashboard")
st.caption(f"data.csv · {len(fdf):,} baris · {fdf['App'].nunique()} aplikasi · {fdf['Year'].nunique()} tahun")

k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 Total Revenue",  rp(fdf["Revenue"].sum(), True), f"{len(fdf):,} paket")
k2.metric("👥 Total Pengguna", num(fdf["Users"].sum()),        f"{fdf['App'].nunique()} aplikasi")
k3.metric("🏷️ Median Harga",   rp(fdf["Price"].median()),      "harga paling umum")
k4.metric("📊 Median ARPU",    rp(fdf["ARPU"].median()),       "revenue / pengguna")

st.markdown("---")

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Simulasi Harga",
    "🔍 Cari Harga Optimal",
    "📈 Tren Historis",
    "📊 Performa Bucket",
    "🏆 Top Paket",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Simulasi
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec">Simulasi: Jika harganya Rp X, berapa user yang akan membayar?</div>',
                unsafe_allow_html=True)

    model_u = fit_model(pp, "total_users",   poly_degree)
    model_r = fit_model(pp, "total_revenue", poly_degree)
    model_a = fit_model(pp, "avg_arpu",      poly_degree)
    pu = float(np.maximum(0, model_u.predict([[sim_price]])[0]))
    pr = float(np.maximum(0, model_r.predict([[sim_price]])[0]))
    pa = float(np.maximum(0, model_a.predict([[sim_price]])[0]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Harga Simulasi",    rp(sim_price))
    c2.metric("Prediksi Pengguna", num(pu),
              delta=f"{diff_pct:+.1f}% vs optimal ({rp(opt_px)})",
              delta_color="normal" if diff_pct > -20 else "inverse")
    c3.metric("Prediksi Revenue",  rp(pr))
    c4.metric("Prediksi ARPU",     rp(pa))

    st.metric("Estimasi Revenue (Simulasi × Target User)", rp(sim_price * target_users))

    if sim_price < 150_000:
        ins = "💡 Harga sangat rendah — volume tinggi tapi margin tipis. Zona Rp 200K–300K lebih seimbang."
    elif sim_price <= 350_000:
        ins = "✅ Zona sweet spot (Rp 200K–350K). Data historis konsisten menunjukkan volume pengguna tertinggi."
    elif sim_price <= 600_000:
        ins = "📊 Tier menengah. Volume lebih selektif, ARPU lebih tinggi. Cocok untuk paket premium."
    else:
        ins = "💎 Harga premium. ARPU & margin tinggi. Pastikan value proposition sangat kuat."
    st.markdown(f'<div class="ins">{ins}</div>', unsafe_allow_html=True)

    pru = np.maximum(0, model_u.predict(p_range.reshape(-1, 1)))
    prr = np.maximum(0, model_r.predict(p_range.reshape(-1, 1)))

    ca, cb = st.columns(2)
    with ca:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=pp["price"], y=pp["total_users"], mode="markers", name="Aktual",
                                   marker=dict(color="#378ADD", size=7, opacity=.65)))
        fig1.add_trace(go.Scatter(x=p_range, y=pru, mode="lines", name="Prediksi",
                                   line=dict(color="#1D9E75", width=2.5)))
        fig1.add_vline(x=opt_px,    line_dash="dash", line_color="#3fb950",
                       annotation_text=f"Optimal: {rp(opt_px)}", annotation_font_color="#3fb950")
        fig1.add_vline(x=sim_price, line_dash="dot",  line_color="#f78166",
                       annotation_text=f"Simulasi: {rp(sim_price)}", annotation_font_color="#f78166")
        fig1.update_xaxes(title="Harga (Rp)", tickformat=",")
        fig1.update_yaxes(title="Pengguna")
        fig1.update_layout(title="Kurva: Harga vs Pengguna")
        st.plotly_chart(ft(fig1, 360), use_container_width=True)

    with cb:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pp["price"], y=pp["total_revenue"], mode="markers", name="Aktual",
                                   marker=dict(color="#BA7517", size=7, opacity=.65)))
        fig2.add_trace(go.Scatter(x=p_range, y=prr, mode="lines", name="Prediksi",
                                   line=dict(color="#D85A30", width=2.5)))
        fig2.add_vline(x=sim_price, line_dash="dot", line_color="#f78166",
                       annotation_text=f"Simulasi: {rp(sim_price)}", annotation_font_color="#f78166")
        fig2.update_xaxes(title="Harga (Rp)", tickformat=",")
        fig2.update_yaxes(title="Revenue (Rp)")
        fig2.update_layout(title="Kurva: Harga vs Revenue")
        st.plotly_chart(ft(fig2, 360), use_container_width=True)

    st.markdown('<div class="sec">Sensitivitas Harga vs Pengguna</div>', unsafe_allow_html=True)
    sc = fdf[fdf["Price"] <= 5_000_000].copy()
    fig_sc = px.scatter(sc, x="Price", y="Users", color="App", size="Revenue",
                        hover_data=["Package","Year","Revenue"], opacity=.7, size_max=28,
                        labels={"Price":"Harga (Rp)","Users":"Pengguna","App":"Aplikasi"},
                        title="Harga vs Pengguna (ukuran = Revenue)")
    fig_sc.add_vline(x=opt_px, line_dash="dash", line_color="#3fb950",
                     annotation_text="Harga Optimal", annotation_font_color="#3fb950")
    st.plotly_chart(ft(fig_sc, 440), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Cari Harga Optimal
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec">Cari harga terbaik sesuai target Anda</div>', unsafe_allow_html=True)

    oc1, oc2 = st.columns(2)
    opt_metric  = oc1.selectbox("Optimasi berdasarkan:",
                                ["Maksimalkan Pengguna","Maksimalkan Revenue","Maksimalkan ARPU"])
    max_price_f = oc1.slider("Harga maksimum (Rp)", 100_000, 2_000_000, 1_000_000, 50_000,
                              format="Rp %d")
    min_user_f  = oc2.slider("Minimum pengguna historis", 0, int(pp["total_users"].max()), 0, 100)
    min_cnt_f   = oc2.slider("Minimum data paket", 1, 50, 2)

    mk = {"Maksimalkan Pengguna":"total_users", "Maksimalkan Revenue":"total_revenue",
          "Maksimalkan ARPU":"avg_arpu"}[opt_metric]

    fagg = pp[(pp["price"] <= max_price_f) &
              (pp["total_users"] >= min_user_f) &
              (pp["count"] >= min_cnt_f)].copy()

    if fagg.empty:
        st.warning("⚠️ Tidak ada data memenuhi kriteria. Longgarkan filter.")
    else:
        top3  = fagg.sort_values(mk, ascending=False).head(3)
        best  = top3.iloc[0]

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(f"""<div class="opt-badge">
                <div class="opt-lbl">Harga Optimal — {opt_metric}</div>
                <div class="opt-v">{rp(best['price'])}</div>
                <div class="opt-lbl">{num(best['total_users'])} user · {rp(best['total_revenue'], True)}</div>
            </div>""", unsafe_allow_html=True)

        badges = ["bi","bs","bg"]
        blbls  = ["🥇 Terbaik","🥈 #2","🥉 #3"]
        for i, (_, row) in enumerate(top3.iterrows()):
            vstr = (num(row[mk])+" user" if mk == "total_users" else rp(row[mk], True))
            st.markdown(f"""<div class="top-card {'best' if i==0 else ''}">
                <div>
                    <div class="top-name">{rp(row['price'])}</div>
                    <div class="top-detail">{num(row['total_users'])} user · {rp(row['total_revenue'],True)} · {int(row['count'])} paket</div>
                </div>
                <span class="badge {badges[i]}">{blbls[i]} &nbsp; {vstr}</span>
            </div>""", unsafe_allow_html=True)

        fig3 = go.Figure(go.Bar(
            x=[rp(p) for p in fagg.sort_values("price")["price"]],
            y=fagg.sort_values("price")[mk],
            marker_color=["#378ADD" if p == best["price"] else "rgba(55,138,221,.3)"
                          for p in fagg.sort_values("price")["price"]],
            hovertemplate="Harga: %{x}<br>Nilai: %{y:,.0f}<extra></extra>"))
        fig3.update_xaxes(title="Harga", tickangle=45)
        fig3.update_yaxes(title=opt_metric)
        st.plotly_chart(ft(fig3, 360), use_container_width=True)

        fig4 = px.scatter(fagg, x="total_users", y="total_revenue", size="count", color="price",
                          color_continuous_scale="Blues",
                          hover_data={"price":":,","total_users":":,","total_revenue":":,"},
                          labels={"total_users":"Total Pengguna","total_revenue":"Total Revenue","price":"Harga"},
                          size_max=28, title="Peta posisi: Pengguna vs Revenue")
        st.plotly_chart(ft(fig4, 360), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Tren Historis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec">Tren revenue & pengguna per tahun</div>', unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    t_app    = tc1.selectbox("Aplikasi",  ["Semua"] + all_apps, key="t_app")
    t_metric = tc2.selectbox("Metrik", ["Pengguna","Revenue","ARPU Median"], key="t_met")

    tdf = fdf.copy()
    if t_app != "Semua":
        tdf = tdf[tdf["App"] == t_app]

    mcfg = {
        "Pengguna":    ("Users",   "sum",    "Pengguna"),
        "Revenue":     ("Revenue", "sum",    "Revenue (Rp)"),
        "ARPU Median": ("ARPU",    "median", "ARPU Median (Rp)"),
    }
    mc, mf, ml = mcfg[t_metric]
    td = tdf.groupby(["Year","App"])[mc].agg(mf).reset_index().rename(columns={mc: ml})

    ct1, ct2 = st.columns(2)
    with ct1:
        fig_tr = px.line(td, x="Year", y=ml, color="App", markers=True,
                         title=f"{t_metric} per Tahun per Aplikasi",
                         labels={"Year":"Tahun","App":"Aplikasi"})
        fig_tr.update_traces(line_width=2, marker_size=5)
        st.plotly_chart(ft(fig_tr, 370), use_container_width=True)

    with ct2:
        pyr = fdf.groupby("Year")["Price"].median().reset_index()
        fig_py = go.Figure(go.Scatter(x=pyr["Year"], y=pyr["Price"],
                                      mode="lines+markers", line=dict(color="#58a6ff", width=2.5),
                                      marker_size=6))
        fig_py.update_xaxes(title="Tahun", dtick=1)
        fig_py.update_yaxes(title="Harga Median (Rp)", tickformat=",")
        fig_py.update_layout(title="Harga Median Paket per Tahun", showlegend=False)
        st.plotly_chart(ft(fig_py, 370), use_container_width=True)

    st.markdown('<div class="sec">Pertumbuhan YoY Pengguna (%)</div>', unsafe_allow_html=True)
    yoy = (fdf.groupby(["Year","App"])["Users"].sum()
              .reset_index().sort_values(["App","Year"]))
    yoy["pct"] = yoy.groupby("App")["Users"].pct_change() * 100
    yoy_c = yoy.dropna(subset=["pct"])
    if not yoy_c.empty:
        fig_y = px.bar(yoy_c, x="Year", y="pct", color="App", barmode="group",
                       labels={"pct":"Pertumbuhan (%)","Year":"Tahun","App":"Aplikasi"},
                       title="Pertumbuhan YoY per Aplikasi")
        fig_y.add_hline(y=0, line_color="rgba(255,255,255,.2)")
        st.plotly_chart(ft(fig_y, 340), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Bucket
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec">Performa per Bucket Harga</div>', unsafe_allow_html=True)

    bkt = (fdf.groupby("Price_Bucket", observed=True)
              .agg(Jumlah=("Package","count"), Revenue=("Revenue","sum"),
                   Users=("Users","sum"), ARPU=("ARPU","median"))
              .reset_index())

    bk1, bk2, bk3, bk4 = st.columns(4)
    bk1.metric("Total Revenue",           rp(bkt["Revenue"].sum(), True))
    bk2.metric("Total Pengguna",          num(bkt["Users"].sum()))
    bk3.metric("Bucket Terbanyak User",   str(bkt.loc[bkt["Users"].idxmax(),  "Price_Bucket"]))
    bk4.metric("Bucket Revenue Terbesar", str(bkt.loc[bkt["Revenue"].idxmax(),"Price_Bucket"]))

    b1, b2 = st.columns(2)
    with b1:
        f1 = px.bar(bkt, x="Price_Bucket", y="Revenue", color="Price_Bucket",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    title="Revenue per Bucket",
                    labels={"Price_Bucket":"Bucket","Revenue":"Revenue (Rp)"})
        f1.update_layout(showlegend=False)
        st.plotly_chart(ft(f1, 300), use_container_width=True)
    with b2:
        f2 = px.bar(bkt, x="Price_Bucket", y="Users", color="Price_Bucket",
                    color_discrete_sequence=px.colors.sequential.Greens_r,
                    title="Pengguna per Bucket",
                    labels={"Price_Bucket":"Bucket","Users":"Pengguna"})
        f2.update_layout(showlegend=False)
        st.plotly_chart(ft(f2, 300), use_container_width=True)

    b3, b4 = st.columns(2)
    with b3:
        f3 = px.bar(bkt, x="Price_Bucket", y="ARPU", color="Price_Bucket",
                    color_discrete_sequence=px.colors.sequential.Oranges_r,
                    title="ARPU Median per Bucket",
                    labels={"Price_Bucket":"Bucket"})
        f3.update_layout(showlegend=False)
        st.plotly_chart(ft(f3, 290), use_container_width=True)
    with b4:
        f4 = px.bar(bkt, x="Price_Bucket", y="Jumlah", color="Price_Bucket",
                    color_discrete_sequence=px.colors.sequential.Purples_r,
                    title="Jumlah Paket per Bucket",
                    labels={"Price_Bucket":"Bucket"})
        f4.update_layout(showlegend=False)
        st.plotly_chart(ft(f4, 290), use_container_width=True)

    st.markdown('<div class="sec">Tabel Ringkasan</div>', unsafe_allow_html=True)
    disp = bkt.copy()
    disp["Revenue"] = disp["Revenue"].apply(lambda x: rp(x, True))
    disp["ARPU"]    = disp["ARPU"].apply(rp)
    disp["Users"]   = disp["Users"].apply(num)
    disp.columns    = ["Bucket","Paket","Revenue","Pengguna","ARPU Median"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Top Paket
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec">Top 15 Paket Terbaik</div>', unsafe_allow_html=True)

    rank_by = st.radio("Ranking by:", ["Revenue","Pengguna","ARPU"], horizontal=True)
    rmap    = {"Revenue":"Revenue","Pengguna":"Users","ARPU":"ARPU"}

    top15 = (fdf.groupby("Package", as_index=False)
               .agg(Revenue=("Revenue","sum"), Users=("Users","sum"),
                    Price=("Price","median"),  ARPU=("ARPU","median"),
                    App=("App","first"))
               .sort_values(rmap[rank_by], ascending=False).head(15))

    fig_top = px.bar(top15, x=rmap[rank_by], y="Package", orientation="h", color="App",
                     hover_data={"Price":":,","Revenue":":,","Users":":,","ARPU":":,.0f"},
                     title=f"Top 15 Paket by {rank_by}",
                     labels={"Package":"Nama Paket"}, height=580)
    fig_top.update_yaxes(autorange="reversed")
    st.plotly_chart(ft(fig_top, 580), use_container_width=True)

    st.markdown('<div class="sec">Highlight Top 3</div>', unsafe_allow_html=True)
    badges = ["bi","bs","bg"]
    blbls  = ["🥇 #1","🥈 #2","🥉 #3"]
    for i, (_, row) in enumerate(top15.head(3).iterrows()):
        vstr = (rp(row[rmap[rank_by]], True) if rank_by in ["Revenue","ARPU"]
                else num(row[rmap[rank_by]]) + " user")
        st.markdown(f"""<div class="top-card {'best' if i==0 else ''}">
            <div>
                <div class="top-name">{str(row['Package'])[:65]}</div>
                <div class="top-detail">
                    {row['App']} · Harga: {rp(row['Price'])} ·
                    {num(row['Users'])} user · Rev: {rp(row['Revenue'],True)} · ARPU: {rp(row['ARPU'])}
                </div>
            </div>
            <span class="badge {badges[i]}">{blbls[i]} &nbsp; {vstr}</span>
        </div>""", unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📄 Data Mentah (500 baris teratas)"):
    st.dataframe(
        fdf[["Year","App","Category","Package","ContentType","Price","Revenue","Users","ARPU"]]
        .sort_values("Revenue", ascending=False).reset_index(drop=True).head(500),
        use_container_width=True,
    )
st.caption("📌 Package Price Optimizer · data.csv · 2020–2026")