"""CineMatch: Streamlit movie recommender.  Run:  python -m streamlit run app.py"""
import html
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from recommender import Recommender

load_dotenv()
ART_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');
:root { --bg:#050705; --card:rgba(255,255,255,.035); --line:rgba(255,255,255,.09); --line2:rgba(255,255,255,.17);
  --txt:#f2f4f0; --mid:#9aa097; --dim:#6b7168; --glow:rgba(84,128,72,.42);
  --sans:'Inter','Helvetica Neue',Arial,sans-serif; --serif:'Instrument Serif','Playfair Display',Georgia,serif; }
html, body, .stApp { background: radial-gradient(70% 36% at 50% 0%, rgba(70,105,62,.22), transparent 72%), var(--bg) !important;
  color: var(--txt); font-family: var(--sans); }
.block-container { max-width: 1120px; padding: 1.4rem 1.5rem 3rem; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent; }
a { color: inherit; }

/* nav */
.nav { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
.logo { font-family:var(--serif); font-style:italic; font-size:1.9rem; letter-spacing:-.01em; line-height:1; }
.navlinks { display:flex; gap:.3rem; padding:.3rem; border:1px solid var(--line); border-radius:999px; background:var(--card); }
.navlinks a { color:var(--txt); text-decoration:none; font-size:.78rem; padding:.45rem 1rem; border-radius:999px; opacity:.72; border:1px solid transparent; }
.navlinks a.on { background:rgba(255,255,255,.08); border-color:var(--line2); opacity:1; }
.navr { font-size:.78rem; border:1px solid var(--line2); border-radius:999px; padding:.5rem 1.1rem; background:var(--card); }
@media (max-width:760px){ .navlinks{display:none;} }

/* hero */
.hero { position:relative; text-align:center; padding:4rem 0 2rem; }
.star { position:absolute; color:rgba(255,255,255,.22); pointer-events:none; }
.badge { display:inline-block; font-size:.72rem; padding:.38rem .95rem; border:1px solid var(--line2); border-radius:999px; background:var(--card); margin-bottom:1.5rem; }
.h1 { font-weight:300; font-size:clamp(2.7rem,6.4vw,4.8rem); line-height:1.04; letter-spacing:-.03em; margin:0 auto; max-width:14ch; }
.h1 em, .sec .st em, .cta .big em { font-family:var(--serif); font-style:italic; font-weight:400; letter-spacing:-.01em; }
.sub { color:var(--mid); font-size:.98rem; line-height:1.65; max-width:34rem; margin:1.4rem auto 1.7rem; }
.pills { display:flex; gap:.45rem; justify-content:center; flex-wrap:wrap; }
.pill { font-size:.72rem; color:var(--mid); border:1px solid var(--line); border-radius:999px; padding:.32rem .8rem; background:rgba(255,255,255,.02); white-space:nowrap; }
.pill b { color:var(--txt); font-weight:500; }

/* controls card */
.st-key-controls { border:1px solid var(--line); border-radius:28px; background:var(--card); padding:1.5rem 1.6rem 1.2rem; margin-top:.6rem; }
[data-testid="stWidgetLabel"] p { color:var(--mid) !important; font-size:.78rem !important; font-family:var(--sans); }
[data-baseweb="select"] > div { border-radius:999px !important; background:rgba(255,255,255,.04) !important;
  border:1px solid var(--line2) !important; min-height:3rem; padding-left:.5rem; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background:#0b0e0b !important; border-radius:16px !important; }
.stButton > button { border-radius:999px !important; background:#fff !important; color:#000 !important; border:1px solid #fff !important;
  font-family:var(--sans); font-weight:500; padding:.7rem 1.4rem; transition:all .15s; }
.stButton > button p { color:inherit !important; font-weight:500; }
.stButton > button:hover { background:transparent !important; color:#fff !important; }
[data-testid="stRadio"] label p { font-size:.8rem; }

/* featured movie */
.feat { position:relative; overflow:hidden; text-align:center; border:1px solid var(--line2); border-radius:30px;
  padding:2.4rem 1.6rem 2.3rem; margin:1.8rem 0 .5rem;
  background:radial-gradient(75% 90% at 50% 135%, var(--glow), transparent 62%), var(--card); }
.feat .av { width:78px; height:78px; border-radius:50%; object-fit:cover; object-position:top; border:1px solid var(--line2); margin-bottom:.9rem; }
.feat .lab { font-size:.72rem; color:var(--mid); }
.feat .ft { font-family:var(--serif); font-size:clamp(2.2rem,5vw,3.4rem); line-height:1.05; margin:.35rem 0 .9rem; letter-spacing:-.01em; }
.feat .ov { color:var(--mid); font-size:.92rem; line-height:1.7; max-width:44rem; margin:1.2rem auto 0; }

/* results */
.sec { display:flex; align-items:baseline; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin:3.2rem 0 1.3rem; }
.sec .st { font-weight:300; font-size:clamp(1.9rem,4vw,2.7rem); letter-spacing:-.02em; }
.sec .cnt { color:var(--mid); font-size:.8rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:2rem 1.2rem; }
.art { position:relative; aspect-ratio:2/3; border-radius:22px; overflow:hidden; border:1px solid var(--line);
  background:linear-gradient(165deg,rgba(255,255,255,.07),rgba(255,255,255,.015)); }
.tile.top .art { border-color:var(--line2);
  background:radial-gradient(90% 70% at 50% 112%, var(--glow), transparent 65%), linear-gradient(165deg,rgba(255,255,255,.08),rgba(255,255,255,.02)); }
.art img { width:100%; height:100%; object-fit:cover; display:block; }
.art.bw img { filter:grayscale(1) contrast(1.05); }
.art .rk { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-family:var(--serif); font-style:italic; font-size:5.6rem; color:rgba(255,255,255,.92); }
.art .tag { position:absolute; left:.8rem; bottom:.8rem; font-size:.68rem; padding:.28rem .7rem; border-radius:999px;
  background:rgba(0,0,0,.5); border:1px solid var(--line2); backdrop-filter:blur(6px); }
.art .rn { position:absolute; left:.8rem; top:.8rem; font-size:.68rem; padding:.28rem .7rem; border-radius:999px;
  background:rgba(0,0,0,.5); border:1px solid var(--line2); backdrop-filter:blur(6px); }
.tile .tt { font-size:1rem; font-weight:500; line-height:1.25; margin:.9rem 0 .55rem; letter-spacing:-.01em; }
.tile .pills { justify-content:flex-start; }
.tile .why { margin-top:.75rem; display:flex; flex-direction:column; gap:.2rem; font-size:.72rem; color:var(--mid); line-height:1.45; }
.tile .why b { color:var(--dim); font-weight:500; margin-right:.35rem; }

/* how it works */
.how { display:grid; grid-template-columns:repeat(3,1fr); gap:1.1rem; }
.hc { border:1px solid var(--line); border-radius:26px; padding:1.7rem; background:var(--card); }
.hc .n { font-family:var(--serif); font-style:italic; font-size:2.6rem; line-height:1; }
.hc .t { font-size:1.05rem; font-weight:500; margin:.7rem 0 .45rem; }
.hc .d { color:var(--mid); font-size:.84rem; line-height:1.65; }
@media (max-width:800px){ .how{grid-template-columns:1fr;} }

/* closing section + footer */
.cta { display:flex; align-items:flex-end; justify-content:space-between; gap:2rem; flex-wrap:wrap; margin:4.5rem 0 2.6rem; }
.ring { width:112px; height:112px; border-radius:50%; border:1px solid var(--line2); display:flex; align-items:center; justify-content:center;
  font-family:var(--serif); font-style:italic; font-size:2.4rem; }
.cta .r { text-align:right; margin-left:auto; }
.cta .l { color:var(--mid); font-size:1.5rem; font-weight:300; }
.cta .big { font-weight:300; font-size:clamp(2.4rem,6vw,4.6rem); line-height:1.02; letter-spacing:-.03em; }
.cta .go { display:inline-flex; align-items:center; gap:.7rem; font-size:.95rem; margin-top:1.1rem; color:#fff; text-decoration:underline; }
.circ { width:34px; height:34px; border-radius:50%; background:#fff; color:#000; display:inline-flex; align-items:center; justify-content:center; text-decoration:none; }
.foot { display:grid; grid-template-columns:1.7fr 1fr; gap:1.1rem; }
@media (max-width:800px){ .foot{grid-template-columns:1fr;} }
.fp { position:relative; border:1px solid var(--line); border-radius:26px; background:var(--card); padding:1.7rem; min-height:220px; overflow:hidden; }
.fp .logo { font-size:1.7rem; margin-bottom:.8rem; }
.fp .d { color:var(--mid); font-size:.82rem; line-height:1.65; max-width:22rem; }
.fp h4 { font-size:1.05rem; font-weight:400; margin:0 0 1rem; }
.fp .fl { position:absolute; right:1.5rem; bottom:1.5rem; display:flex; gap:.4rem; flex-wrap:wrap; justify-content:flex-end; max-width:60%; }
.fp .fl a { text-decoration:none; }
.fp .copy { position:absolute; left:1.7rem; bottom:1.5rem; right:1.7rem; font-size:.7rem; color:var(--dim); line-height:1.5; }
.fp .tags { display:flex; gap:.4rem; flex-wrap:wrap; justify-content:flex-start; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if not (ART_DIR / "movies.pkl").exists():
    st.error("No model found. Download the TMDB CSVs into ./data and run `python build_model.py` first.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def get_recommender():
    return Recommender(ART_DIR)


rec = get_recommender()
esc = html.escape


# ---------- optional posters via the free TMDB API ----------
def get_api_key():
    key = os.getenv("TMDB_API_KEY")
    if not key:
        try:
            key = st.secrets.get("TMDB_API_KEY")
        except Exception:
            key = None
    return key if key and key != "your_key_here" else None


@lru_cache(maxsize=4096)
def poster_url(movie_id, key):
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": key}, timeout=6
        )
        path = r.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except Exception:
        return None


API_KEY = get_api_key()


def pill(text, strong=False):
    return f'<span class="pill">{"<b>" + esc(text) + "</b>" if strong else esc(text)}</span>'


# ---------- nav + hero ----------
st.markdown(
    f"""
<div class="nav">
  <div class="logo">cinematch</div>
  <div class="navlinks"><a class="on" href="#discover" target="_self">Discover</a>
    <a href="#how" target="_self">How it works</a><a href="#about" target="_self">About</a></div>
  <div class="navr">{len(rec.movies):,} films</div>
</div>
<div class="hero">
  <span class="star" style="left:12%;top:5.5rem;font-size:1.6rem">✦</span>
  <span class="star" style="left:21%;top:13rem;font-size:.8rem">✦</span>
  <span class="star" style="right:15%;top:7rem;font-size:1rem">✦</span>
  <span class="star" style="right:9%;top:15rem;font-size:1.8rem">✦</span>
  <div class="badge">Content-based recommender</div>
  <div class="h1">Find your next <em>favorite film</em></div>
  <div class="sub">Pick a movie you love. CineMatch finds more that share its plot, themes, cast and director,
    and tells you exactly why each one matched.</div>
  <div class="pills">{pill(f"{len(rec.movies):,} films", True)}{pill(f"{rec.X.shape[1]:,} features", True)}{pill("TF-IDF", True)}{pill("Cosine similarity", True)}</div>
</div>
<div id="discover"></div>
""",
    unsafe_allow_html=True,
)

# ---------- controls ----------
if "movie" not in st.session_state:
    st.session_state.movie = next(
        (l for l in rec.labels if l.startswith("The Dark Knight (")), rec.labels[0]
    )


def surprise():
    st.session_state.movie = random.choice(rec.labels)


bw = False
with st.container(key="controls"):
    c1, c2 = st.columns([4, 1])
    choice = c1.selectbox("Pick a movie (type to search)", rec.labels, key="movie")
    c2.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    c2.button("Surprise me", on_click=surprise, use_container_width=True)
    c3, c4 = st.columns([3, 2])
    n = c3.slider("Number of recommendations", 5, 20, 10, step=5)
    if API_KEY:
        bw = c4.radio("Posters", ["Color", "Black & white"], horizontal=True) == "Black & white"

idx = rec.label_to_idx[choice]
q = rec.info(idx)
q_cast = list(rec.movies.iloc[idx]["cast"])
results = rec.recommend(idx, n)

# ---------- featured movie ----------
avatar = ""
if API_KEY:
    url = poster_url(q["id"], API_KEY)
    if url:
        avatar = f'<img class="av" src="{esc(url)}" alt="">'
q_pills = "".join(
    [pill(str(q["year"])) if q["year"] else "", pill(f"★ {q['rating']:.1f}")]
    + [pill(g) for g in q["genres"]]
    + ([pill(f"Dir. {q['director']}")] if q["director"] else [])
    + [pill(c) for c in q_cast]
)
overview = q["overview"][:320] + ("..." if len(q["overview"]) > 320 else "")
st.markdown(
    f"""<div class="feat">{avatar}<div class="lab">Because you picked</div>
<div class="ft">{esc(q['title'])}</div><div class="pills">{q_pills}</div>
<div class="ov">{esc(overview)}</div></div>""",
    unsafe_allow_html=True,
)

# ---------- recommendations ----------
posters = {}
if API_KEY:
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r, url in zip(results, pool.map(lambda r: poster_url(r["id"], API_KEY), results)):
            posters[r["id"]] = url

cards = []
for rank, r in enumerate(results, start=1):
    url = posters.get(r["id"])
    tag = f'<span class="tag">{esc(r["genres"][0])}</span>' if r["genres"] else ""
    if url:
        art = (
            f'<div class="art {"bw" if bw else ""}"><img src="{esc(url)}" alt="{esc(r["title"])}">'
            f'<span class="rn">{rank:02d}</span>{tag}</div>'
        )
    else:
        art = f'<div class="art"><div class="rk">{rank:02d}</div>{tag}</div>'
    meta = "".join(
        [pill(str(r["year"])) if r["year"] else "", pill(f"★ {r['rating']:.1f}"), pill(f"sim {r['score']:.2f}", True)]
    )
    why = "".join(
        f"<div><b>{esc(w.split(': ')[0])}</b>{esc(w.split(': ', 1)[1])}</div>"
        if ": " in w
        else f"<div>{esc(w)}</div>"
        for w in r["why"]
    )
    cards.append(
        f"""<div class="tile{' top' if rank == 1 else ''}">{art}<div class="tt">{esc(r['title'])}</div>
<div class="pills">{meta}</div><div class="why">{why}</div></div>"""
    )

st.markdown(
    f"""<div class="sec"><div class="st">Films <em>you might also like</em></div>
<div class="cnt">{len(results)} matches, ranked by similarity</div></div>
<div class="grid">{"".join(cards)}</div>""",
    unsafe_allow_html=True,
)

# ---------- how it works ----------
st.markdown(
    """
<div id="how"></div>
<div class="sec"><div class="st">How it <em>works</em></div><div class="cnt">No ratings needed, just the movies themselves</div></div>
<div class="how">
  <div class="hc"><div class="n">01</div><div class="t">Profile every film</div>
    <div class="d">Each movie becomes one text profile: its plot summary, genres, keywords, top three cast members and director.</div></div>
  <div class="hc"><div class="n">02</div><div class="t">Turn text into numbers</div>
    <div class="d">TF-IDF gives rare, meaningful words (a director's name, a niche theme) more weight than common ones.</div></div>
  <div class="hc"><div class="n">03</div><div class="t">Compare and explain</div>
    <div class="d">Cosine similarity ranks every other film, and the shared genres, cast and themes are shown as the reason.</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- closing section + footer ----------
year = datetime.now().year
data_note = (
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
    if API_KEY
    else "Data: TMDB 5000 Movie Dataset (Kaggle)."
)
st.markdown(
    f"""
<div class="cta">
  <div class="ring">cm</div>
  <div class="r"><div class="l">Ready for another?</div>
    <div class="big">Pick a <em>new movie</em></div>
    <a class="go" href="#discover" target="_self">Back to the top <span class="circ">↑</span></a></div>
</div>
<div id="about"></div>
<div class="foot">
  <div class="fp"><div class="logo">cinematch</div>
    <div class="d">Content-based movie recommendations built with scikit-learn, with a plain-English reason for every match.</div>
    <div class="fl"><a class="pill" href="#discover" target="_self">Discover</a><a class="pill" href="#how" target="_self">How it works</a></div>
    <div class="copy">© {year} CineMatch. {esc(data_note)}</div></div>
  <div class="fp"><h4>Under the hood</h4>
    <div class="tags">{pill("Pandas")}{pill("Scikit-learn")}{pill("TF-IDF")}{pill("Cosine similarity")}{pill("Streamlit")}</div></div>
</div>
""",
    unsafe_allow_html=True,
)