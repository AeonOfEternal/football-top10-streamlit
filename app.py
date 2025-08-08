
import os
import math
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import streamlit as st

# ------------------------------
# Config
# ------------------------------
API_KEY = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"

DEFAULT_COUNTRIES = [
    # Top-5 (DE, FR, EN, ES, IT) with tiers 1 & 2
    "Germany", "France", "England", "Spain", "Italy",
    # 1st division others:
    "Portugal", "Netherlands", "Belgium", "Poland", "Czech Republic",
    "Croatia", "Austria", "Switzerland", "Denmark", "Norway", "Sweden"
]

HEADERS = {"x-apisports-key": API_KEY}

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data(show_spinner=False, ttl=6*3600)
def api_get(path, params=None):
    if not API_KEY:
        raise RuntimeError("Brak API key. Ustaw zmienną środowiskową API_FOOTBALL_KEY.")
    url = f"{BASE_URL}{path}"
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        st.warning(f"API returned errors: {data['errors']}")
    return data

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def score_matrix(lambda_home, lambda_away, max_goals=7):
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            probs[i, j] = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)
    return probs

def probs_from_matrix(M):
    p_home = np.tril(M, -1).sum()     # i>j
    p_draw = np.trace(M)
    p_away = np.triu(M, 1).sum()      # j>i
    totals = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    p_u45 = M[totals <= 4].sum()
    p_o15 = M[totals >= 2].sum()
    return p_home, p_draw, p_away, p_u45, p_o15

def team_goals_prob(M, team="home", low=1, high=3):
    if team == "home":
        return M[low:high+1, :].sum()
    else:
        return M[:, low:high+1].sum()

# ------------------------------
# League discovery
# ------------------------------
@st.cache_data(show_spinner=False, ttl=12*3600)
def discover_leagues(countries, season):
    # Pull all leagues and filter by country + type=League + current season coverage
    res = api_get("/leagues", params={"season": season})
    leagues = res.get("response", [])
    rows = []
    for item in leagues:
        lg = item.get("league", {})
        cn = item.get("country", {})
        seasons = item.get("seasons", [])
        if not lg or not cn:
            continue
        if cn.get("name") not in countries:
            continue
        # keep season if current==True or equals season
        has_season = any(s.get("year") == season for s in seasons)
        if not has_season:
            continue
        if lg.get("type") != "League":
            continue
        rows.append({
            "league_id": lg.get("id"),
            "league_name": lg.get("name"),
            "country": cn.get("name")
        })
    df = pd.DataFrame(rows)
    # Select tiers:
    # - For top-5: keep names containing "Bundesliga 2", "Ligue 2", "Championship", "La Liga 2/Segunda", "Serie B" and their tier-1 counterparts.
    # - For other countries: keep only top leagues (heuristics by name).
    def tier_tag(row):
        name = row["league_name"].lower()
        country = row["country"]
        if country in ["Germany", "France", "England", "Spain", "Italy"]:
            # tier 2 patterns
            if ("2. bundesliga" in name) or ("bundesliga 2" in name) or ("zweite bundesliga" in name):
                return 2
            if ("ligue 2" in name):
                return 2
            if ("championship" in name):
                return 2
            if ("segunda" in name) or ("la liga 2" in name) or ("laliga2" in name):
                return 2
            if ("serie b" in name):
                return 2
            # tier 1:
            if ("bundesliga" in name and "2" not in name) or ("ligue 1" in name) or ("premier league" in name) or ("la liga" in name and "2" not in name) or ("serie a" in name):
                return 1
            return 0  # ignore other cups/reserves
        else:
            # try to detect main league by common naming
            if any(k in name for k in ["superliga", "super league", "premier league", "ekstraklasa", "eredivisie", "jupiler", "pro league", "1. hnl", "1. liga", "tipico bundesliga", "bundesliga", "superettan", "allsvenskan", "eliteserien", "superligaen", "fortuna liga"]):
                return 1
            # exclude seconds/first divisions named explicitly
            if any(k in name for k in ["2. liga", "liga 2", "first division b", "division 2"]):
                return 0
            # fallback: include only top-most by country if single
            return 1
    if df.empty:
        return df
    df["tier"] = df.apply(tier_tag, axis=1)
    # Keep tier=1 & tier=2 for top-5; tier=1 for others
    mask = ((df["country"].isin(["Germany","France","England","Spain","Italy"]) & df["tier"].isin([1,2])) |
            (~df["country"].isin(["Germany","France","England","Spain","Italy"]) & (df["tier"]==1)))
    return df[mask].reset_index(drop=True)

# ------------------------------
# Data collection per fixture
# ------------------------------
@st.cache_data(show_spinner=False, ttl=1*3600)
def get_fixtures(league_ids, day):
    rows = []
    for lid in league_ids:
        res = api_get("/fixtures", params={"league": lid, "date": day.strftime("%Y-%m-%d")})
        for f in res.get("response", []):
            league = f.get("league", {})
            teams = f.get("teams", {})
            fixture = f.get("fixture", {})
            if not teams or not fixture:
                continue
            if not teams.get("home", {}).get("id") or not teams.get("away", {}).get("id"):
                continue
            rows.append({
                "fixture_id": f.get("fixture",{}).get("id"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "country": league.get("country"),
                "date": fixture.get("date"),
                "home_id": teams.get("home",{}).get("id"),
                "home_name": teams.get("home",{}).get("name"),
                "away_id": teams.get("away",{}).get("id"),
                "away_name": teams.get("away",{}).get("name")
            })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=1*3600)
def recent_team_goals(team_id, league_id, last_n=10, season=None):
    # Pull last_n results in the league/season for team_id
    params = {"team": team_id, "league": league_id}
    if season:
        params["season"] = season
    params["last"] = last_n
    res = api_get("/fixtures", params=params)
    g_for_home = g_against_home = g_for_away = g_against_away = gp_home = gp_away = 0
    for r in res.get("response", []):
        if r.get("teams",{}).get("home",{}).get("id")==team_id:
            gp_home += 1
            g_for_home += r.get("goals",{}).get("home",0) or 0
            g_against_home += r.get("goals",{}).get("away",0) or 0
        elif r.get("teams",{}).get("away",{}).get("id")==team_id:
            gp_away += 1
            g_for_away += r.get("goals",{}).get("away",0) or 0
            g_against_away += r.get("goals",{}).get("home",0) or 0
    return {
        "gp_home": gp_home, "gf_home": g_for_home, "ga_home": g_against_home,
        "gp_away": gp_away, "gf_away": g_for_away, "ga_away": g_against_away
    }

@st.cache_data(show_spinner=False, ttl=30*60)
def injuries_for_team(team_id, season=None):
    # API-Football injuries endpoint
    params = {"team": team_id}
    if season:
        params["season"] = season
    res = api_get("/injuries", params=params)
    # Return a set of unavailable player names (best-effort)
    unavail = []
    for item in res.get("response", []):
        player = item.get("player",{}).get("name")
        reason = item.get("player",{}).get("reason") or ""
        if player:
            unavail.append((player, reason))
    return unavail

@st.cache_data(show_spinner=False, ttl=30*60)
def probable_lineup(fixture_id):
    res = api_get("/fixtures/lineups", params={"fixture": fixture_id})
    if not res.get("response"):
        return {"home": [], "away": []}
    data = res["response"]
    # response has two entries: home and away
    lineups = {"home": [], "away": []}
    for x in data:
        team = x.get("team",{})
        start = x.get("startXI", []) or []
        players = [p.get("player",{}).get("name") for p in start if p.get("player")]
        # match to home/away by first item team id? fallback to order
        if not lineups["home"]:
            lineups["home"] = players
        else:
            lineups["away"] = players
    return lineups

# ------------------------------
# Strengths + expected goals
# ------------------------------
def safe_div(a, b, fallback=1.0):
    try:
        b = float(b)
        return float(a)/b if b!=0 else fallback
    except:
        return fallback

def build_strengths(fixtures_df, league_ids, last_n=10, season=None):
    # League averages & team strengths on the fly from recent data
    # 1) Compute per-league averages (home/away goals per game)
    league_avgs = {}
    team_stats = {}

    # First, accumulate team stats per league
    for lid in set(league_ids):
        # pick teams from fixtures_df with this league
        teams = set(fixtures_df[fixtures_df["league_id"]==lid]["home_id"]).union(set(fixtures_df[fixtures_df["league_id"]==lid]["away_id"]))
        agg = {"gp_home":0,"gf_home":0,"ga_home":0,"gp_away":0,"gf_away":0,"ga_away":0}
        for t in teams:
            s = recent_team_goals(t, lid, last_n=last_n, season=season)
            team_stats[(lid,t)] = s
            agg["gp_home"] += s["gp_home"]; agg["gf_home"] += s["gf_home"]; agg["ga_home"] += s["ga_home"]
            agg["gp_away"] += s["gp_away"]; agg["gf_away"] += s["gf_away"]; agg["ga_away"] += s["ga_away"]
        avg_home = safe_div(agg["gf_home"], agg["gp_home"], fallback=1.4)
        avg_away = safe_div(agg["gf_away"], agg["gp_away"], fallback=1.2)
        league_avgs[lid] = {"home_g": avg_home, "away_g": avg_away}

    # 2) Convert each team stat into strength multipliers
    strengths = {}
    for (lid, tid), s in team_stats.items():
        avg_home = league_avgs[lid]["home_g"]
        avg_away = league_avgs[lid]["away_g"]
        att_home = safe_div(safe_div(s["gf_home"], s["gp_home"], 1.0), avg_home, 1.0)
        def_home = safe_div(safe_div(s["ga_home"], s["gp_home"], 1.0), avg_away, 1.0)
        att_away = safe_div(safe_div(s["gf_away"], s["gp_away"], 1.0), avg_away, 1.0)
        def_away = safe_div(safe_div(s["ga_away"], s["gp_away"], 1.0), avg_home, 1.0)
        strengths[(lid, tid)] = {
            "att_home": att_home, "def_home": def_home,
            "att_away": att_away, "def_away": def_away
        }
    return league_avgs, strengths

def expected_goals(home_id, away_id, league_id, league_avgs, strengths, base_home_adv=0.12):
    avg_home = league_avgs[league_id]["home_g"]
    avg_away = league_avgs[league_id]["away_g"]
    H = strengths.get((league_id, home_id), {"att_home":1.0,"def_home":1.0,"att_away":1.0,"def_away":1.0})
    A = strengths.get((league_id, away_id), {"att_home":1.0,"def_home":1.0,"att_away":1.0,"def_away":1.0})
    lam_h = max(0.05, avg_home * H["att_home"] * A["def_away"] * (1.0 + base_home_adv))
    lam_a = max(0.05, avg_away * A["att_away"] * H["def_home"] * (1.0 - base_home_adv/2))
    return lam_h, lam_a

def adjust_for_absences(lam_h, lam_a, home_lineup, away_lineup, home_unavail, away_unavail):
    # Very light-touch impact using simple heuristics:
    # If a player listed in probable XI is unavailable -> apply penalty.
    # No positions → use a flat small penalty per missing player, capped.
    def penalty(lineup, unavail):
        if not lineup:
            return 0.0
        names_in_xi = set([n for n in lineup if n])
        missing = [n for (n,_) in unavail if n in names_in_xi]
        # each missing starter: -3% to own attack; +2% to opponent attack (proxy for defensive holes)
        att_pen = -0.03 * len(missing)
        opp_boost = 0.02 * len(missing)
        # cap
        att_pen = max(att_pen, -0.20)
        opp_boost = min(opp_boost, 0.15)
        return att_pen, opp_boost

    h_att_pen, a_opp_boost = penalty(home_lineup, home_unavail)
    a_att_pen, h_opp_boost = penalty(away_lineup, away_unavail)

    lam_h_adj = max(0.03, lam_h * (1.0 + h_att_pen) * (1.0 + h_opp_boost))
    lam_a_adj = max(0.03, lam_a * (1.0 + a_att_pen) * (1.0 + a_opp_boost))
    return lam_h_adj, lam_a_adj

# ------------------------------
# Markets
# ------------------------------
def combined_market_probs(M):
    maxg = M.shape[0]-1
    p_1x_u45 = p_1x_o15 = p_x2_u45 = p_x2_o15 = 0.0
    for i in range(maxg+1):
        for j in range(maxg+1):
            prob = M[i,j]
            if prob <= 0: continue
            total = i + j
            # 1X
            if i >= j:
                if total <= 4: p_1x_u45 += prob
                if total >= 2: p_1x_o15 += prob
            # X2
            if j >= i:
                if total <= 4: p_x2_u45 += prob
                if total >= 2: p_x2_o15 += prob
    return p_1x_u45, p_1x_o15, p_x2_u45, p_x2_o15

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Top-10 Picks (Poisson + Injuries)", layout="wide")

st.title("Top-10 Picks — Poisson + absencje (API-Football)")
st.caption("Wybierz kraje/liga, dzień i wygeneruj typy: 1X&U4.5 / 1X&O1.5 / X2&U4.5 / X2&O1.5 oraz Gole 1–3 (Home/Away).")

if not API_KEY:
    st.error("Brak API key. Ustaw zmienną środowiskową: API_FOOTBALL_KEY")
    st.stop()

season = st.number_input("Sezon (rok startu, np. 2024)", min_value=2010, max_value=2100, value=int(datetime.utcnow().year))
countries = st.multiselect("Kraje", DEFAULT_COUNTRIES, default=DEFAULT_COUNTRIES)

with st.spinner("Wyszukiwanie lig..."):
    leagues_df = discover_leagues(countries, season)
if leagues_df.empty:
    st.warning("Nie znaleziono lig dla wskazanych krajów/sezonu.")
    st.stop()

# Split for top-5 (allow tier 1&2), others tier=1 already filtered
st.dataframe(leagues_df, use_container_width=True)

target_date = st.date_input("Dzień meczów", value=date.today())
last_n = st.slider("Ile ostatnich meczów do formy/sił (per liga)", min_value=6, max_value=20, value=10)

if st.button("Generate Top 10"):
    with st.spinner("Pobieram mecze i liczę..."):
        fx = get_fixtures(leagues_df["league_id"].tolist(), target_date)
        if fx.empty:
            st.warning("Brak meczów w wybranym dniu.")
            st.stop()

        # Build strengths
        lg_avgs, strengths = build_strengths(fx, leagues_df["league_id"].tolist(), last_n=last_n, season=season)

        rows = []
        for _, r in fx.iterrows():
            try:
                lam_h, lam_a = expected_goals(r["home_id"], r["away_id"], r["league_id"], lg_avgs, strengths)
                # Injuries/lineups
                lineup = probable_lineup(r["fixture_id"])
                un_h = injuries_for_team(r["home_id"], season=season)
                un_a = injuries_for_team(r["away_id"], season=season)
                lam_h, lam_a = adjust_for_absences(lam_h, lam_a, lineup.get("home",[]), lineup.get("away",[]), un_h, un_a)

                M = score_matrix(lam_h, lam_a, max_goals=7)
                p_home, p_draw, p_away, p_u45, p_o15 = probs_from_matrix(M)
                p_1x_u45, p_1x_o15, p_x2_u45, p_x2_o15 = combined_market_probs(M)
                p_home_1_3 = team_goals_prob(M, team="home", low=1, high=3)
                p_away_1_3 = team_goals_prob(M, team="away", low=1, high=3)

                markets = {
                    "1X & U4.5": p_1x_u45,
                    "1X & O1.5": p_1x_o15,
                    "X2 & U4.5": p_x2_u45,
                    "X2 & O1.5": p_x2_o15,
                    "Home 1–3": p_home_1_3,
                    "Away 1–3": p_away_1_3
                }
                best_market = max(markets, key=markets.get)
                best_prob = markets[best_market]

                rows.append({
                    "Date": r["date"],
                    "League": r["league_name"],
                    "Home": r["home_name"],
                    "Away": r["away_name"],
                    "λ Home": round(lam_h,3),
                    "λ Away": round(lam_a,3),
                    "Pr(1X & U4.5)": round(p_1x_u45,3),
                    "Pr(1X & O1.5)": round(p_1x_o15,3),
                    "Pr(X2 & U4.5)": round(p_x2_u45,3),
                    "Pr(X2 & O1.5)": round(p_x2_o15,3),
                    "Pr(Home 1–3)": round(p_home_1_3,3),
                    "Pr(Away 1–3)": round(p_away_1_3,3),
                    "Best Market": best_market,
                    "Best Prob": round(best_prob,3),
                    "Injury H (count)": len(un_h),
                    "Injury A (count)": len(un_a)
                })
            except Exception as e:
                rows.append({
                    "Date": r["date"], "League": r["league_name"],
                    "Home": r["home_name"], "Away": r["away_name"],
                    "Error": str(e)
                })

        out = pd.DataFrame(rows)
        top10 = out.sort_values("Best Prob", ascending=False, na_position="last").head(10)
        st.subheader("Top 10 Picks")
        st.dataframe(top10, use_container_width=True)

        st.download_button("Pobierz Top 10 (CSV)", data=top10.to_csv(index=False), file_name="top10.csv", mime="text/csv")
        st.download_button("Pobierz Wszystkie (CSV)", data=out.to_csv(index=False), file_name="all_picks.csv", mime="text/csv")

        st.caption("Uwaga: prosta korekta absencji: -3% atak / +2% atak rywala na brak każdego startera (cap). Do kalibracji na historii.")
