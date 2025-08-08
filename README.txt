
Top-10 Picks — Streamlit (API-Football)
=======================================

1) Zainstaluj zależności:
   pip install -r requirements.txt

2) Ustaw klucz API-Football (Linux/macOS):
   export API_FOOTBALL_KEY=twoj_klucz

   Windows (PowerShell):
   setx API_FOOTBALL_KEY "twoj_klucz"
   (potem otwórz nową konsolę)

3) Uruchom aplikację:
   streamlit run app.py

Co robi MVP:
- Pobiera ligi dla wybranych krajów i sezonu (top-5 zawiera tier 1 i 2; reszta tier 1).
- Pobiera mecze na wybrany dzień.
- Liczy λ (Poisson) na bazie ostatnich N meczów w danej lidze.
- Koryguje λ o proste penalizacje za brakujących zawodników z przewidywanej XI.
- Wylicza prawdopodobieństwa rynków:
  1X&U4.5, 1X&O1.5, X2&U4.5, X2&O1.5, Home 1–3, Away 1–3.
- Wybiera najlepszy rynek per mecz i pokazuje Top-10.

Dalsze kroki (opcjonalne):
- Lepsze ważenie absencji (pozycje, ratingi, xG/90).
- Dixon–Coles (korekta remisów 0–0/1–1).
- Progi filtrów (min. prawdop., min. liczba meczów itp.).
- Cache i retry na API.
