"""
Crop table for the season planner. One row per SokoMkulima crop.

quefts   : which QUEFTS parameter set to use (agro/quefts.py)
ypot     : potential yield of the harvested part (kg DRY MATTER/ha) with ideal water and nutrients, typical
           tropical rain-fed value; real fields reach far less. Approximate, editable.
dm       : dry-matter share of the product as sold (grain ~0.87, fresh cassava ~0.35) -> converts DM to sold kg
days     : growing season length (sowing to maturity)
ky       : FAO yield response to water deficit (FAO Irrigation & Drainage Paper 33): 1 - Ya/Ym = ky (1 - ETa/ETm)
kc       : crop coefficients (initial, mid-season, late) and the share of the season in each stage (FAO-56 style)
"""
CROPS = {
    "maize":     dict(quefts="Maize",      ypot=9000,  dm=0.87, days=120, ky=1.25, kc=(0.4, 1.15, 0.6), stages=(0.2, 0.35, 0.3, 0.15)),
    "sorghum":   dict(quefts="Sorghum",    ypot=6000,  dm=0.88, days=115, ky=0.90, kc=(0.4, 1.0, 0.55), stages=(0.2, 0.3, 0.35, 0.15)),
    "millet":    dict(quefts="Millet",     ypot=4000,  dm=0.88, days=90,  ky=0.90, kc=(0.35, 1.0, 0.3), stages=(0.2, 0.3, 0.35, 0.15)),
    "rice":      dict(quefts="Rice",       ypot=6500,  dm=0.86, days=120, ky=1.30, kc=(1.05, 1.2, 0.9), stages=(0.2, 0.3, 0.35, 0.15)),
    "wheat":     dict(quefts="Wheat",      ypot=6000,  dm=0.87, days=110, ky=1.15, kc=(0.4, 1.15, 0.3), stages=(0.15, 0.25, 0.4, 0.2)),
    "teff":      dict(quefts="Wheat",      ypot=3000,  dm=0.88, days=100, ky=1.0,  kc=(0.4, 1.05, 0.3), stages=(0.2, 0.3, 0.35, 0.15),
                      note="QUEFTS has no teff parameters; wheat parameters are used as the closest small grain."),
    "beans":     dict(quefts="Field bean", ypot=3000,  dm=0.88, days=90,  ky=1.15, kc=(0.4, 1.15, 0.35), stages=(0.2, 0.3, 0.35, 0.15)),
    "cowpea":    dict(quefts="Cowpea",     ypot=2500,  dm=0.88, days=80,  ky=0.85, kc=(0.4, 1.05, 0.35), stages=(0.2, 0.3, 0.35, 0.15)),
    "groundnut": dict(quefts="Groundnut",  ypot=3500,  dm=0.90, days=110, ky=0.70, kc=(0.4, 1.15, 0.6), stages=(0.2, 0.25, 0.4, 0.15)),
    "cassava":   dict(quefts="Cassava",    ypot=10000, dm=0.35, days=300, ky=0.80, kc=(0.3, 0.8, 0.3), stages=(0.1, 0.25, 0.45, 0.2)),
}

# Fertilizer products sold across Africa: nutrient % by weight as ELEMENTS (P = P2O5 x 0.4364, K = K2O x 0.8301).
# Default prices are typical USD per 50 kg bag; they vary a lot by country and year, so the user edits them.
PRODUCTS = {
    "urea":   dict(name="Urea 46-0-0",     N=46, P=0,     K=0,    price=38),
    "dap":    dict(name="DAP 18-46-0",     N=18, P=20.07, K=0,    price=48),
    "npk":    dict(name="NPK 17-17-17",    N=17, P=7.42,  K=14.1, price=42),
    "mop":    dict(name="MOP 0-0-60",      N=0,  P=0,     K=49.8, price=40),
}
BAG_KG = 50
STEP_KG = 25          # advice in half bags (what a smallholder can actually buy)
MAX_KG = {"urea": 150, "dap": 125, "npk": 150, "mop": 100}
ACRE_HA = 0.4047
FARMGATE_SHARE = 0.75  # farmers sell below the retail market price (traders' margin + transport)
MIN_VCR = 2.0          # value-cost ratio: farmers adopt fertilizer when each $1 spent returns at least $2
