# SokoMkulima — *the farmer's market*

**One platform for an African smallholder farmer's whole year**, by web or SMS in 11 African languages:

| Page | Gap it closes | How |
|---|---|---|
| **Sell or store** (`/`) | Farmers sell at harvest when prices are lowest | Dynamic spatio-temporal graph neural network forecasts crop prices in hundreds of African markets 1–3 months ahead (live WFP data) |
| **Plan the season** (`/plan`) | Yield gap: farms produce half or less of what their land could | Soil maps + 35 years of rain + QUEFTS crop-nutrient model + price forecasts -> planting window and the fertilizer that pays |
| **Insure the crop** (`/insure`) | Climate risk: almost no farmer is insured | Weather-index insurance learned from real district harvests (HarvestStat Africa), tested on years it never saw |

## Page 0: My farm (`/farm`) — the whole year
Town, crop and acres in; out comes a 12-month timeline (buy and insure, plant, grow, harvest and sell) and four steps:
the fertilizer to buy, the **fertilizer insurance bundle** (if the season fails, the fertilizer money comes back), the
planting window, live season monitoring, and **sell or store + where to sell**. It also has a USSD phone simulator.

**Where to sell** (`/api/where_to_sell`, also on page 1): nearby markets for the crop ranked by what the farmer keeps after
transport (default $0.12 per tonne-km, roads ≈ 1.3 × straight line, $5 per trip; all editable).

**USSD** (`POST /api/ussd`, Africa's Talking format): dial `*384#` → 1 prices / 2 plan my season / 3 insurance / 4 language
(English, Kiswahili, Français) → crop → town → answer. Answers longer than one screen (182 characters) are cut and sent in
full by SMS through the gateway.

## More services
| Page | What it does | How |
|---|---|---|
| **Leaf doctor** (`/diagnose`) | Photo of a sick leaf → disease, how sure, what to do (English, Kiswahili, French) | **CNN** (MobileNetV3-Small, ImageNet pre-training, transfer learning) on PlantVillage (maize, potato, tomato) + **iCassava 2019** field photos from Makerere University. Accuracy reported separately for field photos. `disease/` |
| **Alerts** (`/admin`) | Farmers register once (web, cooperative, or SMS `JIUNGE MAHINDI KITALE` / `JOIN MAIZE KANO`; `STOP` to end) and get SMS: planting window opens, season failing, harvest → sell or store and where | Daily alert engine (`service/alerts.py`), each alert once per season, in the farmer's language. Africa's Talking SMS (`AT_USERNAME`, `AT_API_KEY`; free sandbox) or an outbox. Any date can be simulated for demos. |
| **Cooperatives** (`/coop`) | Buy together (bulk fertilizer order with discount, each member's share), protect together (bundle), sell together (one truck vs alone), SMS all members | Season planner for the group's total acres; where-to-sell with full-truck transport cost |
| **Early warning** (`/warning`) | Map of districts heading into a food crisis in 3–6 months, with the reasons | 45% live season loss (insurance model) + 25% price forecast + 15% recent price rise + 15% history. An indicator, not IPC. Daily. |
| **Satellite greenness** (optional) | Improves insurance with MODIS NDVI per district | `insure/ndvi_gee.js` (Google Earth Engine, free non-commercial) exports `data/ndvi_districts.csv`; insurance retrains with it automatically |

First start after this update downloads leaf photos (~0.5 GB PlantVillage subset + ~1 GB iCassava) and trains the CNN on
CPU (roughly 20–60 min). Set `DISEASE: "0"` to skip, or `DISEASE_CROPS: "maize,potato,tomato"` to skip the large cassava download.

## The problem
Smallholder farmers sell most of their harvest when everyone else does, so prices are at their lowest. A few months later prices rise, and traders who stored the crop earn the difference. Farmers can't see where prices are going, so they can't answer one question: *sell now, or store and sell later?*

## How it works

| Step | File | What it does |
|---|---|---|
| 0. Live data | `pipeline/fetch_wfp.py`, `pipeline/update.py` | Downloads the latest WFP food prices for every African country from the HDX API, and retrains only when WFP publishes new prices. |
| 1. Data + graph | `pipeline/prepare.py` | Reads WFP market-price CSVs, keeps African countries, converts prices to **USD/kg**, builds a monthly price table, and builds the **market graph**. Nodes are markets; edges link each market to its nearest markets and to all markets within 250 km. Each edge gets a **dynamic weight** every month (the rolling 12-month correlation of price changes). |
| 2a. Baselines | `model/baseline.py` | Naive, seasonal, and a **graph regression** that uses neighbours' price changes. Runs without PyTorch. |
| 2b. Deep learning | `model/train_gnn.py` | **Dynamic ST-GNN**: GATv2 graph attention at every month (edge features: distance, cross-border flag, monthly correlation), then a GRU over 12 months, then forecasts for 1/2/3 months for every market at once. Also trains the same network **without** the graph, to prove whether the graph helps. |
| 3. API | `api/app.py` | Flask: forecasts, dynamic edge weights, sell/store advice and the SMS endpoint |
| 4. Dashboard | `frontend/index.html` | Animated map of the market network over time, market forecast chart, advice, model comparison, SMS simulator |

**Which model answers?** For each crop the backend automatically uses the model with the lowest 3-month forecast error on the test period (marked ★ in the Model menu). On real WFP data the graph neural network wins for sorghum and millet. For rice, the simpler graph regression is as good, so it is used. Note that this choice is made on the same test period the errors are reported on, so the winner's error is slightly optimistic.

**Over-fitting control:** the networks stop training when the validation loss (the last 12 training months) has not improved for 8 epochs (`PATIENCE`), and the weights from the best epoch are kept. They also use dropout 0.2 and L2 weight decay. The validation loss rising after its best epoch, as seen in the logs, is exactly what early stopping catches.

**Sell/store rule:** STORE for *h* months if (forecast price change − (storage cost + storage loss) × *h*) ≥ 3%. Otherwise SELL NOW. Costs can be changed on the dashboard.

## Live data (automatic)
With `LIVE=1` (the default) the system needs no manual download:
1. At start-up it asks the **HDX API** (data.humdata.org) for the WFP "Food Prices" dataset of every African country and saves them as `data/raw/LIVE_wfp_food_prices_<iso3>.csv`.
2. It cleans the data: it drops "National Average" rows, flour and other processed products, and markets with no price in the last 6 months, converts bags (for example 90 KG) to per-kg prices, and ends the series at the latest month most markets have reported.
3. It rebuilds the market graph and retrains every model.
4. While running, the backend **re-checks WFP every 24 hours** (`REFRESH_HOURS`). If WFP published new prices, it retrains in the background while the old model keeps serving, then switches over. The dashboard reloads by itself.

The pill at the top of the dashboard shows **LIVE** (with the latest WFP month and when it was last checked), **RETRAINING**, or **DEMO DATA**. "Check now" asks WFP immediately.

If HDX can't be reached and there is no data yet, it falls back to **clearly labelled DEMO data**. You can also drop CSVs you downloaded yourself into `data/raw/`.

## Run with Docker (recommended)
```bash
docker compose up --build -d
```
Open **http://localhost:8000**. Settings are in `docker-compose.yml`: `CROPS` (comma-separated list), `COUNTRIES` (empty means all of Africa, or for example `KEN,UGA,TZA`), `REFRESH_HOURS`, `LIVE`.
The first live run downloads Africa's price history and trains, which takes several minutes. After that, restarts are fast unless WFP published new prices.

## Run without Docker (Ubuntu)
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gnn.txt      # for the graph neural network (optional)
./run_all.sh
```

## Step by step
```bash
python pipeline/update.py                         # = fetch + all crops (what the backend does by itself)
# or one step at a time:
python pipeline/fetch_wfp.py                      # live WFP download (or put CSVs in data/raw)
python pipeline/prepare.py --commodity maize --pricetype Retail
cd model && python baseline.py --slug maize_retail
python train_gnn.py --slug maize_retail --epochs 60
cd .. && python api/app.py
```

## API
`GET /api/status` (LIVE/DEMO, data month, last and next check), `POST /api/refresh` (check WFP now), plus `/api/meta`, `/api/series`, `/api/market` and `/api/sms`.

## Page 2: Plan the season (closing the yield gap)
African farms produce about half or less of what their land could, mostly because of too little and wrong fertilizer and poorly timed planting. Open **http://localhost:8000/plan**, click a field on the map (or type a town, or use your phone's location), pick a crop and farm size, and get:

- **When to plant:** the planting window that gives the best yield in a typical year, worked out from **35 years of daily rainfall** (NASA POWER) and an FAO soil-water balance. Places with two rainy seasons get two windows.
- **Which fertilizer, how much and when,** in half-bags per acre: the most profitable bundle of urea, DAP, NPK 17-17-17 and MOP that is also **safe**. It must return at least $2 for every $1, and lose money in no more than 1 in 4 of the past 30 seasons.
- **The yield gap:** best possible yield, what the rain allows, with the plan, and without fertilizer.
- **The money:** fertilizer cost, extra harvest value (using **SokoMkulima's own price forecast** at the nearest market), profit, and $ back per $1.
- **The soil:** pH, organic carbon, P and K from the **iSDAsoil 30 m soil map of Africa**, with advice on lime (acid soils), manure (low carbon) and legumes (no urea).

| Step | File | Method |
|---|---|---|
| Soil | `agro/soil.py` | iSDAsoil 30 m maps read directly from the cloud (rasterio), with SoilGrids as fallback. A soil test typed in by the user always wins. |
| Rain & planting dates | `agro/weather.py` | NASA POWER daily rain and temperature, Hargreaves ET0, a daily soil-water bucket, WRSI, FAO Ky yield response. |
| Nutrient response | `agro/quefts.py` | **QUEFTS** (Janssen et al. 1990), the standard model behind fertilizer advice in Africa. A line-by-line Python port of the reference code (Rquefts by R. Hijmans), **verified to give identical results** to the original C++ on test cases. |
| Economics | `agro/advisor.py` | Every bundle in half-bag steps is scored on profit in the median season, then the best candidates are checked against all 35 past seasons. |
| Crop settings | `agro/crops.py` | Potential yield, season length, water sensitivity and fertilizer products and prices, all editable. |

By SMS: `MBOLEA MAHINDI KITALE`, `PLAN MAIZE KANO`, `ENGRAIS MIL NIAMEY`, `ADUBO MILHO MAPUTO`, `سماد ذرة Khartoum` and so on. The reply gives the planting window, fertilizer per acre, expected yield with and without fertilizer, the cost and the gain.

**Honest notes:**
- This part is agronomic science (QUEFTS plus the FAO water balance) and economics, **not deep learning**. Deep learning needs thousands of field results (yield at a known place with a known fertilizer rate), and no open dataset of that size covers Africa. With field-trial data, QUEFTS's soil supply can be calibrated (Rquefts' `revSupply` method).
- iSDAsoil gives Mehlich-3 P. QUEFTS needs Olsen P, so Olsen ≈ 0.5 × Mehlich-3 is used (an approximation). Teff uses wheat parameters.
- Potential yields, fertilizer prices and the farm-gate share (75% of retail) are typical values. Fertilizer prices vary a lot, so enter your local ones.
- The first plan for a new place downloads 35 years of weather and the soil values (10–30 s). After that it's cached.

## Page 3: Insure the crop (closing the climate-risk gap)
Almost no African smallholder is insured, so one drought wipes out a family's savings, and the fear of it stops
farmers investing in fertilizer and seed. **Index insurance** pays automatically when the weather shows the season
failed, with no inspectors and no claim forms. Open **http://localhost:8000/insure**.

**Ground truth:** HarvestStat Africa v1.2 (Lee et al. 2025, *Scientific Data*, MIT licence): official harvest statistics for
**738 districts in 23 countries since 1991** (maize, sorghum, millet; ~41,000 district-seasons, bundled in `insure/data/`).
A **bad season** is a harvest at least 30% below the district's trend.

**Weather:** NASA POWER daily rain and temperature for each district (≈650 places, downloaded once and cached), turned into
18 ten-day steps per season: rain, rain vs normal, crop water need, water supply ratio (FAO bucket), heat, heavy-rain days.

| Model | What it is |
|---|---|
| `rain_index` | Classic index insurance: pays on the season's rainfall deficit |
| `wrsi_index` | FAO Water Requirement Satisfaction Index, loss = Ky × (1 − WRSI) |
| `ridge` | Linear model on season weather summaries |
| `deep_cnn_gru` | **Deep learning:** 1-D convolution + GRU over the season's ten-day steps (sklearn MLP if PyTorch is missing) |

**Honest testing:** train ≤ 2012, validation 2013–2015, **test 2016 onward** (never seen). For each model the page shows
how many real bad seasons it would have paid and how many payouts went to seasons that were not bad. Those two
errors are the **basis risk** of index insurance. The model with the best *validation* score prices the policies.

**Policy per district:** sum insured = expected harvest (district trend) × SokoMkulima farm-gate price. When the model's
predicted loss passes the trigger, it pays that share of the insured value. Premium = average past payout × 1.3. A district is
**not offered** cover when the premium would exceed 20% or the index caught fewer than 40% of its bad seasons, because
selling cover that does not protect would hurt farmers. **Live:** during a season, the page and SMS (`BIMA MAHINDI
KITALE`, `INSURANCE MAIZE KANO`, `ASSURANCE MIL NIAMEY`...) show rain so far vs normal, the expected loss, and whether a
payout is likely.

| File | Step |
|---|---|
| `insure/prepare_labels.py` | HarvestStat → labels + district outlines (`--download` for the latest release) |
| `insure/features.py` | Season weather → ten-day sequence + summaries |
| `insure/train.py` | Four models, time-split test, triggers, pricing → `outputs/insure_model.json` |
| `insure/live.py` | This season, live, for one district |

The first run downloads rainfall for ~650 places (roughly 20–60 minutes, in the background; the other pages keep working).
`INSURE=0` skips it, and `INSURE_MAX_CELLS=100` does a quick trial run.
It's a research prototype: a real product needs an insurer, a regulator, mobile-money integration and local validation.

## Crops
Ten storable staples by default (`CROPS` in `docker-compose.yml`): maize, sorghum, millet, rice, wheat, teff, beans, cowpeas, groundnuts and cassava. Each crop gets **its own market graph and its own models**, because sorghum markets don't move like rice markets. A crop with too few markets in the WFP data is skipped and named in the status message. Perishables such as tomatoes are left out on purpose: you can't store them for three months, so "sell or store" doesn't apply.

Training order: first the fast models for **every** crop (the dashboard works after a few minutes), then the graph neural network crop by crop. The crop menu shows "(deep model training…)" until a crop's network is ready.

## SMS in 11 languages
`POST /api/sms` with `{"text": "..."}` returns the reply a farmer would receive, **in the language they wrote in**:

| Language | Example |
|---|---|
| English | PRICE MAIZE KANO |
| Français | PRIX MAÏS BAMAKO |
| Português | PREÇO MILHO MAPUTO |
| العربية | سعر ذرة رفيعة Khartoum |
| Kiswahili | BEI MAHINDI ELDORET |
| አማርኛ | ዋጋ ጤፍ Addis Ababa |
| Hausa | FARASHI MASARA KANO |
| Soomaali | QIIME BARIIS BAIDOA |
| Yorùbá | IYE IRESI LAGOS |
| Kinyarwanda | IGICIRO IBIGORI KIGALI |
| isiZulu | INTENGO UMMBILA ... |

The crop words and reply texts are in `api/languages.py`. Replies in Amharic, Hausa, Somali, Yoruba, Kinyarwanda and Zulu **should be checked by native speakers** before real use. Market names are matched in Latin letters, as WFP spells them. If a market has no prices for the crop asked, the reply says which crops it does have. To connect to real phones, point an SMS gateway's webhook (for example Africa's Talking) at this endpoint.

## Honest limits
- "Live" means "as fresh as WFP publishes". WFP collects market prices monthly and publishes them with a lag of a few weeks, so the newest month shown is usually 1–2 months behind today.
- Forecasts are only as good as the WFP data. Markets with gaps or short histories are less reliable.
- Prices are compared in USD. Exchange-rate swings can move USD prices even when local prices don't change.
- The advice ignores the farmer's cash needs. Some farmers must sell now regardless.

## Tests
```bash
sudo docker compose exec web python -m unittest discover -s tests -v
```
These check that QUEFTS matches the original C++, that SMS is understood in all 11 languages, that the insurance features
behave sensibly, and that every page and API endpoint answers.

## Sharing safely
The app runs on **gunicorn** (a production web server). Before sharing it online (for example with ngrok), set
`APP_USER` and `APP_PASSWORD` in `docker-compose.yml`: every page then asks for a login. `/api/sms` stays open for SMS gateways.

## Put it on GitHub
1. Create an empty repository on github.com/new (for example `sokomkulima`), with no README.
2. In the project folder:
```bash
git init
git add .
git commit -m "SokoMkulima: price forecasts, season planner, crop insurance"
git branch -M main
git remote add origin https://github.com/<your-username>/sokomkulima.git
git push -u origin main
```
When asked for a password, paste a GitHub **personal access token** (classic, `repo` scope), never your password.
Downloaded data and trained models (`data/`, `outputs/`) are not uploaded. A friend who clones the repo runs
`docker compose up --build -d` and the system downloads and trains everything by itself.
