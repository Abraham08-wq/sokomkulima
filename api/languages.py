"""
SMS LANGUAGES — the farmer can text in the language they use every day.

Covers the African Union's working languages (English, French, Portuguese, Arabic, Swahili) plus
Amharic, Hausa, Somali, Yoruba, Kinyarwanda and Zulu: together the official or main market language
of almost every country in the WFP data.

How a message is understood:
  1. The text is normalised (lower case, accents removed: "MAÏS" -> "mais", "feijão" -> "feijao").
  2. Crop words and "price" words are matched in every language (longest phrase first, so the Arabic
     "ذرة رفيعة" = sorghum wins over "ذرة" = maize).
  3. The language with the most matched words is the reply language. Arabic or Ethiopic script decides
     ties. Market names are matched in Latin letters, as WFP spells them.

NOTE: replies in Amharic, Hausa, Somali, Yoruba, Kinyarwanda and Zulu should be reviewed by native
speakers before real use. Edit the TEXT and CROP_WORDS tables below; no other code needs to change.
"""
import re
import unicodedata

LANGS = {"en": "English", "fr": "Français", "pt": "Português", "ar": "العربية", "sw": "Kiswahili",
         "am": "አማርኛ", "ha": "Hausa", "so": "Soomaali", "yo": "Yorùbá", "rw": "Kinyarwanda", "zu": "isiZulu"}

# Which language to suggest for a country's markets (used for the example messages on the dashboard).
COUNTRY_LANG = {**{c: "fr" for c in "MLI NER BFA SEN TCD CMR BEN TGO CIV GIN COD COG GAB CAF BDI MDG DJI COM".split()},
                **{c: "pt" for c in "MOZ AGO GNB CPV STP".split()},
                **{c: "ar" for c in "SDN EGY LBY DZA TUN MAR MRT".split()},
                **{c: "sw" for c in "KEN TZA UGA".split()},
                "ETH": "am", "NGA": "ha", "SOM": "so", "RWA": "rw", "ZAF": "zu",
                **{c: "en" for c in "GHA LBR SLE GMB ZMB ZWE MWI SSD LSO SWZ NAM BWA".split()}}

# "price" words: tell us the farmer is asking for a price, and in which language
PRICE_WORDS = {"en": ["price", "prices"], "fr": ["prix"], "pt": ["preco", "precos"], "ar": ["سعر", "اسعار", "أسعار"],
               "sw": ["bei", "soko"], "am": ["ዋጋ"], "ha": ["farashi", "kudin"], "so": ["qiime", "qiimaha"],
               "yo": ["iye", "owo"], "rw": ["igiciro", "ibiciro"], "zu": ["intengo", "inani"]}

# crop -> language -> words. The FIRST word is the name used in replies.
# The crop keys are the WFP commodity keywords (same as CROP in docker-compose).
CROP_WORDS = {
    "maize":     {"en": ["maize", "corn"], "fr": ["maïs", "mais"], "pt": ["milho"], "ar": ["ذرة", "ذرة شامية"],
                  "sw": ["mahindi"], "am": ["በቆሎ"], "ha": ["masara"], "so": ["galley"], "yo": ["agbado"],
                  "rw": ["ibigori"], "zu": ["ummbila"]},
    "sorghum":   {"en": ["sorghum"], "fr": ["sorgho"], "pt": ["sorgo", "mapira"], "ar": ["ذرة رفيعة"],
                  "sw": ["mtama"], "am": ["ማሽላ"], "ha": ["dawa"], "so": ["masago"], "yo": ["oka baba"],
                  "rw": ["amasaka"], "zu": ["amabele"]},
    "millet":    {"en": ["millet"], "fr": ["mil"], "pt": ["milheto", "mexoeira"], "ar": ["دخن"],
                  "sw": ["uwele", "ulezi"], "am": ["ዳጉሳ"], "ha": ["gero"], "yo": ["jero"], "rw": ["uburo"]},
    "rice":      {"en": ["rice"], "fr": ["riz"], "pt": ["arroz"], "ar": ["أرز", "ارز", "رز"], "sw": ["mchele", "mpunga"],
                  "am": ["ሩዝ"], "ha": ["shinkafa"], "so": ["bariis"], "yo": ["iresi"], "rw": ["umuceri"],
                  "zu": ["irayisi"]},
    "wheat":     {"en": ["wheat"], "fr": ["blé", "ble"], "pt": ["trigo"], "ar": ["قمح"], "sw": ["ngano"], "am": ["ስንዴ"],
                  "ha": ["alkama"], "so": ["sarreen"], "yo": ["alikama"], "rw": ["ingano"], "zu": ["ukolweni"]},
    "teff":      {"en": ["teff", "tef"], "am": ["ጤፍ"]},
    "beans":     {"en": ["beans", "bean"], "fr": ["haricot", "haricots"], "pt": ["feijão", "feijao"], "ar": ["فاصوليا"],
                  "sw": ["maharagwe", "maharage"], "am": ["ባቄላ"], "so": ["digir"], "rw": ["ibishyimbo"],
                  "zu": ["ubhontshisi"]},
    "cowpea":    {"en": ["cowpeas", "cowpea"], "fr": ["niébé", "niebe"], "pt": ["feijão-nhemba", "feijao nhemba"],
                  "sw": ["kunde"], "ha": ["wake"], "yo": ["ewa"]},
    "groundnut": {"en": ["groundnuts", "groundnut", "peanuts"], "fr": ["arachide", "arachides"], "pt": ["amendoim"],
                  "ar": ["فول سوداني"], "sw": ["karanga"], "am": ["ኦቾሎኒ"], "ha": ["gyada"], "so": ["looska"],
                  "yo": ["epa"], "rw": ["ubunyobwa"], "zu": ["amakinati"]},
    "cassava":   {"en": ["cassava"], "fr": ["manioc"], "pt": ["mandioca"], "ar": ["كسافا"], "sw": ["muhogo", "mihogo"],
                  "ha": ["rogo"], "yo": ["gbaguda"], "rw": ["imyumbati"]},
}
SIBLINGS = {"beans": "cowpea", "cowpea": "beans"}       # if one has no data, answer with the other

TEXT = {
    "en": dict(now="{crop} {market}: ${cur}/kg now. In 3 months: ~${f3}/kg.", advice="Advice",
               store="STORE for {n} month(s) (about +{g}% after storage costs)",
               sell="SELL NOW (price not expected to rise enough to cover storage)",
               nomarket="Market not found. Example: {ex}", nocrop="No {crop} prices for {market}. Available there: {alt}.",
               nodata="No {crop} prices yet. Available crops: {alt}."),
    "fr": dict(now="{crop} {market} : {cur} $/kg aujourd'hui. Dans 3 mois : ~{f3} $/kg.", advice="Conseil",
               store="STOCKER {n} mois (environ +{g}% après coûts de stockage)",
               sell="VENDRE MAINTENANT (la hausse prévue ne couvre pas le stockage)",
               nomarket="Marché introuvable. Exemple : {ex}", nocrop="Pas de prix {crop} à {market}. Disponible : {alt}.",
               nodata="Pas encore de prix {crop}. Cultures disponibles : {alt}."),
    "pt": dict(now="{crop} {market}: {cur} $/kg hoje. Em 3 meses: ~{f3} $/kg.", advice="Conselho",
               store="ARMAZENAR {n} mês(es) (cerca de +{g}% após custos de armazenamento)",
               sell="VENDER AGORA (a subida prevista não cobre o armazenamento)",
               nomarket="Mercado não encontrado. Exemplo: {ex}", nocrop="Sem preços de {crop} em {market}. Disponível: {alt}.",
               nodata="Ainda sem preços de {crop}. Culturas disponíveis: {alt}."),
    "ar": dict(now="{crop} {market}: {cur} دولار/كغ الآن. بعد 3 أشهر: ~{f3} دولار/كغ.", advice="النصيحة",
               store="خزّن لمدة {n} شهر (ربح حوالي +{g}% بعد تكاليف التخزين)",
               sell="بِع الآن (لا يُتوقع أن يرتفع السعر بما يكفي لتغطية التخزين)",
               nomarket="لم يتم العثور على السوق. مثال: {ex}", nocrop="لا توجد أسعار {crop} في {market}. المتوفر: {alt}.",
               nodata="لا توجد أسعار {crop} بعد. المحاصيل المتوفرة: {alt}."),
    "sw": dict(now="{crop} {market}: ${cur}/kg leo. Miezi 3: ~${f3}/kg.", advice="Ushauri",
               store="HIFADHI miezi {n} (faida ~{g}% baada ya gharama za kuhifadhi)",
               sell="UZA SASA (bei haitarajiwi kupanda vya kutosha)",
               nomarket="Soko halikupatikana. Mfano: {ex}", nocrop="Hakuna bei za {crop} {market}. Zilizopo: {alt}.",
               nodata="Bado hakuna bei za {crop}. Mazao yaliyopo: {alt}."),
    "am": dict(now="{crop} {market}: ዛሬ ${cur}/ኪግ። በ3 ወር: ~${f3}/ኪግ።", advice="ምክር",
               store="ለ{n} ወር አከማች (ከማከማቻ ወጪ በኋላ ~+{g}% ትርፍ)",
               sell="አሁን ሽጥ (ዋጋው የማከማቻ ወጪን ለመሸፈን በቂ አይጨምርም)",
               nomarket="ገበያው አልተገኘም። ምሳሌ: {ex}", nocrop="በ{market} የ{crop} ዋጋ የለም። ያሉት: {alt}።",
               nodata="የ{crop} ዋጋ ገና የለም። ያሉ ሰብሎች: {alt}።"),
    "ha": dict(now="{crop} {market}: ${cur}/kg yau. Nan da wata 3: ~${f3}/kg.", advice="Shawara",
               store="AJIYE na wata {n} (riba kusan +{g}% bayan kudin ajiya)",
               sell="SAYAR YANZU (ba a tsammanin farashi zai tashi isa ya rufe kudin ajiya)",
               nomarket="Ba a samu kasuwar ba. Misali: {ex}", nocrop="Babu farashin {crop} a {market}. Akwai: {alt}.",
               nodata="Babu farashin {crop} tukuna. Amfanin gona da ake da su: {alt}."),
    "so": dict(now="{crop} {market}: ${cur}/kg maanta. 3 bilood kadib: ~${f3}/kg.", advice="Talo",
               store="KAYDI {n} bilood (faa'iido ~+{g}% kadib kharashka kaydinta)",
               sell="HADDA IIBI (qiimaha lagama filayo inuu kor u kaco ku filan)",
               nomarket="Suuqa lama helin. Tusaale: {ex}", nocrop="Qiimaha {crop} kama jiro {market}. Waxaa jira: {alt}.",
               nodata="Qiimaha {crop} weli ma jiro. Dalagyada jira: {alt}."),
    "yo": dict(now="{crop} {market}: ${cur}/kg lónìí. Ní oṣù 3: ~${f3}/kg.", advice="Ìmọ̀ràn",
               store="TỌ́JÚ fún oṣù {n} (èrè tó tó +{g}% lẹ́yìn owó ìtọ́jú)",
               sell="TÀÁ NÍSINSÌN YÌÍ (owó kò ní gòkè tó láti bo owó ìtọ́jú)",
               nomarket="A kò rí ọjà náà. Àpẹẹrẹ: {ex}", nocrop="Kò sí iye {crop} ní {market}. Èyí tó wà: {alt}.",
               nodata="Kò sí iye {crop} síbẹ̀. Àwọn irúgbìn tó wà: {alt}."),
    "rw": dict(now="{crop} {market}: ${cur}/kg uyu munsi. Mu mezi 3: ~${f3}/kg.", advice="Inama",
               store="BIKA amezi {n} (inyungu ~+{g}% nyuma y'ikiguzi cyo kubika)",
               sell="GURISHA UBU (igiciro ntikizazamuka bihagije)",
               nomarket="Isoko ntiryabonetse. Urugero: {ex}", nocrop="Nta biciro bya {crop} i {market}. Ibihari: {alt}.",
               nodata="Nta biciro bya {crop} biraboneka. Ibihingwa bihari: {alt}."),
    "zu": dict(now="{crop} {market}: ${cur}/kg namuhla. Ezinyangeni ezi-3: ~${f3}/kg.", advice="Iseluleko",
               store="GCINA izinyanga ezingu-{n} (inzuzo ~+{g}% ngemuva kwezindleko zokugcina)",
               sell="THENGISA MANJE (intengo ayilindelekile ukukhuphuka ngokwanele)",
               nomarket="Imakethe ayitholakalanga. Isibonelo: {ex}", nocrop="Azikho izintengo ze-{crop} e-{market}. Ezikhona: {alt}.",
               nodata="Azikho izintengo ze-{crop} okwamanje. Izitshalo ezikhona: {alt}."),
}


def norm(s):
    """lower case, remove accents and Arabic vowel marks, keep letters of every script."""
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s'-]|_|\d", " ", s).replace("-", " ")
    return " ".join(s.split())


# (normalised phrase, kind, value, language), longest first
_PHRASES = sorted([(norm(w), "crop", c, l) for c, by in CROP_WORDS.items() for l, ws in by.items() for w in ws] +
                  [(norm(w), "price", None, l) for l, ws in PRICE_WORDS.items() for w in ws],
                  key=lambda x: -len(x[0]))


def parse(text):
    """-> (crop or None, language, leftover normalised words for market matching)"""
    t = f" {norm(text)} "
    votes, crop = {}, None
    for ph, kind, val, lang in _PHRASES:
        if f" {ph} " in t:
            t = t.replace(f" {ph} ", " ", 1)
            votes[lang] = votes.get(lang, 0) + (2 if kind == "price" else 1)
            if kind == "crop" and crop is None:
                crop = val
    if re.search(r"[؀-ۿ]", text): votes["ar"] = votes.get("ar", 0) + 0.5
    if re.search(r"[ሀ-፿]", text): votes["am"] = votes.get("am", 0) + 0.5
    lang = max(votes, key=votes.get) if votes else "en"
    return crop, lang, t.split()


def crop_name(crop, lang):
    by = CROP_WORDS.get(crop, {})
    return (by.get(lang) or by.get("en") or [crop])[0]


def example(lang, crop, market):
    return f"{PRICE_WORDS[lang][0].upper()} {crop_name(crop, lang).upper()} {market.upper()}"


def reply(lang, crop, market, cur, f3, action, n=0, g=0.0):
    T = TEXT[lang]
    head = T["now"].format(crop=crop_name(crop, lang).capitalize(), market=market, cur=f"{cur:.2f}", f3=f"{f3:.2f}")
    act = T["store"].format(n=n, g=g) if action == "STORE" else T["sell"]
    return f"{head} {T['advice']}: {act}."


# ---------------- season plan by SMS ("MBOLEA MAHINDI KITALE", "PLAN MAIZE KITALE") ----------------
PLAN_WORDS = {"en": ["plan", "fertilizer", "fertiliser", "plant"], "fr": ["engrais", "semis"], "pt": ["adubo", "plantio"],
              "ar": ["سماد"], "sw": ["mbolea", "panda"], "am": ["ማዳበሪያ"], "ha": ["taki"], "so": ["bacrin"],
              "yo": ["ajile"], "rw": ["ifumbire"], "zu": ["umanyolo"]}
_PLAN = {norm(w): l for l, ws in PLAN_WORDS.items() for w in ws}

PLAN_TEXT = {
    "en": dict(plan="{crop} {place}: plant {window}. Per acre: {products}. Expected {y} kg/acre (without fertilizer {y0}). Cost ${c}, extra harvest worth ~${v}.",
               none="{crop} {place}: plant {window}. Fertilizer does not pay here at current prices; expected {y0} kg/acre. Use manure.",
               dry="Rain at {place} is too unreliable for {crop}. Choose a drought-tolerant crop."),
    "fr": dict(plan="{crop} {place} : semer {window}. Par acre : {products}. Récolte attendue {y} kg/acre (sans engrais {y0}). Coût {c} $, récolte en plus ~{v} $.",
               none="{crop} {place} : semer {window}. L'engrais n'est pas rentable ici aux prix actuels ; récolte attendue {y0} kg/acre. Utilisez du fumier.",
               dry="Pluie trop irrégulière à {place} pour {crop}. Choisissez une culture résistante à la sécheresse."),
    "pt": dict(plan="{crop} {place}: semear {window}. Por acre: {products}. Colheita esperada {y} kg/acre (sem adubo {y0}). Custo {c} $, colheita extra ~{v} $.",
               none="{crop} {place}: semear {window}. O adubo não compensa aos preços atuais; colheita esperada {y0} kg/acre. Use estrume.",
               dry="Chuva demasiado irregular em {place} para {crop}. Escolha uma cultura resistente à seca."),
    "ar": dict(plan="{crop} {place}: ازرع {window}. لكل إيكر: {products}. المحصول المتوقع {y} كغ/إيكر (بدون سماد {y0}). التكلفة {c} دولار، قيمة المحصول الإضافي ~{v} دولار.",
               none="{crop} {place}: ازرع {window}. السماد غير مربح هنا بالأسعار الحالية؛ المحصول المتوقع {y0} كغ/إيكر. استخدم السماد العضوي.",
               dry="الأمطار في {place} غير منتظمة لزراعة {crop}. اختر محصولاً مقاوماً للجفاف."),
    "sw": dict(plan="{crop} {place}: panda {window}. Kwa ekari: {products}. Mavuno yanayotarajiwa {y} kg/ekari (bila mbolea {y0}). Gharama ${c}, mavuno ya ziada ~${v}.",
               none="{crop} {place}: panda {window}. Mbolea hailipi hapa kwa bei za sasa; mavuno yanayotarajiwa {y0} kg/ekari. Tumia samadi.",
               dry="Mvua {place} haitoshi kwa {crop}. Chagua zao linalostahimili ukame."),
    "am": dict(plan="{crop} {place}: ዝሩ {window}። በኤከር: {products}። የሚጠበቅ ምርት {y} ኪግ/ኤከር (ያለ ማዳበሪያ {y0})። ወጪ ${c}፣ ተጨማሪ ምርት ~${v}።",
               none="{crop} {place}: ዝሩ {window}። በአሁኑ ዋጋ ማዳበሪያ እዚህ አያዋጣም፤ የሚጠበቅ ምርት {y0} ኪግ/ኤከር። ፍግ ይጠቀሙ።",
               dry="በ{place} ዝናብ ለ{crop} በቂ አይደለም። ድርቅን የሚቋቋም ሰብል ይምረጡ።"),
    "ha": dict(plan="{crop} {place}: shuka {window}. Kowace eka: {products}. Ana sa ran girbi {y} kg/eka (ba tare da taki ba {y0}). Kudi ${c}, karin girbi ~${v}.",
               none="{crop} {place}: shuka {window}. Taki ba ya da riba a nan a farashin yanzu; ana sa ran {y0} kg/eka. Yi amfani da takin dabbobi.",
               dry="Ruwan sama a {place} bai isa ga {crop} ba. Zabi amfanin gona mai jure fari."),
    "so": dict(plan="{crop} {place}: abuur {window}. Acre kasta: {products}. Wax-soo-saarka la filayo {y} kg/acre (bacrin la'aan {y0}). Kharash ${c}, dalag dheeraad ah ~${v}.",
               none="{crop} {place}: abuur {window}. Bacrintu halkan faa'iido ma leh qiimaha hadda; la filayo {y0} kg/acre. Isticmaal digo.",
               dry="Roobka {place} kuma filna {crop}. Dooro dalag u adkaysta abaarta."),
    "yo": dict(plan="{crop} {place}: gbìn {window}. Fún acre kan: {products}. Ìkórè tí a retí {y} kg/acre (láìsí ajílẹ̀ {y0}). Iye owó ${c}, ìkórè àfikún ~${v}.",
               none="{crop} {place}: gbìn {window}. Ajílẹ̀ kò ní èrè níbí ní iye owó báyìí; ìkórè tí a retí {y0} kg/acre. Lo ìgbẹ́ ẹran.",
               dry="Òjò ní {place} kò tó fún {crop}. Yan irúgbìn tó lè fara da ọ̀dá."),
    "rw": dict(plan="{crop} {place}: tera {window}. Kuri acre: {products}. Umusaruro witezwe {y} kg/acre (nta fumbire {y0}). Igiciro ${c}, umusaruro w'inyongera ~${v}.",
               none="{crop} {place}: tera {window}. Ifumbire ntiyunguka hano ku biciro by'ubu; umusaruro witezwe {y0} kg/acre. Koresha ifumbire y'imborera.",
               dry="Imvura i {place} ntihagije kuri {crop}. Hitamo igihingwa cyihanganira amapfa."),
    "zu": dict(plan="{crop} {place}: tshala {window}. Nge-acre: {products}. Isivuno esilindelekile {y} kg/acre (ngaphandle komanyolo {y0}). Izindleko ${c}, isivuno esengeziwe ~${v}.",
               none="{crop} {place}: tshala {window}. Umanyolo awukhokheli lapha ngamanani amanje; isivuno esilindelekile {y0} kg/acre. Sebenzisa umquba.",
               dry="Imvula e-{place} ayanele i-{crop}. Khetha isitshalo esimelana nesomiso."),
}


def is_plan(text):
    """-> language code if the message asks for a season plan, else None."""
    t = f" {norm(text)} "
    return next((l for w, l in _PLAN.items() if f" {w} " in t), None)


def plan_reply(lang, crop, place, p):
    T = PLAN_TEXT[lang]; name = crop_name(crop, lang).capitalize()
    if not p.get("ok"):
        return T["dry"].format(crop=crop_name(crop, lang), place=place)
    s, g, pl = p["season"], p["yield_gap"], p["plan"]
    r5 = lambda v: int(5 * round(v / 5))
    prods = " + ".join(f"{r5(x['kg_acre'])} kg {x['name'].split()[0]}" for x in pl["products"])
    k = "plan" if pl["recommended"] else "none"
    return T[k].format(crop=name, place=place, window=s["window_text"], products=prods,
                       y=r5(g["with_plan"]["kg_acre"]), y0=r5(g["without"]["kg_acre"]),
                       c=f"{pl['cost_farm']:.0f}", v=f"{pl['extra_value_farm']:.0f}")


# ---------------- crop insurance by SMS ("BIMA MAHINDI KITALE", "INSURANCE MAIZE KANO") ----------------
INSURE_WORDS = {"en": ["insurance", "insure", "cover"], "fr": ["assurance"], "pt": ["seguro"], "ar": ["تأمين"],
                "sw": ["bima"], "am": ["መድን", "ኢንሹራንስ"], "ha": ["inshora"], "so": ["caymis"], "yo": ["iseduro"],
                "rw": ["ubwishingizi"], "zu": ["umshwalense", "inshuwalense"]}
_INS = {norm(w): l for l, ws in INSURE_WORDS.items() for w in ws}

INSURE_TEXT = {
    "en": dict(offer="{crop} insurance, {place}: premium ${p} per acre covers ${si} of harvest. Pays automatically by mobile money if satellite weather shows a loss over {t}%. {now}",
               run="This season so far: rain {r}% of normal, expected loss {l}%{pay}.", pay=" - payout likely: ${amt}/acre",
               up="Next season starts {month}: buy before planting.", no="{crop} insurance is not offered for {place}: the weather index does not predict harvests well enough there."),
    "fr": dict(offer="Assurance {crop}, {place} : prime {p} $ par acre, couvre {si} $ de récolte. Paiement automatique par mobile money si la météo satellite montre une perte de plus de {t} %. {now}",
               run="Cette saison : pluie {r} % de la normale, perte prévue {l} %{pay}.", pay=" - paiement probable : {amt} $/acre",
               up="Prochaine saison en {month} : souscrivez avant de semer.", no="Assurance {crop} non proposée à {place} : l'indice météo n'y prédit pas assez bien les récoltes."),
    "pt": dict(offer="Seguro de {crop}, {place}: prémio {p} $ por acre, cobre {si} $ de colheita. Paga automaticamente por dinheiro móvel se o clima por satélite mostrar perda acima de {t}%. {now}",
               run="Esta época: chuva {r}% do normal, perda prevista {l}%{pay}.", pay=" - pagamento provável: {amt} $/acre",
               up="Próxima época começa em {month}: compre antes de semear.", no="Seguro de {crop} não disponível em {place}: o índice climático não prevê bem as colheitas aí."),
    "ar": dict(offer="تأمين {crop}، {place}: القسط {p} دولار لكل إيكر ويغطي {si} دولار من المحصول. يُدفع تلقائياً عبر الهاتف إذا أظهر الطقس بالأقمار الصناعية خسارة تزيد على {t}%. {now}",
               run="هذا الموسم: المطر {r}% من المعدل، الخسارة المتوقعة {l}%{pay}.", pay=" - تعويض مرجح: {amt} دولار/إيكر",
               up="الموسم القادم يبدأ في {month}: اشترِ قبل الزراعة.", no="تأمين {crop} غير متاح في {place}: مؤشر الطقس لا يتنبأ بالمحصول جيداً هناك."),
    "sw": dict(offer="Bima ya {crop}, {place}: ada ${p} kwa ekari, inalinda mavuno ya ${si}. Inalipa yenyewe kwa pesa ya simu hali ya hewa ya setilaiti ikionyesha hasara zaidi ya {t}%. {now}",
               run="Msimu huu: mvua {r}% ya kawaida, hasara inayotarajiwa {l}%{pay}.", pay=" - malipo yanatarajiwa: ${amt}/ekari",
               up="Msimu ujao unaanza {month}: nunua kabla ya kupanda.", no="Bima ya {crop} haipatikani {place}: kipimo cha hali ya hewa hakitabiri mavuno vizuri huko."),
    "am": dict(offer="የ{crop} መድን፣ {place}: ክፍያ ${p} በኤከር፣ ${si} ምርት ይሸፍናል። የሳተላይት የአየር ሁኔታ ከ{t}% በላይ ኪሳራ ካሳየ በሞባይል ገንዘብ በራሱ ይከፍላል። {now}",
               run="ይህ ወቅት: ዝናብ ከመደበኛው {r}%፣ የሚጠበቅ ኪሳራ {l}%{pay}።", pay=" - ክፍያ ሊኖር ይችላል: ${amt}/ኤከር",
               up="ቀጣዩ ወቅት በ{month} ይጀምራል: ከመዝራትዎ በፊት ይግዙ።", no="የ{crop} መድን በ{place} የለም: የአየር ሁኔታ መለኪያው እዚያ ምርትን በደንብ አይተነብይም።"),
    "ha": dict(offer="Inshorar {crop}, {place}: kudin inshora ${p} kowace eka, yana kare girbi na ${si}. Yana biya kai tsaye ta kudin waya idan yanayin tauraron dan adam ya nuna asara fiye da {t}%. {now}",
               run="Wannan damina: ruwan sama {r}% na yadda aka saba, ana sa ran asara {l}%{pay}.", pay=" - mai yiwuwa a biya: ${amt}/eka",
               up="Damina mai zuwa za ta fara a {month}: saya kafin shuka.", no="Babu inshorar {crop} a {place}: ma'aunin yanayi ba ya hasashen girbi da kyau a can."),
    "so": dict(offer="Caymiska {crop}, {place}: lacag ${p} acre kasta, wuxuu daboolaa ${si} dalag ah. Si toos ah ayuu u bixiyaa lacagta mobilka haddii cimilada dayax-gacmeedku muujiso khasaare ka badan {t}%. {now}",
               run="Xilligan: roobka {r}% caadiga, khasaaraha la filayo {l}%{pay}.", pay=" - lacag bixin la filayo: ${amt}/acre",
               up="Xilliga xiga wuxuu bilaabmayaa {month}: iibso ka hor beerashada.", no="Caymiska {crop} lagama helo {place}: tusaha cimiladu si fiican uma saadaaliyo dalagga halkaas."),
    "yo": dict(offer="Ìdáàbòbò {crop}, {place}: owó ìdáàbòbò ${p} fún acre kan, ó bo ìkórè ${si}. Ó ń sanwó fúnra rẹ̀ lórí fóònù tí ojú ọjọ́ sátẹ́láìtì bá fi àdánù tó ju {t}% hàn. {now}",
               run="Àsìkò yìí: òjò {r}% ti ìwọ̀n déédé, àdánù tí a retí {l}%{pay}.", pay=" - ìsanwó ṣeé ṣe: ${amt}/acre",
               up="Àsìkò tó ń bọ̀ bẹ̀rẹ̀ ní {month}: ra kí o tó gbìn.", no="Kò sí ìdáàbòbò {crop} ní {place}: àmì ojú ọjọ́ kò sọ àsọtẹ́lẹ̀ ìkórè dáadáa níbẹ̀."),
    "rw": dict(offer="Ubwishingizi bwa {crop}, {place}: umusanzu ${p} kuri acre, burinda umusaruro wa ${si}. Bwishyura ubwabwo kuri mobile money iyo ikirere cy'icyogajuru cyerekanye igihombo kirenze {t}%. {now}",
               run="Iki gihembwe: imvura {r}% y'isanzwe, igihombo giteganyijwe {l}%{pay}.", pay=" - kwishyurwa birashoboka: ${amt}/acre",
               up="Igihembwe gikurikira gitangira {month}: gura mbere yo gutera.", no="Ubwishingizi bwa {crop} ntibuboneka i {place}: igipimo cy'ikirere ntigiteganya neza umusaruro aho."),
    "zu": dict(offer="Umshwalense we-{crop}, {place}: inkokhelo ${p} nge-acre, ivikela isivuno esingu-${si}. Ikhokha ngokuzenzakalelayo ngemali yeselula uma isimo sezulu sesathelayithi sibonisa ukulahlekelwa okungaphezu kuka-{t}%. {now}",
               run="Le sizini: imvula {r}% yejwayelekile, ukulahlekelwa okulindelekile {l}%{pay}.", pay=" - inkokhelo kungenzeka: ${amt}/acre",
               up="Isizini elandelayo iqala ngo-{month}: thenga ngaphambi kokutshala.", no="Umshwalense we-{crop} awutholakali e-{place}: inkomba yesimo sezulu ayibikezeli kahle isivuno lapho."),
}


def is_insure(text):
    t = f" {norm(text)} "
    return next((l for w, l in _INS.items() if f" {w} " in t), None)


def insure_reply(lang, crop, place, q):
    """q = /api/insure/policy result (per 1 acre)."""
    T = INSURE_TEXT[lang]; name = crop_name(crop, lang)
    if not q.get("offered"):
        out = T["no"].format(crop=name, place=place)
        return out[0].upper() + out[1:]
    now, live = "", q.get("season_now") or {}
    if live.get("state") == "running":
        r = round(100 * live["rain_so_far"] / max(live["normal_so_far"], 1))
        pay = T["pay"].format(amt=f"{live['payout'] * q['sum_insured']:.0f}") if live.get("would_pay") else ""
        now = T["run"].format(r=r, l=round(100 * max(live["pred_loss"], 0)), pay=pay)
    elif live.get("state") == "upcoming" and live.get("starts"):
        import datetime as _d
        now = T["up"].format(month=_d.date.fromisoformat(live["starts"]).strftime("%b %Y"))
    out = T["offer"].format(crop=name, place=place, p=f"{q['premium']:.2f}", si=f"{q['sum_insured']:.0f}",
                            t=round(100 * (q["bad_threshold"] if "probability" in q.get("trigger_on", "") else q["trigger"])),
                            now=now).strip()
    return out[0].upper() + out[1:]


# ---------------- registration for automatic alerts: "JIUNGE MAHINDI KITALE" / "JOIN MAIZE KITALE" / "STOP" ----------------
JOIN_WORDS = {"en": ["join", "register", "subscribe"], "fr": ["inscrire", "inscription"], "pt": ["registar", "registo"],
              "ar": ["اشترك"], "sw": ["jiunge", "sajili"], "am": ["ተመዝገብ"], "ha": ["rijista"], "so": ["isdiiwaangeli"],
              "yo": ["forukosile"], "rw": ["iyandikishe", "kwiyandikisha"], "zu": ["bhalisa"]}
STOP_WORDS = {"en": ["stop", "unsubscribe"], "fr": ["arret", "arreter"], "pt": ["parar"], "ar": ["توقف"], "sw": ["acha", "sitisha"],
              "am": ["አቁም"], "ha": ["daina"], "so": ["jooji"], "yo": ["duro"], "rw": ["hagarika"], "zu": ["yeka"]}
_JOIN = {norm(w): l for l, ws in JOIN_WORDS.items() for w in ws}
_STOP = {norm(w): l for l, ws in STOP_WORDS.items() for w in ws}

JOIN_TEXT = {
    "en": dict(ok="You are registered for {crop} alerts near {place}. We will SMS you when to plant, if the season fails, and when to sell. Reply STOP to end.",
               stop="You will no longer receive SokoMkulima alerts."),
    "fr": dict(ok="Vous êtes inscrit aux alertes {crop} près de {place}. Nous vous enverrons un SMS pour semer, si la saison échoue et pour vendre. Répondez STOP pour arrêter.",
               stop="Vous ne recevrez plus d'alertes SokoMkulima."),
    "pt": dict(ok="Está registado para alertas de {crop} perto de {place}. Enviaremos SMS para semear, se a época falhar e para vender. Responda PARAR para terminar.",
               stop="Não receberá mais alertas SokoMkulima."),
    "ar": dict(ok="تم تسجيلك لتنبيهات {crop} قرب {place}. سنرسل لك رسالة عند موعد الزراعة، وإذا فشل الموسم، وعند البيع. أرسل توقف للإلغاء.",
               stop="لن تتلقى تنبيهات SokoMkulima بعد الآن."),
    "sw": dict(ok="Umesajiliwa kupata ujumbe wa {crop} karibu na {place}. Tutakutumia SMS wakati wa kupanda, msimu ukiharibika, na wakati wa kuuza. Jibu ACHA kusitisha.",
               stop="Hutapokea tena ujumbe wa SokoMkulima."),
    "am": dict(ok="ለ{crop} ማሳወቂያዎች በ{place} አቅራቢያ ተመዝግበዋል። የመዝሪያ ጊዜ፣ ወቅቱ ቢበላሽ እና የሽያጭ ጊዜ በSMS እንልክልዎታለን። ለማቆም STOP ይላኩ።",
               stop="ከአሁን በኋላ የSokoMkulima ማሳወቂያዎች አይደርስዎትም።"),
    "ha": dict(ok="An yi muku rajista don sanarwar {crop} kusa da {place}. Za mu aiko SMS lokacin shuka, idan damina ta lalace, da lokacin sayarwa. Aika STOP don dakatarwa.",
               stop="Ba za ku ƙara samun sanarwar SokoMkulima ba."),
    "so": dict(ok="Waxaa laguu diiwaangeliyay digniinaha {crop} ee u dhow {place}. Waxaan kuu soo diri doonnaa SMS waqtiga beerashada, haddii xilligu fashilmo, iyo waqtiga iibka. U dir STOP si aad u joojiso.",
               stop="Mar dambe ma heli doontid digniinaha SokoMkulima."),
    "yo": dict(ok="A ti forúkọ rẹ sílẹ̀ fún ìkìlọ̀ {crop} nítòsí {place}. A ó fi SMS ránṣẹ́ nígbà gbígbìn, tí àsìkò bá bàjẹ́, àti nígbà títà. Fi STOP ránṣẹ́ láti dúró.",
               stop="O kò ní gba ìkìlọ̀ SokoMkulima mọ́."),
    "rw": dict(ok="Wiyandikishije ku butumwa bwa {crop} hafi ya {place}. Tuzakoherereza SMS igihe cyo gutera, niba igihembwe cyangiritse, n'igihe cyo kugurisha. Ohereza STOP guhagarika.",
               stop="Ntuzongera kubona ubutumwa bwa SokoMkulima."),
    "zu": dict(ok="Ubhalisiwe ukuthola izexwayiso ze-{crop} eduze kwe-{place}. Sizokuthumela i-SMS uma kufanele utshale, uma isizini yehluleka, nokuthi uthengise nini. Thumela u-STOP ukuze uyeke.",
               stop="Ngeke usathola izexwayiso ze-SokoMkulima."),
}


def is_join(text):
    t = f" {norm(text)} "
    return next((l for w, l in _JOIN.items() if f" {w} " in t), None)


def is_stop(text):
    t = f" {norm(text)} "
    return next((l for w, l in _STOP.items() if f" {w} " in t), None)
