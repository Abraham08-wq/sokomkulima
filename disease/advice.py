"""
What to do, per diagnosis, in English, Kiswahili and French. Practical, low-cost steps first; chemicals only as
"a recommended product from your agro-dealer or extension officer" (products and rules differ by country).
Please have local extension staff review these texts before real use.
"""
A = {
 "maize_gray_leaf_spot": dict(
  en=("Maize: grey leaf spot (fungus)", "Rotate with beans or another non-cereal crop and bury or remove old maize stalks. Plant tolerant varieties next season. If it spreads to upper leaves before tasselling, a recommended fungicide can pay."),
  sw=("Mahindi: madoa ya kijivu (kuvu)", "Badilisha zao (k.m. maharagwe) na ondoa mabua ya zamani. Panda aina zinazostahimili msimu ujao. Ikienea kwenye majani ya juu kabla ya kutoa maua, dawa ya kuvu inayopendekezwa inaweza kulipa."),
  fr=("Maïs : cercosporiose (champignon)", "Faites une rotation avec le haricot et enfouissez les vieilles tiges. Semez des variétés tolérantes. Si elle atteint les feuilles du haut avant la floraison, un fongicide recommandé peut être rentable.")),
 "maize_common_rust": dict(
  en=("Maize: common rust (fungus)", "Usually minor. Plant resistant varieties and plant early. Spray a recommended fungicide only if young plants are badly covered."),
  sw=("Mahindi: kutu (kuvu)", "Kwa kawaida si hatari. Panda aina zinazostahimili na panda mapema. Nyunyizia dawa tu kama mimea michanga imeathirika sana."),
  fr=("Maïs : rouille commune (champignon)", "En général sans gravité. Semez tôt des variétés résistantes. Traitez seulement si les jeunes plants sont très atteints.")),
 "maize_northern_leaf_blight": dict(
  en=("Maize: northern leaf blight (fungus)", "Remove or bury infected crop residue, rotate crops and use resistant varieties. If long grey lesions spread before tasselling, use a recommended fungicide."),
  sw=("Mahindi: ukungu wa majani (kuvu)", "Ondoa au fukia mabaki ya mazao, badilisha mazao na tumia aina zinazostahimili. Madoa marefu yakienea kabla ya kutoa maua, tumia dawa inayopendekezwa."),
  fr=("Maïs : helminthosporiose (champignon)", "Enfouissez les résidus, faites une rotation et utilisez des variétés résistantes. Si les longues taches grises s'étendent avant la floraison, utilisez un fongicide recommandé.")),
 "maize_healthy": dict(
  en=("Maize: healthy leaf", "No disease seen. Keep weeding and check the crop every week, including for fall armyworm (ragged holes and sawdust-like droppings in the leaf funnel)."),
  sw=("Mahindi: jani zima", "Hakuna ugonjwa. Endelea kupalilia na kukagua kila wiki, pamoja na viwavijeshi vamizi (matundu na kinyesi kama unga wa mbao katikati ya majani)."),
  fr=("Maïs : feuille saine", "Pas de maladie. Continuez à désherber et inspectez chaque semaine, y compris pour la chenille légionnaire (trous et sciure dans le cornet).")),
 "potato_early_blight": dict(
  en=("Potato: early blight (fungus)", "Remove lower infected leaves, avoid wetting the leaves, and keep potatoes and tomatoes off this field for 2-3 years. Use a recommended fungicide if it spreads."),
  sw=("Viazi: ukungu wa mapema (kuvu)", "Ondoa majani ya chini yaliyoathirika, usinyunyizie maji juu ya majani, na usipande viazi au nyanya hapa kwa miaka 2-3. Tumia dawa inayopendekezwa ukienea."),
  fr=("Pomme de terre : alternariose (champignon)", "Retirez les feuilles basses atteintes, n'arrosez pas le feuillage et évitez pomme de terre et tomate ici pendant 2-3 ans. Fongicide recommandé si ça s'étend.")),
 "potato_late_blight": dict(
  en=("Potato: late blight (spreads fast in cool, wet weather)", "Act now: remove and destroy infected plants, protect the rest with a recommended fungicide, and do not keep tubers from this field as seed."),
  sw=("Viazi: baridi kali / blight (huenea haraka hali ya ubaridi na unyevu)", "Chukua hatua sasa: ng'oa na choma mimea iliyoathirika, linda iliyobaki kwa dawa inayopendekezwa, na usitumie viazi vya shamba hili kama mbegu."),
  fr=("Pomme de terre : mildiou (se propage vite par temps frais et humide)", "Agissez tout de suite : arrachez et détruisez les plants atteints, protégez les autres avec un fongicide recommandé, ne gardez pas ces tubercules comme semence.")),
 "potato_healthy": dict(
  en=("Potato: healthy leaf", "No disease seen. In cool, wet weeks check every few days for late blight (dark, water-soaked patches)."),
  sw=("Viazi: jani zima", "Hakuna ugonjwa. Wiki za baridi na mvua, kagua kila baada ya siku chache kwa blight (madoa meusi yenye maji)."),
  fr=("Pomme de terre : feuille saine", "Pas de maladie. Par temps frais et humide, surveillez le mildiou (taches sombres et humides) tous les quelques jours.")),
 "tomato_bacterial_spot": dict(
  en=("Tomato: bacterial spot", "Use clean seed, remove infected leaves, avoid working in the crop when wet, and rotate. Copper-based products from an agro-dealer can slow it."),
  sw=("Nyanya: madoa ya bakteria", "Tumia mbegu safi, ondoa majani yaliyoathirika, usifanye kazi shambani likiwa na unyevu, na badilisha mazao. Dawa za shaba zinaweza kupunguza."),
  fr=("Tomate : tache bactérienne", "Semences saines, retirez les feuilles atteintes, n'entrez pas dans la culture mouillée, faites une rotation. Les produits à base de cuivre la freinent.")),
 "tomato_early_blight": dict(
  en=("Tomato: early blight (fungus)", "Remove lower infected leaves, mulch the soil, stake plants for air flow, rotate crops. Use a recommended fungicide if it spreads."),
  sw=("Nyanya: ukungu wa mapema (kuvu)", "Ondoa majani ya chini yaliyoathirika, weka matandazo, simika mimea ipate hewa, badilisha mazao. Tumia dawa inayopendekezwa ukienea."),
  fr=("Tomate : alternariose (champignon)", "Retirez les feuilles basses, paillez, tuteurez pour aérer, faites une rotation. Fongicide recommandé si ça s'étend.")),
 "tomato_late_blight": dict(
  en=("Tomato: late blight (spreads fast)", "Remove and destroy infected plants immediately and protect healthy plants with a recommended fungicide. Do not plant tomatoes next to potatoes."),
  sw=("Nyanya: blight (huenea haraka)", "Ng'oa na choma mimea iliyoathirika mara moja na linda mimea mizima kwa dawa inayopendekezwa. Usipande nyanya karibu na viazi."),
  fr=("Tomate : mildiou (se propage vite)", "Détruisez tout de suite les plants atteints et protégez les autres avec un fongicide recommandé. Pas de tomates à côté des pommes de terre.")),
 "tomato_leaf_mold": dict(
  en=("Tomato: leaf mould (fungus)", "Mostly in humid greenhouses: open them for air flow, avoid wetting leaves, remove infected leaves."),
  sw=("Nyanya: ukungu wa majani (kuvu)", "Hasa kwenye vitalu nyumba vyenye unyevu: fungua hewa ipite, usilowanishe majani, ondoa majani yaliyoathirika."),
  fr=("Tomate : cladosporiose (champignon)", "Surtout en serre humide : aérez, ne mouillez pas le feuillage, retirez les feuilles atteintes.")),
 "tomato_septoria_leaf_spot": dict(
  en=("Tomato: septoria leaf spot (fungus)", "Remove spotted lower leaves, mulch, water at the base, rotate. Use a recommended fungicide if it climbs the plant."),
  sw=("Nyanya: madoa ya septoria (kuvu)", "Ondoa majani ya chini yenye madoa, weka matandazo, mwagilia chini, badilisha mazao. Tumia dawa inayopendekezwa ikipanda juu."),
  fr=("Tomate : septoriose (champignon)", "Retirez les feuilles basses tachées, paillez, arrosez au pied, rotation. Fongicide recommandé si elle monte.")),
 "tomato_spider_mites": dict(
  en=("Tomato: spider mites (pest)", "Worse in hot dry weather: spray leaves (undersides) with water, remove badly infested leaves, use a recommended miticide only if needed, and protect natural enemies."),
  sw=("Nyanya: utitiri buibui (mdudu)", "Huzidi wakati wa joto na ukame: nyunyizia maji chini ya majani, ondoa majani yaliyoathirika sana, tumia dawa ya utitiri tu ikihitajika."),
  fr=("Tomate : acariens (ravageur)", "Pire par temps chaud et sec : aspergez d'eau le dessous des feuilles, retirez les feuilles très atteintes, acaricide recommandé seulement si nécessaire.")),
 "tomato_target_spot": dict(
  en=("Tomato: target spot (fungus)", "Remove infected leaves, improve air flow, avoid overhead watering, rotate crops; a recommended fungicide if severe."),
  sw=("Nyanya: madoa ya shabaha (kuvu)", "Ondoa majani yaliyoathirika, ruhusu hewa ipite, usimwagilie juu ya majani, badilisha mazao; dawa inayopendekezwa ikizidi."),
  fr=("Tomate : corynesporiose (champignon)", "Retirez les feuilles atteintes, aérez, n'arrosez pas par-dessus, rotation ; fongicide recommandé si grave.")),
 "tomato_yellow_leaf_curl_virus": dict(
  en=("Tomato: yellow leaf curl virus (spread by whiteflies)", "No cure: pull out and destroy infected plants early, control whiteflies (yellow sticky traps, nets on seedlings), use tolerant varieties."),
  sw=("Nyanya: virusi vya kukunjika majani (huenezwa na inzi weupe)", "Hakuna tiba: ng'oa na haribu mimea iliyoathirika mapema, dhibiti inzi weupe (mitego ya njano, nyavu kwa miche), tumia aina zinazostahimili."),
  fr=("Tomate : virus de l'enroulement jaune (transmis par l'aleurode)", "Pas de traitement : arrachez tôt les plants atteints, luttez contre l'aleurode (pièges jaunes, filets), variétés tolérantes.")),
 "tomato_mosaic_virus": dict(
  en=("Tomato: mosaic virus", "No cure: remove infected plants, wash hands and tools (it spreads by touch and tobacco), use clean seed and resistant varieties."),
  sw=("Nyanya: virusi vya mosaic", "Hakuna tiba: ondoa mimea iliyoathirika, osha mikono na zana (huenea kwa kugusa na tumbaku), tumia mbegu safi na aina zinazostahimili."),
  fr=("Tomate : virus de la mosaïque", "Pas de traitement : retirez les plants atteints, lavez mains et outils (se transmet au toucher et par le tabac), semences saines et variétés résistantes.")),
 "tomato_healthy": dict(
  en=("Tomato: healthy leaf", "No disease seen. Keep checking twice a week; tomatoes catch diseases fast in wet weather."),
  sw=("Nyanya: jani zima", "Hakuna ugonjwa. Endelea kukagua mara mbili kwa wiki; nyanya hupata magonjwa haraka wakati wa mvua."),
  fr=("Tomate : feuille saine", "Pas de maladie. Inspectez deux fois par semaine ; la tomate tombe vite malade par temps humide.")),
 "cassava_mosaic": dict(
  en=("Cassava: mosaic disease (virus, spread by whiteflies and cuttings)", "Pull out infected plants early. Plant only clean cuttings of resistant varieties (ask your extension officer for certified ones). Never take cuttings from sick plants."),
  sw=("Muhogo: batobato (virusi, huenezwa na inzi weupe na vipandikizi)", "Ng'oa mimea iliyoathirika mapema. Panda vipandikizi safi vya aina zinazostahimili (uliza afisa ugani). Usichukue vipandikizi kutoka mimea migonjwa."),
  fr=("Manioc : mosaïque (virus, transmis par l'aleurode et les boutures)", "Arrachez tôt les plants atteints. Plantez des boutures saines de variétés résistantes. Ne prenez jamais de boutures sur des plants malades.")),
 "cassava_brown_streak": dict(
  en=("Cassava: brown streak disease (virus)", "It rots the roots: harvest affected fields early, destroy infected plants, and replant only with certified clean cuttings of tolerant varieties."),
  sw=("Muhogo: michirizi ya kahawia (virusi)", "Huozesha mizizi: vuna mashamba yaliyoathirika mapema, haribu mimea iliyoathirika, na panda tena kwa vipandikizi safi vilivyothibitishwa."),
  fr=("Manioc : striure brune (virus)", "Elle pourrit les racines : récoltez tôt, détruisez les plants atteints et replantez avec des boutures saines certifiées de variétés tolérantes.")),
 "cassava_bacterial_blight": dict(
  en=("Cassava: bacterial blight", "Use clean cuttings, remove and burn infected plants, clean tools, and rotate with other crops."),
  sw=("Muhogo: ukungu wa bakteria", "Tumia vipandikizi safi, ng'oa na choma mimea iliyoathirika, safisha zana, na badilisha mazao."),
  fr=("Manioc : bactériose", "Boutures saines, arrachez et brûlez les plants atteints, nettoyez les outils, faites une rotation.")),
 "cassava_green_mite": dict(
  en=("Cassava: green mite (pest)", "Worse in the dry season. Plant tolerant varieties and plant at the start of the rains; avoid broad insecticides that kill the mite's natural enemies."),
  sw=("Muhogo: utitiri kijani (mdudu)", "Huzidi wakati wa kiangazi. Panda aina zinazostahimili mwanzo wa mvua; epuka viuatilifu vinavyoua maadui wa asili wa utitiri."),
  fr=("Manioc : acarien vert (ravageur)", "Pire en saison sèche. Variétés tolérantes, plantation en début de pluies ; évitez les insecticides qui tuent ses ennemis naturels.")),
 "cassava_healthy": dict(
  en=("Cassava: healthy leaf", "No disease seen. Keep your clean plants as the source of next season's cuttings."),
  sw=("Muhogo: jani zima", "Hakuna ugonjwa. Tumia mimea hii safi kupata vipandikizi vya msimu ujao."),
  fr=("Manioc : feuille saine", "Pas de maladie. Gardez ces plants sains pour les boutures de la prochaine saison.")),
}
UNSURE = dict(en="Not sure from this photo. Take a sharp photo of one leaf in daylight, filling the frame, or show the plant to your extension officer.",
              sw="Sina uhakika kutoka picha hii. Piga picha wazi ya jani moja mchana likijaza picha, au mwonyeshe afisa ugani.",
              fr="Pas sûr avec cette photo. Prenez une photo nette d'une seule feuille en plein jour, ou montrez la plante à votre agent de vulgarisation.")


def advice(label, lang="en"):
    a = A.get(label, {})
    name, todo = a.get(lang) or a.get("en") or (label, "")
    return {"name": name, "todo": todo}
