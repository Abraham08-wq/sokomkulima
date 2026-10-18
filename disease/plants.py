"""
What the universal leaf doctor can NAME and which SYMPTOMS it looks for.

PLANTS: crops, vegetables, fruit, trees, herbs and weeds common on African farms (key -> English, Kiswahili).
  The CLIP model compares the photo with the text "a photo of a <plant> leaf" for every plant here, so adding a plant
  is one line. Plants outside this list are named as the closest one in it (with low confidence), or via Pl@ntNet
  (50,000 species) when PLANTNET_API_KEY is set.
SPECIALIST: plants that also have a trained disease model (disease/train.py) with named diseases.
NOT_SICK: groups that are not a disease (natural ageing / autumn colour).
SYMPTOMS: what a sick leaf looks like, for ANY plant. Each is matched against the photo with a few text prompts.
PV_SYMPTOM: PlantVillage / iCassava labels -> symptom group, used to train and test the health check on 15 plants.
"""

PLANTS = {
    # cereals
    "maize": ("maize (corn)", "mahindi"), "sorghum": ("sorghum", "mtama"), "pearl_millet": ("pearl millet", "uwele"),
    "finger_millet": ("finger millet", "ulezi"), "rice": ("rice", "mpunga"), "wheat": ("wheat", "ngano"),
    "teff": ("teff", "tefu"), "barley": ("barley", "shayiri"), "oats": ("oats", "shayiri ya oti"),
    # legumes
    "bean": ("common bean", "maharagwe"), "cowpea": ("cowpea", "kunde"), "pigeon_pea": ("pigeon pea", "mbaazi"),
    "groundnut": ("groundnut (peanut)", "karanga"), "soybean": ("soybean", "soya"), "green_gram": ("green gram (mung bean)", "choroko"),
    "chickpea": ("chickpea", "dengu"), "lablab": ("lablab bean", "fiwi"), "bambara": ("bambara groundnut", "njugu mawe"),
    "pea": ("garden pea", "njegere"),
    # roots and tubers
    "cassava": ("cassava", "muhogo"), "sweet_potato": ("sweet potato", "viazi vitamu"), "potato": ("potato", "viazi mviringo"),
    "yam": ("yam", "viazi vikuu"), "taro": ("taro (cocoyam)", "magimbi"),
    # vegetables
    "tomato": ("tomato", "nyanya"), "onion": ("onion", "kitunguu"), "cabbage": ("cabbage", "kabichi"),
    "kale": ("kale (sukuma wiki)", "sukuma wiki"), "spinach": ("spinach", "spinachi"), "amaranth": ("amaranth", "mchicha"),
    "nightshade": ("African nightshade", "managu"), "spider_plant": ("spider plant (cleome)", "mgagani"),
    "okra": ("okra", "bamia"), "eggplant": ("eggplant (brinjal)", "biringanya"), "pepper": ("bell pepper (capsicum)", "pilipili hoho"),
    "chili": ("chili pepper", "pilipili"), "carrot": ("carrot", "karoti"), "pumpkin": ("pumpkin", "malenge"),
    "squash": ("squash (zucchini)", "maboga"), "watermelon": ("watermelon", "tikiti maji"), "cucumber": ("cucumber", "tango"),
    "lettuce": ("lettuce", "saladi"), "garlic": ("garlic", "kitunguu saumu"), "beetroot": ("beetroot", "kiazi sukari"),
    "jute_mallow": ("jute mallow", "mlenda"),
    # fruit
    "banana": ("banana or plantain", "ndizi"), "mango": ("mango", "embe"), "avocado": ("avocado", "parachichi"),
    "orange": ("orange or other citrus", "chungwa"), "lemon": ("lemon", "limau"), "papaya": ("papaya (pawpaw)", "papai"),
    "pineapple": ("pineapple", "nanasi"), "passion_fruit": ("passion fruit", "pasheni"), "guava": ("guava", "pera"),
    "jackfruit": ("jackfruit", "fenesi"), "coconut": ("coconut palm", "mnazi"), "apple": ("apple", "tufaha"),
    "grape": ("grape vine", "mzabibu"), "strawberry": ("strawberry", "stroberi"), "peach": ("peach", "pichi"),
    "cherry": ("cherry", "cheri"), "blueberry": ("blueberry", "bluberi"), "raspberry": ("raspberry", "rasiberi"),
    "date_palm": ("date palm", "mtende"), "tree_tomato": ("tree tomato (tamarillo)", "nyanya mti"),
    # cash crops
    "coffee": ("coffee", "kahawa"), "tea": ("tea", "chai"), "cocoa": ("cocoa", "kakao"), "cotton": ("cotton", "pamba"),
    "sugarcane": ("sugarcane", "miwa"), "tobacco": ("tobacco", "tumbaku"), "sunflower": ("sunflower", "alizeti"),
    "sesame": ("sesame (simsim)", "ufuta"), "oil_palm": ("oil palm", "mchikichi"), "sisal": ("sisal", "mkonge"),
    "cashew": ("cashew", "mkorosho"), "macadamia": ("macadamia", "makadamia"), "vanilla": ("vanilla", "vanila"),
    "clove": ("clove tree", "mkarafuu"), "pyrethrum": ("pyrethrum", "pareto"), "rubber": ("rubber tree", "mpira"),
    # herbs, fodder and useful trees
    "coriander": ("coriander (dhania)", "dhania"), "basil": ("basil", "mrehani"), "mint": ("mint", "nanaa"),
    "moringa": ("moringa", "mlonge"), "neem": ("neem", "mwarobaini"), "aloe": ("aloe vera", "shubiri"),
    "lemongrass": ("lemongrass", "mchaichai"), "napier": ("napier grass", "nyasi ya napia"), "lucerne": ("lucerne (alfalfa)", "lusina"),
    "eucalyptus": ("eucalyptus", "mkaratusi"), "acacia": ("acacia", "mgunga"), "baobab": ("baobab", "mbuyu"),
    "grevillea": ("grevillea", "mkabilishemu"), "cypress": ("cypress", "msaiprasi"), "pine": ("pine", "msonobari"),
    "bamboo": ("bamboo", "mianzi"), "calliandra": ("calliandra", "kaliandra"), "leucaena": ("leucaena", "lusina mti"),
    # flowers and garden plants
    "rose": ("rose", "waridi"), "hibiscus": ("hibiscus", "haibiskasi"), "bougainvillea": ("bougainvillea", "bogenvilia"),
    "marigold": ("marigold", "marigoldi"),
    "maple": ("maple", "mpingo wa mepu"), "oak": ("oak", "mwaloni"), "ficus": ("fig or ficus", "mtini"),
    "jacaranda": ("jacaranda", "mjakaranda"), "croton": ("croton", "krotoni"), "ivy": ("ivy", "mtambazi"),
    "fern": ("fern", "kangaga"), "cactus": ("cactus", "mpungate"), "palm": ("ornamental palm", "mchikichi wa mapambo"),
    # weeds
    "striga": ("striga (witchweed)", "kiduha"), "lantana": ("lantana", "mvamia"), "parthenium": ("parthenium weed", "gugu karoti"),
    "black_jack": ("black jack (bidens)", "kishonange"), "couch_grass": ("couch grass", "sangari"), "grass": ("grass", "nyasi"),
}

# other names the image model knows better (used together with the main name)
ALIASES = {
    "cassava": ["manioc", "yuca", "tapioca plant", "cassava shrub with hand-shaped leaves"],
    "maize": ["corn", "corn stalk"], "sorghum": ["milo", "guinea corn"], "groundnut": ["peanut"],
    "kale": ["collard greens"], "amaranth": ["pigweed", "callaloo"], "cowpea": ["black-eyed pea"],
    "pigeon_pea": ["gandule"], "taro": ["cocoyam", "elephant ear"], "banana": ["plantain", "banana tree"],
    "sweet_potato": ["sweet potato vine"], "papaya": ["pawpaw"], "eggplant": ["aubergine", "brinjal"],
    "pepper": ["capsicum", "bell pepper"], "chili": ["hot pepper"], "okra": ["lady's finger"],
    "sesame": ["simsim"], "napier": ["elephant grass"], "nightshade": ["black nightshade"],
    "coffee": ["coffee shrub"], "tea": ["tea bush"], "cocoa": ["cacao"], "mango": ["mango tree"],
    "bean": ["green bean", "kidney bean"], "pumpkin": ["pumpkin vine"],
}

# PlantVillage / iCassava species -> PLANTS key (used to test how well the plant is NAMED)
SPECIES_KEY = {"apple": "apple", "blueberry": "blueberry", "cherry": "cherry", "maize": "maize", "grape": "grape",
               "orange": "orange", "peach": "peach", "pepper": "pepper", "potato": "potato", "raspberry": "raspberry",
               "soybean": "soybean", "squash": "squash", "strawberry": "strawberry", "tomato": "tomato", "cassava": "cassava"}

GATE_PLANT = ["a close-up photo of a plant leaf", "a photo of leaves on a plant", "a photo of a diseased leaf",
              "a photo of a crop growing in a field", "a photo of a tree with leaves", "a photo of a houseplant",
              "a photo of a leaf on a plain background", "a photo of a weed"]
GATE_OTHER = ["a photo of a person", "a selfie", "a photo of an animal", "a photo of a dog or cat", "a photo of a cow or goat",
              "a document with text", "a screenshot of a phone", "a photo of a building", "a photo of a car or motorbike",
              "a photo of cooked food on a plate", "a photo of a room", "a photo of bare soil", "a photo of the sky",
              "a photo of fruit and vegetables in a market", "a photo of a bag of grain", "a photo of a tool or machine",
              "a black image", "a blurry photo"]

SYMPTOMS = {   # group -> prompts ({p} = plant name)
    "healthy": ["a photo of a healthy green {p} leaf", "a healthy {p} leaf with no disease", "a fresh, clean, green {p} leaf",
                "a healthy {p} plant with lush green leaves", "healthy green {p} leaves with rain drops on them",
                "healthy {p} plants growing in a field"],
    "leaf_spot": ["a {p} leaf with brown or black spots", "a {p} leaf with small round spots with yellow halos",
                  "a {p} leaf with many dark lesions"],
    "blight": ["a {p} leaf with large dead brown patches", "a {p} leaf with blight, dry brown dead tissue",
               "a {p} leaf with long grey or brown lesions"],
    "rust": ["a {p} leaf with orange rust pustules", "a {p} leaf covered in rusty orange-brown powder"],
    "powdery_mildew": ["a {p} leaf covered in white powdery mildew", "a {p} leaf with a white fungal coating"],
    "mosaic": ["a {p} leaf with a yellow and green mosaic pattern from a virus", "a mottled, distorted {p} leaf from a virus"],
    "yellowing": ["a yellowing {p} leaf", "a pale yellow {p} leaf with green veins", "a {p} leaf turning yellow from nutrient deficiency"],
    "leaf_curl": ["a curled and crinkled {p} leaf", "a {p} leaf curling upward and deformed"],
    "insect_damage": ["a {p} leaf with holes eaten by insects", "a {p} leaf chewed by caterpillars", "a {p} leaf with ragged eaten edges"],
    "pests": ["a {p} leaf with aphids or whiteflies on it", "a {p} leaf with tiny mites and fine webbing", "a {p} leaf with insects on it"],
    "wilting": ["a wilted, drooping {p} leaf", "a dry, scorched {p} leaf with burnt edges"],
    "old_leaf": ["an old {p} leaf changing colour naturally in autumn", "a fallen yellow, orange or red autumn leaf",
                 "a dry fallen {p} leaf from the ground"],
}

NOT_SICK = ("healthy", "old_leaf")

# label prefix -> group. PlantVillage folder names, then our iCassava labels
PV_SYMPTOM = {
    "Apple___Apple_scab": "leaf_spot", "Apple___Black_rot": "leaf_spot", "Apple___Cedar_apple_rust": "rust",
    "Cherry_(including_sour)___Powdery_mildew": "powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "leaf_spot", "Corn_(maize)___Common_rust_": "rust",
    "Corn_(maize)___Northern_Leaf_Blight": "blight",
    "Grape___Black_rot": "leaf_spot", "Grape___Esca_(Black_Measles)": "leaf_spot", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "blight",
    "Orange___Haunglongbing_(Citrus_greening)": "yellowing", "Peach___Bacterial_spot": "leaf_spot",
    "Pepper,_bell___Bacterial_spot": "leaf_spot", "Potato___Early_blight": "blight", "Potato___Late_blight": "blight",
    "Squash___Powdery_mildew": "powdery_mildew", "Strawberry___Leaf_scorch": "blight",
    "Tomato___Bacterial_spot": "leaf_spot", "Tomato___Early_blight": "blight", "Tomato___Late_blight": "blight",
    "Tomato___Leaf_Mold": "leaf_spot", "Tomato___Septoria_leaf_spot": "leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "pests", "Tomato___Target_Spot": "leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "leaf_curl", "Tomato___Tomato_mosaic_virus": "mosaic",
    "cassava_bacterial_blight": "blight", "cassava_brown_streak": "yellowing", "cassava_green_mite": "pests",
    "cassava_mosaic": "mosaic", "cassava_healthy": "healthy",
}
PV_SPECIES = {"Apple": "apple", "Blueberry": "blueberry", "Cherry_(including_sour)": "cherry", "Corn_(maize)": "maize",
              "Grape": "grape", "Orange": "orange", "Peach": "peach", "Pepper,_bell": "pepper", "Potato": "potato",
              "Raspberry": "raspberry", "Soybean": "soybean", "Squash": "squash", "Strawberry": "strawberry", "Tomato": "tomato"}


def pv_group(folder):
    """PlantVillage folder -> (species, symptom group)."""
    sp = PV_SPECIES.get(folder.split("___")[0])
    g = "healthy" if folder.endswith("___healthy") else PV_SYMPTOM.get(folder)
    return sp, g


def name(key, lang="en"):
    en, sw = PLANTS.get(key, (key.replace("_", " "), key.replace("_", " ")))
    return sw if lang == "sw" else en

# scientific names: the species model (BioCLIP) was trained with these, so it names plants far better with them
SCI = {
    "maize": "Zea mays", "sorghum": "Sorghum bicolor", "pearl_millet": "Pennisetum glaucum", "finger_millet": "Eleusine coracana",
    "rice": "Oryza sativa", "wheat": "Triticum aestivum", "teff": "Eragrostis tef", "barley": "Hordeum vulgare",
    "oats": "Avena sativa", "bean": "Phaseolus vulgaris", "cowpea": "Vigna unguiculata", "pigeon_pea": "Cajanus cajan",
    "groundnut": "Arachis hypogaea", "soybean": "Glycine max", "green_gram": "Vigna radiata", "chickpea": "Cicer arietinum",
    "lablab": "Lablab purpureus", "bambara": "Vigna subterranea", "pea": "Pisum sativum", "cassava": "Manihot esculenta",
    "sweet_potato": "Ipomoea batatas", "potato": "Solanum tuberosum", "yam": "Dioscorea alata", "taro": "Colocasia esculenta",
    "tomato": "Solanum lycopersicum", "onion": "Allium cepa", "cabbage": "Brassica oleracea var. capitata",
    "kale": "Brassica oleracea var. acephala", "spinach": "Spinacia oleracea", "amaranth": "Amaranthus hybridus",
    "nightshade": "Solanum scabrum", "spider_plant": "Cleome gynandra", "okra": "Abelmoschus esculentus",
    "eggplant": "Solanum melongena", "pepper": "Capsicum annuum", "chili": "Capsicum frutescens", "carrot": "Daucus carota",
    "pumpkin": "Cucurbita maxima", "squash": "Cucurbita pepo", "watermelon": "Citrullus lanatus", "cucumber": "Cucumis sativus",
    "lettuce": "Lactuca sativa", "garlic": "Allium sativum", "beetroot": "Beta vulgaris", "jute_mallow": "Corchorus olitorius",
    "banana": "Musa acuminata", "mango": "Mangifera indica", "avocado": "Persea americana", "orange": "Citrus sinensis",
    "lemon": "Citrus limon", "papaya": "Carica papaya", "pineapple": "Ananas comosus", "passion_fruit": "Passiflora edulis",
    "guava": "Psidium guajava", "jackfruit": "Artocarpus heterophyllus", "coconut": "Cocos nucifera", "apple": "Malus domestica",
    "grape": "Vitis vinifera", "strawberry": "Fragaria ananassa", "peach": "Prunus persica", "cherry": "Prunus avium",
    "blueberry": "Vaccinium corymbosum", "raspberry": "Rubus idaeus", "date_palm": "Phoenix dactylifera",
    "tree_tomato": "Solanum betaceum", "coffee": "Coffea arabica", "tea": "Camellia sinensis", "cocoa": "Theobroma cacao",
    "cotton": "Gossypium hirsutum", "sugarcane": "Saccharum officinarum", "tobacco": "Nicotiana tabacum",
    "sunflower": "Helianthus annuus", "sesame": "Sesamum indicum", "oil_palm": "Elaeis guineensis", "sisal": "Agave sisalana",
    "cashew": "Anacardium occidentale", "macadamia": "Macadamia integrifolia", "vanilla": "Vanilla planifolia",
    "clove": "Syzygium aromaticum", "pyrethrum": "Tanacetum cinerariifolium", "rubber": "Hevea brasiliensis",
    "coriander": "Coriandrum sativum", "basil": "Ocimum basilicum", "mint": "Mentha spicata", "moringa": "Moringa oleifera",
    "neem": "Azadirachta indica", "aloe": "Aloe vera", "lemongrass": "Cymbopogon citratus", "napier": "Pennisetum purpureum",
    "lucerne": "Medicago sativa", "eucalyptus": "Eucalyptus grandis", "acacia": "Vachellia tortilis",
    "baobab": "Adansonia digitata", "grevillea": "Grevillea robusta", "cypress": "Cupressus lusitanica", "pine": "Pinus patula",
    "bamboo": "Bambusa vulgaris", "calliandra": "Calliandra calothyrsus", "leucaena": "Leucaena leucocephala",
    "rose": "Rosa", "hibiscus": "Hibiscus rosa-sinensis", "bougainvillea": "Bougainvillea spectabilis",
    "marigold": "Tagetes erecta", "maple": "Acer", "oak": "Quercus", "ficus": "Ficus", "jacaranda": "Jacaranda mimosifolia",
    "croton": "Codiaeum variegatum", "ivy": "Hedera helix", "fern": "Polypodiopsida", "cactus": "Opuntia",
    "palm": "Arecaceae", "striga": "Striga hermonthica", "lantana": "Lantana camara", "parthenium": "Parthenium hysterophorus",
    "black_jack": "Bidens pilosa", "couch_grass": "Cynodon dactylon", "grass": "Poaceae",
}
