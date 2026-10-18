"""
Automated checks. Run inside Docker:   sudo docker compose exec web python -m unittest discover -s tests -v
or locally from the project folder:    python -m unittest discover -s tests -v
"""
import os, sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ["AUTO_UPDATE"] = "0"                    # never start downloads/training from a test
os.environ["AT_API_KEY"] = ""                      # never send a real SMS from a test
for p in (ROOT, ROOT / "api", ROOT / "pipeline", ROOT / "model"):
    sys.path.insert(0, str(p))


class TestQuefts(unittest.TestCase):
    """The Python QUEFTS must give exactly what the original C++ (Rquefts, quefts.cpp) gives."""
    CASES = [((46.7, 5.8, 103.9, 0, 0, 0, 6000, 120), 2262.5619), ((46.7, 5.8, 103.9, 92, 46, 30, 6000, 120), 3961.0849),
             ((60, 10, 60, 60, 10, 60, 5000, 120), 3756.9493), ((20, 2, 40, 50, 20, 0, 8000, 120), 1443.7362),
             ((80, 15, 30, 0, 30, 60, 3000, 90), 2607.2683), ((46.7, 5.8, 103.9, 150, 60, 60, 6000, 300), 5662.3192),
             ((10, 1, 10, 200, 100, 100, 9000, 120), 3629.2006)]

    def test_matches_cpp(self):
        from agro.quefts import quefts
        for c, ref in self.CASES:
            y = quefts("Maize", *c[:6], store_att=c[6], season_days=c[7])[0]
            self.assertAlmostEqual(y, ref, places=3, msg=str(c))

    def test_more_fertilizer_never_lowers_yield(self):
        from agro.quefts import quefts
        ys = [quefts("Maize", 40, 6, 90, n, n / 3, n / 3, 6000)[0] for n in range(0, 200, 20)]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(ys, ys[1:])))


class TestSmsLanguages(unittest.TestCase):
    CASES = [("PRICE MAIZE NAIROBI", "maize", "en"), ("BEI MAHINDI ELDORET", "maize", "sw"),
             ("PRIX MAÏS BAMAKO", "maize", "fr"), ("Preço do milho Maputo", "maize", "pt"),
             ("سعر ذرة رفيعة Khartoum", "sorghum", "ar"), ("سعر أرز Khartoum", "rice", "ar"),
             ("ዋጋ ጤፍ Addis Ababa", "teff", "am"), ("FARASHI MASARA KANO", "maize", "ha"),
             ("qiimaha bariis Mogadishu", "rice", "so"), ("igiciro ibigori Kigali", "maize", "rw"),
             ("iye iresi Lagos", "rice", "yo"), ("intengo ummbila Durban", "maize", "zu"),
             ("prix niébé Niamey", "cowpea", "fr")]

    def test_parse(self):
        import languages as L
        for text, crop, lang in self.CASES:
            c, l, _ = L.parse(text)
            self.assertEqual((c, l), (crop, lang), text)

    def test_kind_of_message(self):
        import languages as L
        self.assertTrue(L.is_plan("MBOLEA MAHINDI KITALE"))
        self.assertTrue(L.is_insure("BIMA MAHINDI KITALE"))
        self.assertFalse(L.is_plan("BEI MAHINDI KITALE") or L.is_insure("BEI MAHINDI KITALE"))

    def test_every_language_has_every_reply(self):
        import languages as L
        for lang in L.LANGS:
            for table in (L.TEXT, L.PLAN_TEXT, L.INSURE_TEXT):
                self.assertIn(lang, table)


class TestInsuranceFeatures(unittest.TestCase):
    def test_shapes_and_wrsi(self):
        from insure.features import season_features, N_DEKADS, SEQ, STATIC
        L = 120
        wet = season_features(np.full(L, 8.0), np.full(L, 4.0), np.full(L, 28.0), np.full(L, 5.0), "maize", 1.0, 35.0)
        dry = season_features(np.full(L, 0.5), np.full(L, 4.0), np.full(L, 28.0), np.full(L, 5.0), "maize", 1.0, 35.0)
        self.assertEqual(wet[0].shape, (N_DEKADS, len(SEQ))); self.assertEqual(wet[2].shape, (len(STATIC),))
        w = STATIC.index("wrsi")
        self.assertGreater(wet[2][w], 0.95); self.assertLess(dry[2][w], 0.3)


class TestDiseaseAdvice(unittest.TestCase):
    def test_every_class_has_advice_in_three_languages(self):
        from disease.advice import A
        from disease.fetch import PV, CASSAVA
        for lab in set(PV.values()) | set(CASSAVA.values()):
            self.assertIn(lab, A)
            for lang in ("en", "sw", "fr"):
                self.assertEqual(len(A[lab][lang]), 2, (lab, lang))


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.c = app.app.test_client()

    def test_pages(self):
        for p in ("/", "/plan", "/insure"):
            self.assertEqual(self.c.get(p).status_code, 200, p)

    def test_endpoints_answer(self):
        for p in ("/api/status", "/api/datasets", "/api/plan/meta", "/api/insure/meta"):
            self.assertEqual(self.c.get(p).status_code, 200, p)

    def test_sms_always_replies(self):
        for t in ("BEI MAHINDI NAIROBI", "MBOLEA MAHINDI KITALE", "BIMA MAHINDI KITALE", "hello"):
            r = self.c.post("/api/sms", json={"text": t})
            self.assertEqual(r.status_code, 200, t); self.assertTrue(r.get_json()["reply"])

    def test_new_pages_and_endpoints(self):
        self.assertEqual(self.c.get("/farm").status_code, 200)
        self.assertEqual(self.c.get("/api/farm?lat=x").status_code, 400)
        self.assertEqual(self.c.get("/api/bundle?lat=1&lon=35&cost=100").status_code, 200)
        self.assertEqual(self.c.get("/api/where_to_sell").status_code, 400)

    def test_ussd_menus(self):
        def say(text):
            return self.c.post("/api/ussd", data={"text": text}).get_data(as_text=True)
        self.assertTrue(say("").startswith("CON ") and "1." in say(""))
        self.assertTrue(say("1").startswith("CON "))            # crop menu
        self.assertTrue(say("1*1").startswith("CON "))          # ask for the town
        self.assertTrue(say("1*1*KITALE").startswith("END "))   # final answer
        self.assertIn("Karibu", say("4*2"))                     # Swahili menu
        self.assertTrue(say("9").startswith("END "))            # invalid choice ends politely
        for t in ("", "1*1*KITALE", "4*2*2*1*KITALE"):
            self.assertLessEqual(len(say(t)), 4 + 182)          # fits one USSD screen

    def test_more_pages(self):
        for p in ("/admin", "/coop", "/warning", "/diagnose"):
            self.assertEqual(self.c.get(p).status_code, 200, p)
        for p in ("/api/farmers", "/api/messages", "/api/coops", "/api/warning", "/api/diagnose/meta"):
            self.assertEqual(self.c.get(p).status_code, 200, p)

    def test_join_and_stop_words(self):
        import languages as L
        self.assertEqual(L.is_join("JIUNGE MAHINDI KITALE"), "sw")
        self.assertEqual(L.is_join("JOIN MAIZE KANO"), "en")
        self.assertEqual(L.is_stop("ACHA"), "sw")
        for lang in L.LANGS:
            self.assertIn(lang, L.JOIN_TEXT)

    def test_bad_input_is_refused_politely(self):
        self.assertEqual(self.c.post("/api/farmers", json={"phone": "", "crop": "maize", "place": "Kitale"}).status_code, 400)
        self.assertEqual(self.c.post("/api/coops", json={"name": ""}).status_code, 400)
        self.assertIn(self.c.post("/api/diagnose").status_code, (400, 503))

    def test_plan_needs_coordinates(self):
        self.assertEqual(self.c.get("/api/plan?lat=x").status_code, 400)


if __name__ == "__main__":
    unittest.main()


class FakeClip:
    """Stands in for CLIP in tests (no download): deterministic vectors from the text / the image pixels."""
    D = 32

    def texts(self, ps):
        import hashlib
        out = []
        for p in ps:
            r = np.random.default_rng(int(hashlib.md5(p.encode()).hexdigest()[:8], 16)); v = r.normal(size=self.D)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def images(self, ims):
        out = []
        for im in ims:
            a = np.asarray(im.resize((4, 4))).astype(float).ravel()[:self.D]; a = a - a.mean() + 1e-3
            out.append(a / np.linalg.norm(a))
        return np.stack(out)


def _photo(color=(60, 140, 50)):
    import io
    from PIL import Image
    im = Image.new("RGB", (64, 64), color)
    for x in range(10, 30):
        im.putpixel((x, x), (120, 70, 20))
    b = io.BytesIO(); im.save(b, "JPEG"); return b.getvalue()


class TestUniversalLeafDoctor(unittest.TestCase):
    def test_advice_tables_complete(self):
        from disease import care, plants
        for g in plants.SYMPTOMS:
            if g == "healthy":
                continue
            for lang in ("en", "sw", "fr"):
                s = care.symptom(g, lang)
                self.assertTrue(s["name"] and s["why"] and len(s["steps"]) >= 3, (g, lang))
        for k, t in care.TIP.items():
            self.assertIn(k, plants.PLANTS); self.assertEqual(set(t), {"en", "sw", "fr"})
        for lang in ("en", "sw", "fr"):
            self.assertGreaterEqual(len(care.care("coffee", lang)), 5)
            self.assertEqual(set(care.TEXT[lang]), set(care.TEXT["en"]))
        self.assertEqual(care.lang_ok("xx"), "en")

    def test_plantvillage_labels_map_to_groups(self):
        from disease import plants
        for f in ["Apple___Apple_scab", "Tomato___Tomato_mosaic_virus", "Soybean___healthy", "Squash___Powdery_mildew",
                  "Corn_(maize)___Common_rust_", "Pepper,_bell___healthy", "Orange___Haunglongbing_(Citrus_greening)"]:
            sp, g = plants.pv_group(f)
            self.assertIn(sp, plants.SPECIES_KEY); self.assertIn(g, plants.SYMPTOMS)
        for sp in plants.SPECIES_KEY.values():
            self.assertIn(sp, plants.PLANTS)

    def test_engine_and_thresholds(self):
        from disease import universal as U
        e = U.Engine(FakeClip()); e.probe = None
        V = e.b.images([__import__("PIL.Image", fromlist=["Image"]).new("RGB", (8, 8), (50, 150, 40))])
        keys, pr = e.plant_probs(V)
        self.assertEqual(len(keys), pr.shape[1]); self.assertAlmostEqual(float(pr.sum()), 1.0, places=5)
        probs, _ = e.health(V, "coffee")
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertEqual(U.status_of(0.9), "sick"); self.assertEqual(U.status_of(0.1), "healthy")
        self.assertEqual(U.status_of(0.5), "unsure")
        self.assertEqual(U.match_key("Arabica coffee", "Coffea arabica"), "coffee")
        self.assertEqual(U.match_key("Mango", ""), "mango")

    def test_api_any_plant(self):
        import app as A
        from disease import universal as U
        e = U.Engine(FakeClip()); e.probe = None
        old = (U.engine, U.ready)
        U.engine, U.ready = (lambda: e), (lambda: True)
        try:
            c = A.app.test_client()
            m = c.get("/api/diagnose/meta").get_json()
            self.assertTrue(m["ready"]); self.assertGreater(len(m["plants"]), 100)
            for crop in ("", "coffee", "maize"):
                r = c.post("/api/diagnose", data={"photo": (__import__("io").BytesIO(_photo()), "leaf.jpg"), "crop": crop,
                                                  "lang": "sw"}, content_type="multipart/form-data")
                self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
                j = r.get_json()
                self.assertIn(j["kind"], ("plant", "not_plant"))
                if crop:
                    self.assertEqual(j["kind"], "plant"); self.assertEqual(j["plant"]["key"], crop)
                if j["kind"] == "plant":
                    self.assertIn(j["health"]["status"], ("sick", "healthy", "unsure"))
                    self.assertTrue(j["care"])
                    if j["health"]["status"] != "healthy":
                        self.assertTrue(j["symptom"]["steps"])
            bad = c.post("/api/diagnose", data={"photo": (__import__("io").BytesIO(b"not an image"), "x.jpg")},
                         content_type="multipart/form-data")
            self.assertEqual(bad.status_code, 400)
        finally:
            U.engine, U.ready = old


class TestSpecialistOnlyFallback(unittest.TestCase):
    def test_unsure_specialist_does_not_force_a_crop(self):
        """Any-plant model not downloaded yet + a maple leaf: must NOT answer 'sick tomato'."""
        import io, app as A
        from disease import universal as U
        old = (A.disease_model.ready, A.disease_model.diagnose, A.disease_model.meta, U.ready)
        A.disease_model.ready = lambda: True
        A.disease_model.meta = lambda: {"classes": ["maize_healthy", "tomato_late_blight", "cassava_healthy"]}
        A.disease_model.diagnose = lambda d, c=None: {"top": [("tomato_late_blight", 0.35), ("cassava_healthy", 0.09)], "sure": False}
        U.ready = lambda: False
        try:
            j = A.app.test_client().post("/api/diagnose", data={"photo": (io.BytesIO(_photo()), "x.jpg")},
                                         content_type="multipart/form-data").get_json()
            self.assertEqual(j["kind"], "unknown")
            A.disease_model.diagnose = lambda d, c=None: {"top": [("tomato_late_blight", 0.9)], "sure": True, "p_sick": 0.9}
            j = A.app.test_client().post("/api/diagnose", data={"photo": (io.BytesIO(_photo()), "x.jpg")},
                                         content_type="multipart/form-data").get_json()
            self.assertEqual(j["plant"]["key"], "tomato"); self.assertEqual(j["health"]["status"], "sick")
        finally:
            A.disease_model.ready, A.disease_model.diagnose, A.disease_model.meta, U.ready = old


class TestAutumnLeaf(unittest.TestCase):
    def test_colour_rule(self):
        from PIL import Image
        from disease import universal as U
        orange = Image.new("RGB", (64, 64), (255, 255, 255))
        green = Image.new("RGB", (64, 64), (255, 255, 255))
        for x in range(8, 56):
            for y in range(8, 56):
                orange.putpixel((x, y), (235, 110, 20)); green.putpixel((x, y), (40, 150, 50))
        self.assertTrue(U.is_autumn(U.colour_stats(orange)))
        self.assertFalse(U.is_autumn(U.colour_stats(green)))
        self.assertFalse(U.is_autumn(U.colour_stats(Image.new("RGB", (64, 64), (255, 255, 255)))))


class TestBothModelsVote(unittest.TestCase):
    def test_specialist_alone_cannot_call_a_healthy_field_sick(self):
        import io, app as A
        from disease import universal as U
        old = (A.disease_model.ready, A.disease_model.diagnose, A.disease_model.meta, U.ready, U.analyse)
        A.disease_model.ready = lambda: True
        A.disease_model.meta = lambda: {"classes": ["cassava_green_mite", "cassava_healthy"]}
        A.disease_model.diagnose = lambda d, c=None: {"top": [("cassava_green_mite", 0.75)], "sure": True, "p_sick": 0.8}
        U.ready = lambda: True
        U.analyse = lambda d, lang="en", crop=None: {
            "is_plant": True, "p_plant": 0.99, "plant": {"key": "cassava", "name": "cassava", "prob": 0.57, "source": "clip"},
            "alternatives": [], "autumn": False, "colour": {}, "p_sick": 0.1, "p_old": 0.0,
            "symptoms": [("leaf_spot", 0.5), ("pests", 0.3)], "probe": True, "probe_weight": 0.5}
        try:
            j = A.app.test_client().post("/api/diagnose", data={"photo": (io.BytesIO(_photo()), "x.jpg")},
                                         content_type="multipart/form-data").get_json()
            self.assertNotEqual(j["verdict"]["kind"], "sick")          # 0.5*0.8 + 0.5*0.1 = 0.45 -> not sick
            self.assertEqual(j["health"]["source"], "both")
        finally:
            A.disease_model.ready, A.disease_model.diagnose, A.disease_model.meta, U.ready, U.analyse = old


class TestPhotoPrototypes(unittest.TestCase):
    def test_prototype_can_lift_the_right_plant(self):
        from disease import universal as U
        e = U.Engine(FakeClip()); e.probe = None
        v = np.ones(FakeClip.D) / np.sqrt(FakeClip.D)
        V = v[None, :]
        keys, L = e.plant_logits(V)
        plain = keys[int(np.argmax(L[0]))]
        B = e.proto_boost(V, keys, alpha=2.0, proto={"cassava": (v, 0.0, 0.1)})       # z = 10 -> +16
        self.assertGreater(B[0, keys.index("cassava")], 10)
        big = e.proto_boost(V, keys, alpha=100.0, proto={"cassava": (v, 0.0, 0.1)})
        self.assertEqual(keys[int(np.argmax(L[0] + big[0]))], "cassava")
        self.assertTrue(np.all(e.proto_boost(V, keys, alpha=0.0, proto={"cassava": (v, 0.0, 0.1)}) == 0))
        self.assertIn(plain, keys)


class TestSpeciesNamer(unittest.TestCase):
    def test_namer_path(self):
        from PIL import Image
        from disease import universal as U, plants as Pl
        e = U.Engine(FakeClip(), namer=FakeClip()); e.probe = None
        img = Image.new("RGB", (16, 16), (30, 140, 40))
        keys, pr = e.name_plant(e.b.images([img]), img)
        self.assertEqual(len(keys), len(Pl.PLANTS)); self.assertAlmostEqual(float(pr.sum()), 1.0, places=5)
        self.assertEqual(set(Pl.SCI), set(Pl.PLANTS))
