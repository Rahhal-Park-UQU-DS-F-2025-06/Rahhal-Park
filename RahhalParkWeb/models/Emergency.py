import re
import uuid


def detect_lang_auto(text: str) -> str:
    return "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"


def normalize_ar(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_case_id():
    return "RHL-" + uuid.uuid4().hex[:8].upper()


KEYWORDS = {
    "ambulance": {
        "ar": ["اغماء", "مغمى", "دوخه", "غثيان", "نزيف", "كسر", "مريض", "تنفس"],
        "en": ["faint", "dizzy", "nausea", "bleeding", "injury"]
    },
    "police": {
        "ar": ["شجار", "سرقه", "تحرش", "اعتداء", "تهديد"],
        "en": ["fight", "theft", "harassment", "attack"]
    },
    "civil_defense": {
        "ar": ["حريق", "دخان", "غاز", "تماس", "تعطل", "احتجاز"],
        "en": ["fire", "smoke", "gas", "electric", "stuck"]
    }
}

AGENCY_NAME = {
    "ar": {
        "ambulance": "فريق الإسعاف داخل المدينة",
        "police": "فريق الأمن داخل المدينة",
        "civil_defense": "فريق السلامة داخل المدينة"
    },
    "en": {
        "ambulance": "Park Medical Team",
        "police": "Park Security Team",
        "civil_defense": "Park Safety Team"
    }
}


def keyword_category(text: str, lang: str):
    clean_text = normalize_ar(text) if lang == "ar" else text.lower()

    scores = {"ambulance": 0, "police": 0, "civil_defense": 0}

    for cat in KEYWORDS:
        for word in KEYWORDS[cat][lang]:
            if word in clean_text:
                scores[cat] += 1

    best_cat = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[best_cat] / total if total > 0 else 0

    return best_cat, round(confidence, 2)


def ask_menu(lang):
    if lang == "ar":
        return (
            "ما فهمت الحالة.\n"
            "اختار:\n"
            "1) طبية\n"
            "2) أمنية\n"
            "3) سلامة / حريق"
        )
    else:
        return (
            "I couldn't identify the emergency.\n"
            "Choose:\n"
            "1) Medical\n"
            "2) Security\n"
            "3) Safety / Fire"
        )


def format_handoff(lang, cat):
    case_id = make_case_id()
    agency = AGENCY_NAME[lang][cat]

    if lang == "ar":
        return (
            f"تم إرسال البلاغ إلى: {agency}\n"
            f"رقم البلاغ: {case_id}\n"
            f"يرجى البقاء في موقعك."
        )
    else:
        return (
            f"Report sent to: {agency}\n"
            f"Case ID: {case_id}\n"
            f"Please stay in your location."
        )