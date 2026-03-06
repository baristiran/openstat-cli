"""Lightweight i18n: translate UI strings for OpenStat.

Usage::

    from openstat.i18n import t, set_locale
    set_locale("tr")
    print(t("no_data"))  # → "Veri yüklenmedi."

Supported locales: en (default), tr.
Additional locales can be registered at runtime via ``register_locale()``.
"""

from __future__ import annotations

_LOCALE: str = "en"

_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # Generic errors
        "no_data": "No dataset loaded. Use: load <path>",
        "col_not_found": "Column not found: {col}",
        "unknown_subcmd": "Unknown sub-command: {subcmd}",
        # Data commands
        "load_ok": "Loaded {rows:,} rows × {cols} columns from {path}",
        "save_ok": "Saved to: {path}",
        "describe_header": "Dataset: {name}  |  {rows:,} rows × {cols} columns",
        "summarize_header": "Summary Statistics",
        # Model results
        "model_fitted": "{model} fitted. {info}",
        "model_none": "No model fitted yet.",
        # Export
        "export_docx_ok": "Word document saved: {path}",
        "export_pptx_ok": "PowerPoint saved: {path}",
        # Session
        "session_info_header": "Session Information",
        "seed_set": "Seed set to {seed}. Reproducible random operations enabled.",
        "seed_none": "No seed set.",
        # Dashboard
        "dashboard_closed": "Dashboard closed.",
        "dashboard_missing": (
            "textual is required for the dashboard.\n"
            "Install: pip install textual"
        ),
        # Misc
        "undo_ok": "Undo successful. Restored previous dataset.",
        "undo_fail": "Nothing to undo.",
    },
    "tr": {
        # Generic errors
        "no_data": "Veri kümesi yüklenmedi. Kullanım: load <yol>",
        "col_not_found": "Sütun bulunamadı: {col}",
        "unknown_subcmd": "Bilinmeyen alt komut: {subcmd}",
        # Data commands
        "load_ok": "{path} dosyasından {rows:,} satır × {cols} sütun yüklendi",
        "save_ok": "Kaydedildi: {path}",
        "describe_header": "Veri kümesi: {name}  |  {rows:,} satır × {cols} sütun",
        "summarize_header": "Özet İstatistikler",
        # Model results
        "model_fitted": "{model} tahmin edildi. {info}",
        "model_none": "Henüz model tahmin edilmedi.",
        # Export
        "export_docx_ok": "Word belgesi kaydedildi: {path}",
        "export_pptx_ok": "PowerPoint kaydedildi: {path}",
        # Session
        "session_info_header": "Oturum Bilgisi",
        "seed_set": "Başlangıç değeri {seed} olarak ayarlandı. Tekrarlanabilir rastgele işlemler etkin.",
        "seed_none": "Başlangıç değeri ayarlanmadı.",
        # Dashboard
        "dashboard_closed": "Gösterge paneli kapatıldı.",
        "dashboard_missing": (
            "Gösterge paneli için textual gereklidir.\n"
            "Kurulum: pip install textual"
        ),
        # Misc
        "undo_ok": "Geri alma başarılı. Önceki veri kümesi geri yüklendi.",
        "undo_fail": "Geri alınacak bir şey yok.",
    },
}


def set_locale(locale: str) -> None:
    """Set the active locale (e.g. 'en', 'tr')."""
    global _LOCALE
    if locale not in _STRINGS:
        raise ValueError(
            f"Locale '{locale}' not available. "
            f"Available: {', '.join(_STRINGS)}"
        )
    _LOCALE = locale


def get_locale() -> str:
    """Return the currently active locale code."""
    return _LOCALE


def register_locale(locale: str, strings: dict[str, str]) -> None:
    """Register (or extend) a locale with a mapping of key → translated string.

    Strings for keys not provided fall back to English.
    """
    if locale not in _STRINGS:
        _STRINGS[locale] = {}
    _STRINGS[locale].update(strings)


def t(key: str, **kwargs: object) -> str:
    """Translate *key* using the active locale, with optional format args.

    Falls back to English if the key is missing in the active locale.
    Returns the key itself if missing everywhere.
    """
    locale_map = _STRINGS.get(_LOCALE, {})
    template = locale_map.get(key) or _STRINGS["en"].get(key) or key
    if kwargs:
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    return template
