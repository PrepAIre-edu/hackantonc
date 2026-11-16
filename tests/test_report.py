import report


def test_build_summary_contains_all_sections(hourly_df):
    text = report.build_summary("kyiv", {"kyiv": hourly_df, "lviv": hourly_df})
    for expected in ("Київ", "1. Зараз", "2. Закономірності", "3. Прогноз", "4. Якість прогнозу",
                     "5. Порівняння міст", "SaveEcoBot"):
        assert expected in text


def test_labelled_frames_use_ukrainian_names(hourly_df):
    assert list(report.labelled_frames({"kyiv": hourly_df, "unknown": hourly_df})) == ["Київ", "Unknown"]
