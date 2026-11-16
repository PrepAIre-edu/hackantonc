import analysis


def test_daily_shape(hourly_df):
    result = analysis.daily(hourly_df)
    assert list(result.columns) == ["date", "pm25", "aqi"]
    assert len(result) == 120


def test_profiles_have_expected_keys(hourly_df):
    assert analysis.hourly_profile(hourly_df)["hour"].tolist() == list(range(24))
    assert analysis.weekday_profile(hourly_df)["weekday"].tolist() == list(range(7))


def test_matrix_shape_and_weekend_is_worse(hourly_df):
    matrix = analysis.hour_weekday_matrix(hourly_df)
    assert matrix.shape == (7, 24)
    assert matrix.loc[5].mean() > matrix.loc[1].mean()


def test_monthly_and_summary(hourly_df):
    assert {"year", "month", "pm25", "aqi"} <= set(analysis.monthly(hourly_df).columns)
    result = analysis.summary(hourly_df)
    assert result["last_time"] == hourly_df["time"].iloc[-1]
    assert result["worst_day_aqi"] >= result["mean_aqi_30d"]


def test_summary_on_empty_data(hourly_df):
    assert analysis.summary(hourly_df.iloc[:0])["last_aqi"] is None
