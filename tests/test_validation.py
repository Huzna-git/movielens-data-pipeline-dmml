"""The validation stage must catch every class of bad record and separate it."""
import pandas as pd

from src.validation.validate import validate_tables


def _raw():
    users = pd.DataFrame({"user_id": ["1", "2", "3", "3", "x"], "age": ["24", "200", "30", "30", "40"],
                          "gender": ["M", "F", "Q", "Q", "M"], "occupation": ["student", "writer", "", "", "doctor"],
                          "zip_code": ["85711", "94043", "32067", "32067", None]})
    items = pd.DataFrame({"item_id": ["1", "2", "3", "4"], "title": ["Toy Story (1995)", "GoldenEye (1995)", "Toy Story (1995)", "unknown"],
                          "release_date": ["01-Jan-1995", "31-Feb-1995", "01-Jan-1995", None],
                          "imdb_url": ["", "", "", ""], "genres": ["Animation|Comedy", "Action", "Animation", ""]})
    inter = pd.DataFrame({"user_id": ["1", "1", "2", "2", "99", "1", "1", "2"],
                          "item_id": ["1", "2", "1", "77", "1", "1", "3", "2"],
                          "rating": ["4", "9", "three", "3", "4", "4", None, "5"],
                          "timestamp": ["880000000", "880000001", "880000002", "880000003", "880000004", "880000000", "880000005", "10"]})
    return {"users": users, "items": items, "interactions": inter}


def test_every_bad_record_is_flagged_and_removed(cfg):
    clean, issues, stats = validate_tables(cfg, _raw())
    checks = set(zip(issues["table"], issues["check"]))
    assert ("users", "invalid_type") in checks            # user_id "x"
    assert ("users", "out_of_range") in checks            # age 200
    assert ("users", "invalid_category") in checks        # gender Q
    assert ("users", "missing_value") in checks           # occupation "" is a missing value after trimming
    assert ("users", "duplicate_row") in checks           # user 3 twice
    assert ("items", "unparseable_date") in checks        # 31-Feb-1995
    assert ("items", "missing_release_date") in checks    # warning
    assert ("items", "duplicate_title") in checks         # warning
    assert ("interactions", "out_of_range") in checks     # rating 9, timestamp 10
    assert ("interactions", "invalid_type") in checks     # rating "three"
    assert ("interactions", "missing_value") in checks    # rating None
    assert ("interactions", "unknown_user") in checks     # user 99
    assert ("interactions", "unknown_item") in checks     # item 77
    assert ("interactions", "duplicate_row") in checks    # (1, 1, 4, ...) twice
    # warnings keep the row, errors remove it
    assert stats["items"]["rows_out"] == 3                # only the unparseable date is an error
    assert stats["users"]["rows_out"] == 1                # user 1 is the only clean profile
    assert stats["interactions"]["rows_out"] == 1         # (1, 1, 4) survives once
    assert set(clean["interactions"]["user_id"]) == {1}
    # every error row is in the issues table with its reason
    errors = issues[issues["severity"] == "error"]
    assert len(errors) >= 12


def test_clean_data_passes(cfg):
    raw = {"users": pd.DataFrame({"user_id": ["1"], "age": ["30"], "gender": ["F"], "occupation": ["engineer"], "zip_code": ["1"]}),
           "items": pd.DataFrame({"item_id": ["1"], "title": ["A"], "release_date": ["01-Jan-2000"], "imdb_url": [""], "genres": ["Drama"]}),
           "interactions": pd.DataFrame({"user_id": ["1"], "item_id": ["1"], "rating": ["5"], "timestamp": ["946684800"]})}
    clean, issues, stats = validate_tables(cfg, raw)
    assert issues.empty
    assert all(s["rows_out"] == 1 for s in stats.values())
    assert clean["interactions"]["rating"].dtype == pd.Float64Dtype()
