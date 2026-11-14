"""CYP450 profiles: the enzyme roles the interaction model is built on."""
from ml.features.cyp450_features import CYP_REGEX, CYP_RISK_WEIGHT, DrugCYPProfile


def test_enzyme_names_are_normalized():
    profile = DrugCYPProfile("fluconazole")
    profile.add_role("cyp 2c9", "inhibitor")
    profile.add_role("CYP3A4", "inhibitor")
    assert profile.inhibitors == {"CYP2C9", "CYP3A4"}


def test_roles_are_kept_separate():
    profile = DrugCYPProfile("warfarin")
    profile.add_role("CYP2C9", "substrate")
    profile.add_role("CYP3A4", "substrate")
    assert profile.substrates == {"CYP2C9", "CYP3A4"}
    assert profile.inhibitors == set() and profile.inducers == set()
    assert profile.all_enzymes == {"CYP2C9", "CYP3A4"}


def test_regex_finds_enzymes_in_label_text():
    text = "Warfarin is metabolized by CYP2C9 and, to a lesser extent, CYP 3A4."
    assert [m.upper().replace(" ", "") for m in CYP_REGEX.findall(text)] == ["CYP2C9", "CYP3A4"]


def test_major_enzymes_carry_the_highest_risk_weight():
    assert CYP_RISK_WEIGHT["CYP3A4"] == max(CYP_RISK_WEIGHT.values())
    assert CYP_RISK_WEIGHT["CYP2D6"] == 3
