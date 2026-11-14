"""Alert routing: severity sets urgency, and ignored alerts escalate."""
from types import SimpleNamespace

from agents.alert_routing_agent import (
    SEVERITY_BASE_ROUTING,
    AlertRoutingAgent,
    Destination,
    Urgency,
)

classify = AlertRoutingAgent._classify_urgency


def prediction(a="warfarin", b="aspirin"):
    return SimpleNamespace(drug_a=a, drug_b=b)


def patient(**overrides):
    base = dict(is_pregnant=False, is_dialysis=False, inr=None, prior_alerts=[])
    base.update(overrides)
    return SimpleNamespace(**base)


def test_severity_sets_base_urgency():
    assert classify(0, prediction(), None)[0] == Urgency.ROUTINE
    assert classify(2, prediction(), None)[0] == Urgency.PRIORITY
    assert classify(3, prediction(), None)[0] == Urgency.URGENT


def test_major_interactions_reach_the_physician():
    assert Destination.PHYSICIAN in SEVERITY_BASE_ROUTING[3]
    assert Destination.PHYSICIAN not in SEVERITY_BASE_ROUTING[1]


def test_high_inr_with_anticoagulant_is_critical():
    urgency, reason = classify(2, prediction(), patient(inr=4.2))
    assert urgency == Urgency.CRITICAL
    assert "INR" in reason


def test_pregnancy_with_contraindicated_drug_is_critical():
    urgency, reason = classify(1, prediction(), patient(is_pregnant=True))
    assert urgency == Urgency.CRITICAL
    assert "Pregnancy" in reason


def test_unacknowledged_prior_alert_bumps_urgency_one_tier():
    prior = [{"drug_a": "Aspirin", "drug_b": "Warfarin", "was_acknowledged": False}]
    urgency, reason = classify(2, prediction(), patient(prior_alerts=prior))
    assert urgency == Urgency.URGENT
    assert "Prior alert" in reason


def test_acknowledged_prior_alert_does_not_escalate():
    prior = [{"drug_a": "warfarin", "drug_b": "aspirin", "was_acknowledged": True}]
    assert classify(2, prediction(), patient(prior_alerts=prior))[0] == Urgency.PRIORITY
