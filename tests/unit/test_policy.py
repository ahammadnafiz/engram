"""Unit tests for MemoryPolicy slotting and typing."""

from __future__ import annotations


class TestAllergySlots:
    """Positive and negative allergy statements must share one conflict slot
    so a correction supersedes the contradicted fact."""

    def test_positive_allergy_slot(self) -> None:
        from engram.policy import DEFAULT_MEMORY_POLICY

        slot = DEFAULT_MEMORY_POLICY.critical_slot(
            "User is allergic to penicillin", "profile"
        )
        assert slot == "profile:allergy:penicillin"

    def test_negative_allergy_shares_slot_with_positive(self) -> None:
        from engram.policy import DEFAULT_MEMORY_POLICY

        positive = DEFAULT_MEMORY_POLICY.critical_slot(
            "User is allergic to penicillin", "profile"
        )
        negative = DEFAULT_MEMORY_POLICY.critical_slot(
            "User is not allergic to penicillin", "profile"
        )
        assert negative == positive, (
            "contradictory allergy facts must conflict, not coexist"
        )

    def test_compound_allergens_not_truncated(self) -> None:
        from engram.policy import DEFAULT_MEMORY_POLICY

        slot = DEFAULT_MEMORY_POLICY.critical_slot(
            "User is allergic to shellfish and peanuts", "profile"
        )
        assert slot == "profile:allergy:shellfish_and_peanuts"

    def test_unslotted_critical_fact_gets_no_generic_conflict_key(self) -> None:
        """Generic sha-digest slots can never match a reworded contradiction,
        so they only add a junk conflict_key and a useless supersede query
        per add. Critical typing must still apply."""
        from engram.policy import DEFAULT_MEMORY_POLICY

        mtype, meta = DEFAULT_MEMORY_POLICY.apply_metadata(
            content="User's daughter Lily started school",
            agent_id="agent",
            user_id="u1",
            memory_type="profile",
        )

        assert mtype == "profile"
        assert meta.get("critical") is True  # critical pinning stays
        assert "conflict_key" not in meta  # no inert digest slot
        assert "critical_slot" not in meta

    def test_generic_slots_still_available_by_opt_in(self) -> None:
        from engram.policy import MemoryPolicy

        policy = MemoryPolicy(name="opt_in", generic_critical_slots=True)
        slot = policy.critical_slot("User's daughter started school", "profile")

        assert slot is not None
        assert slot.startswith("profile:generic:")

    def test_contradiction_produces_same_conflict_key(self) -> None:
        from engram.policy import DEFAULT_MEMORY_POLICY

        _, meta_pos = DEFAULT_MEMORY_POLICY.apply_metadata(
            content="User is allergic to shellfish",
            agent_id="agent",
            user_id="u1",
            memory_type="profile",
        )
        _, meta_neg = DEFAULT_MEMORY_POLICY.apply_metadata(
            content="User is NOT allergic to shellfish",
            agent_id="agent",
            user_id="u1",
            memory_type="profile",
        )
        assert meta_pos["conflict_key"] == meta_neg["conflict_key"]
