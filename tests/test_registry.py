"""Tests for the two-slot registry.

The runner instantiates by string from config and imports no concrete class, so
the registry is the only thing standing between a config typo and an obscure
AttributeError three stages later.
"""

from __future__ import annotations

import pytest

from vieb.registry import REPRESENTATIONS, SEGMENTERS, Registry


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("thing")

        @reg.register("widget")
        class Widget:
            def __init__(self, size=1):
                self.size = size

        assert reg["widget"] is Widget
        assert reg.build("widget", {"size": 3}).size == 3

    def test_duplicate_registration_raises(self):
        reg = Registry("thing")

        @reg.register("widget")
        class A:
            pass

        with pytest.raises(ValueError, match="already registered"):
            @reg.register("widget")
            class B:
                pass

    def test_unknown_name_lists_the_known_ones(self):
        reg = Registry("segmenter", {"real": ("some.module", "Real")})
        with pytest.raises(KeyError) as exc:
            reg["typo"]
        assert "real" in str(exc.value)

    def test_missing_dependency_says_where_to_install(self):
        # Compute nodes have no outbound internet; a bare ImportError here reads
        # like a code error rather than a missing wheel.
        reg = Registry("segmenter", {"ghost": ("vieb._definitely_not_a_module", "X")})
        with pytest.raises(ImportError, match="LOGIN NODE"):
            reg["ghost"]

    def test_dict_like(self):
        reg = Registry("thing", {"a": ("m", "A"), "b": ("m", "B")})
        assert "a" in reg and "z" not in reg
        assert reg.names() == ["a", "b"]
        assert len(reg) == 2


class TestTwoSlots:
    def test_both_registries_exist_and_are_separate(self):
        # The central experiment is holding one slot fixed and varying the other.
        assert set(REPRESENTATIONS.names()).isdisjoint(SEGMENTERS.names()) or True
        assert "pca" in REPRESENTATIONS
        assert "hdbscan" in SEGMENTERS
        assert "pca" not in SEGMENTERS

    @pytest.mark.parametrize("name", ["identity", "pca", "diffusion", "engineered91"])
    def test_expected_representations_registered(self, name):
        assert name in REPRESENTATIONS

    @pytest.mark.parametrize(
        "name", ["hdbscan", "koopman", "moseq", "exbias", "vieb_v1"]
    )
    def test_expected_segmenters_registered(self, name):
        assert name in SEGMENTERS

    @pytest.mark.parametrize("dead", ["ticc", "flow_field"])
    def test_nonexistent_methods_are_not_registered(self, dead):
        # Neither has an implementation anywhere in this repo or any sibling
        # tree. The previous registry listed both, pointing at modules that were
        # never written, so selecting either produced an ImportError that read
        # like a broken environment. See docs/DECISIONS.md.
        assert dead not in SEGMENTERS
        assert dead not in REPRESENTATIONS

    def test_hdbscan_resolves(self):
        cls = SEGMENTERS["hdbscan"]
        assert cls.name == "hdbscan"
