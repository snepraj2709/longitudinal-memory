from __future__ import annotations

import unittest

from extraction.contracts import (
    ALLOWED_EPISTEMIC_STATUSES,
    ALLOWED_POLARITIES,
    ALLOWED_PREDICATES,
)
from extraction.schema import (
    ATOMIC_EXTRACTION_SCHEMA_VERSION,
    atomic_extraction_text_format,
)


class AtomicExtractionSchemaTests(unittest.TestCase):
    def test_schema_freezes_the_atomic_claim_envelope(self) -> None:
        text_format = atomic_extraction_text_format()

        self.assertEqual(text_format["type"], "json_schema")
        self.assertEqual(text_format["name"], ATOMIC_EXTRACTION_SCHEMA_VERSION)
        self.assertTrue(text_format["strict"])
        root = text_format["schema"]
        self.assertEqual(root["required"], ["claims"])
        self.assertFalse(root["additionalProperties"])
        claim = root["properties"]["claims"]["items"]
        self.assertEqual(set(claim["required"]), set(claim["properties"]))
        self.assertFalse(claim["additionalProperties"])
        evidence = claim["properties"]["evidence"]["items"]
        self.assertEqual(set(evidence["required"]), set(evidence["properties"]))
        self.assertFalse(evidence["additionalProperties"])
        self.assertEqual(
            set(claim["properties"]["predicate"]["enum"]),
            set(ALLOWED_PREDICATES),
        )
        self.assertEqual(
            set(claim["properties"]["polarity"]["enum"]),
            set(ALLOWED_POLARITIES),
        )
        self.assertEqual(
            set(claim["properties"]["epistemic_status"]["enum"]),
            set(ALLOWED_EPISTEMIC_STATUSES),
        )

    def test_each_nested_object_is_closed_and_schema_is_fresh(self) -> None:
        first = atomic_extraction_text_format()
        second = atomic_extraction_text_format()
        first["name"] = "mutated"

        self.assertEqual(second["name"], ATOMIC_EXTRACTION_SCHEMA_VERSION)
        self._assert_objects_closed(second["schema"])

    def _assert_objects_closed(self, value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                self.assertFalse(value["additionalProperties"])
                self.assertEqual(set(value["required"]), set(value["properties"]))
            for item in value.values():
                self._assert_objects_closed(item)
        elif isinstance(value, list):
            for item in value:
                self._assert_objects_closed(item)


if __name__ == "__main__":
    unittest.main()
