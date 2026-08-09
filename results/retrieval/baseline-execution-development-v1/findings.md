# Findings

All 24 deterministic baseline runs completed without a failure. B2 returned only atomic records, B3 returned only session records, and B4 searched both kinds. Every accepted row kept its claim and source-span lineage.

This is a runtime mechanics release. It contains no relevance labels and reports no retrieval-quality or timing measurements. The local signed token-hash vector mostly rewards shared tokens and hash collisions; it is not a semantic embedding.

The frozen development index is candidate-heavy. It has no accepted durative claims, checked relation links, or multi-version claim chain, so the development run produced 0 expansion traces. The integration fixtures cover both expansion paths separately.
