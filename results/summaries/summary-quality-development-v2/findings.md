# Step 6.4 summary-quality findings

The frozen runtime produced a bundle for all 10 cases without a failure. All 170 statement instances kept exact statement-to-Claim-version-to-span provenance.

Strict Claim-and-evidence matching found no event matches. Recall was 0/22 and precision was 0/165. Evidence recall was 0/31 and precision was 0/165. Because no event matched, the current-versus-historical metric is null rather than zero. The correction and uncertainty checks were 0/2 and 0/9.

Two inherited limits explain the low score. The baseline returns every visible session summary instead of selecting for the instruction, so one user's broad bundle repeats across cases. The extracted Claims also differ from the reviewed Claims on at least one exact semantic or evidence field. The scorer does not use lexical or semantic similarity to hide those differences.

There were no cross-user, stale, unsupported, or runtime-failure records. Durative output remained empty. No model ran; historical OpenAI spend remains $0.2314404.

This is not a blind evaluation. The implementing agent had already seen the authorized v1 development gold before the whitespace-only runtime correction. The v2 runtime and predictions were frozen while all scorer and gold files were absent, and the carried gold bytes were not reinterpreted or changed.
