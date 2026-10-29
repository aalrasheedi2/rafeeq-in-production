<!--
Configuration B: the ONE change appended to A's instructions (rafeeq.configs()["B"]["system"] = A's system + "\n" + the
text below this comment). This comment is stripped before loading; only the text after it reaches the model.

Status: CHOSEN from A's error analysis (rules-structural-wall, fingerprint 5c438a45ebf5). Rafeeq searched three times
in Arabic and always got POL-01's short Arabic summary, which does not name the required document; the English
policy text ("for structural works, a structural engineer's report") is found by an English query (verified locally
with search_policies, no model). The change is about retrieval only: no tool, permission, approval or policy changes.

Editing this file changes B's config_id only. A's instructions and config_id do not read it.
Rejected (do not reintroduce): booking a slot the applicant did not pick themselves. It contradicts POL-10
("confirm the slot with the applicant") and data/skills/inspection-booking/SKILL.md ("Book only after the person
picks a slot"). See EVAL_NOTES_A.md.
-->
Madar's policies are written in English with only a short Arabic summary. When a question comes in Arabic, also call
search_policies with an English query for the key terms, and base the answer on the English policy text. Reply in
Arabic, keeping the policy's own terms for required documents, roles and deadlines (for example "structural engineer"
as «مهندس إنشائي»).
