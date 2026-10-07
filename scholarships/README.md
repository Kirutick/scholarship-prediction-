# Scholarship knowledge base

The catalog is intentionally separate from the synthetic ML training data.
`scholarships.json` contains only the official National Scholarship Portal
announcement facts verified on 2026-10-07. Both records are marked `partial`:
the announcement confirms the scheme names, merit-based label, renewal window,
and deadlines, but does not provide full eligibility rules, award amounts,
document lists, or scheme-specific application forms.

Unknown facts are `null`; do not treat them as unrestricted or as a passing
eligibility rule. The recommendation engine reports `Needs Verification` and
does not calculate a match score until documented criteria are present and the
student has supplied the required fields. It never reports official eligibility.

## Updating the data

1. Find current scheme guidelines on the official government/institution
   website, not a third-party blog.
2. Add the source URL, publisher, verification date, and concise evidence to
   `sources.json`.
3. Update the matching record in `scholarships.json`; leave unverified values
   `null`.
4. Change `source_status` to `official` only after all eligibility, benefit,
   application, document, and deadline facts shown as verified are supported
   by the cited official source. A partial record remains clearly marked.
5. Run `python -m unittest discover -s tests` and `python scripts/test_prediction_flow.py`.

The catalog is a reviewed static snapshot. The app does not scrape official
sites at prediction time, and saved user status/checklist data stays in that
browser's local storage.
