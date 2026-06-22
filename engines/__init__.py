"""Scoring engines for the Dub Worthiness Score app.

Each engine is a pure-ish function operating on a transcript (and a little
audio metadata) and returning a structured dict. The app wires them together
via scorer.py. Keeping them separate makes each one independently testable and
independently defensible.
"""
