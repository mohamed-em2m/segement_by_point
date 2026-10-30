#!/usr/bin/env python
"""Thin wrapper around ``pointseg-check-data`` for reviewers."""

from pointseg.datasets.loveda import check_data_main

if __name__ == "__main__":
    raise SystemExit(check_data_main())
