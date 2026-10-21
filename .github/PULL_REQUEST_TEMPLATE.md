<!-- One change per pull request. Describe what changes for the user, not the diff. -->

## What and why



Closes #

## Checklist

- [ ] `python -m pytest -q` passes (Node 22 installed, so the JS tests run)
- [ ] New or changed texts exist in both `locales/ru` and `locales/en`; `python tools/i18n_extract.py --summary` did not grow
- [ ] New `.py` and `static/js/*.js` files carry `SPDX-License-Identifier: Apache-2.0` (`python tools/license_headers.py --check`)
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]` for every user-visible change
- [ ] Documentation updated (`docs/*.md`; `README.ru.md` if the Russian README is affected)
- [ ] No secrets, internal hostnames or company names in code, fixtures, screenshots or tests
- [ ] Every commit is signed off (`git commit -s`, DCO) and has an English, imperative subject
