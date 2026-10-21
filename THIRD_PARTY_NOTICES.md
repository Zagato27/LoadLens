# Third-party notices

LoadLens is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`). It redistributes the third-party
components listed below. Section A covers the files bundled in this repository, with the full text of their
licenses; section B lists the Python packages installed from `requirements.txt`; section C lists the Docker base
images of `docker/docker-compose.yml`.

## A. Files bundled in this repository

| Component | Version | Files | License |
|---|---|---|---|
| Chart.js | 4.4.1 | `static/vendor/chart.umd.js` | MIT |
| marked | 12.0.2 | `static/vendor/marked.min.js` | MIT |
| DOMPurify | 3.1.6 | `static/vendor/purify.min.js` | Apache-2.0 OR MPL-2.0 (redistributed under Apache-2.0) |
| Ace | 1.23.4 | `static/vendor/ace/ace.js`, `static/vendor/ace/ext-searchbox.js`, `static/vendor/ace/mode-json.js`, `static/vendor/ace/theme-chrome.js`, `static/vendor/ace/theme-twilight.js`, `static/vendor/ace/worker-json.js` | BSD-3-Clause |
| Golos Text | @fontsource/golos-text 5.3.0: Latin and Cyrillic woff2 subsets, weights 400, 500, 600 | `static/vendor/fonts/golos-text-*.woff2` | OFL-1.1 |
| JetBrains Mono | @fontsource/jetbrains-mono 5.3.0: Latin and Cyrillic woff2 subsets, weights 400, 500, 600 | `static/vendor/fonts/jetbrains-mono-*.woff2` | OFL-1.1 |

The license of the fonts is also kept next to them in `static/vendor/fonts/OFL.txt`.

### Chart.js

https://www.chartjs.org, https://github.com/chartjs/Chart.js. The bundled file carries the notice
`Chart.js v4.4.1 (c) 2023 Chart.js Contributors, Released under the MIT License`.

```
The MIT License (MIT)

Copyright (c) 2023 Chart.js Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### marked

https://marked.js.org, https://github.com/markedjs/marked. The bundled file carries the notice
`marked v12.0.2 - a markdown parser, Copyright (c) 2011-2024, Christopher Jeffrey. (MIT Licensed)`.

```
Copyright (c) 2018+, MarkedJS (https://github.com/markedjs/)
Copyright (c) 2011-2018, Christopher Jeffrey (https://github.com/chjj/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### DOMPurify

https://github.com/cure53/DOMPurify. The bundled file carries the notice
`DOMPurify 3.1.6 | (c) Cure53 and other contributors | Released under the Apache license 2.0 and Mozilla Public
License 2.0 | github.com/cure53/DOMPurify/blob/3.1.6/LICENSE`.

DOMPurify is dual-licensed: its authors offer it under either the Apache License, Version 2.0, or the Mozilla
Public License, Version 2.0, at the recipient's choice. LoadLens uses and redistributes DOMPurify under the
**Apache License, Version 2.0**; the full text of that license is the file `LICENSE` at the root of this
repository (it is the same license text that applies to LoadLens itself). The MPL-2.0 option is not exercised.

```
DOMPurify
Copyright (c) Cure53 and other contributors (Dr.-Ing. Mario Heiderich, Cure53)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Ace (Ajax.org Cloud9 Editor)

https://ace.c9.io, https://github.com/ajaxorg/ace. Version 1.23.4 (the version string is embedded in
`static/vendor/ace/ace.js`); the bundled files are the core editor, the search box extension, the JSON mode and
worker and the Chrome and Twilight themes.

```
Copyright (c) 2010, Ajax.org B.V.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Ajax.org B.V. nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL AJAX.ORG B.V. BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Golos Text and JetBrains Mono (fonts)

Golos Text: https://github.com/googlefonts/golos-text. JetBrains Mono: https://github.com/JetBrains/JetBrainsMono.
The files in `static/vendor/fonts/` are the Latin and Cyrillic woff2 subsets of the two families published by
Fontsource (https://fontsource.org, packages `@fontsource/golos-text` and `@fontsource/jetbrains-mono` 5.3.0),
weights 400, 500 and 600 of each. Both are licensed under the SIL Open Font License, Version 1.1, whose text
follows and is also kept in `static/vendor/fonts/OFL.txt`.

```
Golos Text
Copyright 2019 The Golos Text Project Authors (https://github.com/googlefonts/golos-text)

JetBrains Mono
Copyright 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono)

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
https://openfontlicense.org


-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```

## B. Python dependencies

The direct dependencies of `requirements.txt`. The table is generated by `tools/third_party_table.py` from the
metadata of the installed packages (`python tools/third_party_table.py --write` refreshes it, `--check` verifies
it; the release script runs it). The packages are not bundled in this repository: pip installs them, together with
their own dependencies, into the Docker image or the virtual environment, where every package keeps its own
license file (`pip show -f <package>` lists it).

<!-- python-deps:start -->

| Package | Version | License | Homepage |
|---|---|---|---|
| Flask | 3.1.3 | BSD-3-Clause | https://flask.palletsprojects.com/ |
| Werkzeug | 3.1.9 | BSD-3-Clause | https://werkzeug.palletsprojects.com/ |
| gunicorn | 26.2.0 | MIT | https://gunicorn.org/ |
| requests | 2.34.2 | Apache-2.0 | https://requests.readthedocs.io/ |
| urllib3 | 2.8.0 | MIT | https://urllib3.readthedocs.io/ |
| psycopg2 | 2.9.13 | LGPL-3.0-or-later WITH OpenSSL linking exception | https://www.psycopg.org/ |
| pandas | 2.3.3 | BSD-3-Clause | https://pandas.pydata.org/ |
| numpy | 1.26.4 | BSD-3-Clause | https://numpy.org/ |
| tabulate | 0.10.0 | MIT | https://github.com/astanin/python-tabulate |
| matplotlib | 3.10.9 | Matplotlib License (PSF-based) | https://matplotlib.org/ |
| pillow | 12.3.0 | MIT-CMU | https://python-pillow.github.io/ |
| celery | 5.4.0 | BSD-3-Clause | https://docs.celeryq.dev/ |
| atlassian-python-api | 3.41.21 | Apache-2.0 | https://github.com/atlassian-api/atlassian-python-api |
| beautifulsoup4 | 4.15.0 | MIT | https://www.crummy.com/software/BeautifulSoup/ |
| pydantic | 2.14.0 | MIT | https://docs.pydantic.dev/ |
| pypdf | 6.7.0 | BSD-3-Clause | https://pypdf.readthedocs.io/ |
| python-docx | 1.2.0 | MIT | https://github.com/python-openxml/python-docx |
| langchain-core | 1.6.8 | MIT | https://github.com/langchain-ai/langchain |
| langchain-gigachat | 0.5.1 | MIT | https://github.com/ai-forever/langchain-gigachat |
| paramiko | 5.0.0 | LGPL-2.1-or-later | https://www.paramiko.org/ |
| PyYAML | 6.0.3 | MIT | https://pyyaml.org/ |
| Jinja2 | >=3.1.6 | BSD-3-Clause | https://jinja.palletsprojects.com/ |
| idna | >=3.7 | BSD-3-Clause | https://github.com/kjd/idna |
| certifi | >=2024.7.4 | MPL-2.0 | https://github.com/certifi/python-certifi |
| h11 | >=0.16.0 | MIT | https://github.com/python-hyper/h11 |

<!-- python-deps:end -->

Notes:

- psycopg2 (LGPL-3.0-or-later with the OpenSSL linking exception) and paramiko (LGPL-2.1-or-later) are used
  unmodified as separately installed packages imported at run time; certifi (MPL-2.0) is used unmodified. Their
  source is available from the homepages above.
- The `celery[redis]` extra installs the redis-py client (MIT).
- Development-only tools (`requirements-dev.txt`: pytest, pip-audit) are not part of the product.

## C. Docker images

`docker/docker-compose.yml` runs LoadLens from the image built by `docker/Dockerfile` next to two upstream images.
They are pulled at deployment time and are not part of this repository.

| Image | Used as | License |
|---|---|---|
| `python:3.12-slim` (Docker official image) | base image of `docker/Dockerfile` | CPython: Python Software Foundation License 2.0 (https://docs.python.org/3/license.html). The image is built on Debian; the Debian packages it contains carry their own licenses, recorded in `/usr/share/doc/*/copyright` inside the image. |
| `timescale/timescaledb:latest-pg16` | metrics and report storage (`timescaledb` service) | PostgreSQL 16: PostgreSQL License (https://www.postgresql.org/about/licence/). TimescaleDB: the core is Apache-2.0, and this image is the Community edition whose additional features (compression, continuous aggregates, multi-node and others) are under the Timescale License, TSL (https://github.com/timescale/timescaledb/blob/main/tsl/LICENSE-TIMESCALE). The TSL allows running the software as part of your own application; it does not allow offering TimescaleDB itself as a hosted service. The image is Alpine-based. |
| `redis:7.4-alpine` (Docker official image) | Celery broker and result backend (`redis` service) | Redis 7.4 is distributed under the Redis Source Available License v2 (RSALv2) or the Server Side Public License v1 (SSPLv1), at the user's choice (https://redis.io/legal/licenses/). Neither is an OSI-approved open-source license; both allow running Redis unmodified as a component of an application such as LoadLens, but not offering it as a managed service. Redis is used unmodified and is not bundled in this repository. A drop-in BSD-3-Clause alternative is Valkey (`valkey/valkey:7.2-alpine`). The image is Alpine-based. |
