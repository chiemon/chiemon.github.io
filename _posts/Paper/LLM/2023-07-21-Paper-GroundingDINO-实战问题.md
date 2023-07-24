---
layout: post
title: GroundingDINO 实战问题
category: Paper
tags: GroundingDINO
keywords: GroundingDINO
description:
---

## 1. MaxRetryError

**Error:**

```bash
'(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /bert-base-uncased/resolve/main/tokenizer_config.json (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x7f2a1c08aeb0>, 'Connection to huggingface.co timed out. (connect timeout=10)'))"), '(Request ID: b9b328c8-dbcb-42f0-921b-a50d27911ece)')' thrown while requesting HEAD https://huggingface.co/bert-base-uncased/resolve/main/tokenizer_config.json
```

**Solver:**

```python
# tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, local_files_only=True)
```

## 2. SSLError

```bash
SSLError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /dslim/bert-base-NER/resolve/main/tokenizer_config.json (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1108)')))
```

**Solver:**

- [github](https://github.com/huggingface/transformers/issues/17611#issuecomment-1323272726)

- [stackoverflow](https://stackoverflow.com/questions/75110981/sslerror-httpsconnectionpoolhost-huggingface-co-port-443-max-retries-exce)

1.

```bash
pip install requests==2.27.1
# or
conda install requests==2.27.1
```
2.

```python
import os
os.environ['CURL_CA_BUNDLE'] = ''
```