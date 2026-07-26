# 문피아 웹소설 데이터 수집 · 전처리 파이프라인

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey">
  <img alt="Tests" src="https://img.shields.io/badge/tests-77%20passing-brightgreen">
  <img alt="Status" src="https://img.shields.io/badge/status-verified%20on%20live%20site-success">
</p>

웹소설 **독자 이탈률 추정**과 **고정 팬층 반응 분석**을 위한 도구입니다.
작품 메타데이터 · 회차별 시계열 지표 · 독자 댓글을 수집해 정수화·구조화된
CSV/JSON으로 내보내고, 이탈률·충성도 파생 지표를 만든 다음,
**scikit-learn 모델로 이탈을 예측**하는 데까지 이어집니다.

```
crawl  ──▶  features  ──▶  sentiment  ──▶  train
수집        파생 지표      댓글 감정(KOTE)    이탈 예측 · 요인 순위
```

**실측 검증 완료** (2026-07-26):

```
작품 389건 · 회차 10,845건 · 댓글 95,125건 · 고유 댓글 작성자 16,580명
KOTE 감정 채점 78,754건 (작가 본인 댓글 3,473건 제외)
조인 무결성 100% (고아 레코드 0건, 중복 0건)

모델 성능 (GroupKFold 교차검증, 기저선 0.500)
  회차 이탈   ROC-AUC 0.897 ± 0.012   (작품 298 · 회차 5,923)
  독자 이탈   ROC-AUC 0.844 ± 0.004   (독자 16,546 · 등장 89,885)
  작품 간     ROC-AUC 0.838 ± 0.064   (작품 260)
```

Windows에서 `run.bat` 더블클릭이면 끝납니다. → [빠른 시작](#빠른-시작--그냥-실행하기)

---

## ⚠️ 사용 전 반드시 읽어주세요

**이 도구는 개인 연구·분석 목적으로 만들어졌습니다.**

- **수집한 데이터를 재배포하지 마세요.** 댓글에는 실제 이용자의 닉네임과 회원
  식별자(`blogUrl`)가 포함됩니다. 개인정보이므로 공개 저장소·공개 데이터셋에
  올리면 안 됩니다. 이 저장소의 `.gitignore`가 `data/`와 모든 `*.csv`를 막고 있는
  이유입니다.
- **작품 본문은 수집하지 않습니다.** `robots.txt`가 막은 경로이며 저작물입니다.
  ([상세](#본문-전문을-수집하지-않는-이유))
- **요청 간격을 줄이지 마세요.** 기본 딜레이는 사람이 읽는 속도보다 느리게
  잡혀 있습니다. 서버에 부담을 주지 않는 선을 지켜주세요.
- 문피아 이용약관과 관련 법령을 준수하는 것은 사용자 본인의 책임입니다.

---

## 목차

- [빠른 시작 — 그냥 실행하기](#빠른-시작--그냥-실행하기)
- [수집 방식이 Selenium이 아닌 이유](#수집-방식이-selenium이-아닌-이유)
- [수집 가능 범위 (실측)](#수집-가능-범위-실측)
- [유료 회차 결제 구조](#-유료-회차는-계정이-유료면-전부가-아닙니다)
- [사용법](#사용법)
- [출력 스키마](#출력-스키마)
- [전처리 규칙](#전처리-규칙-요구사항-34)
- [파생 피처](#파생-피처-features-명령)
- [댓글 감정 분석](#댓글-감정-분석-sentiment-명령)
- [이탈 예측 모델](#이탈-예측-모델-train-명령)
- [이탈 요인 순위](#이탈-요인-순위)
- [예외 처리 · 크롤링 매너](#예외-처리--크롤링-매너-요구사항-4)
- [구조](#구조)
- [분석 예시](#분석-예시)
- [알려진 제약](#알려진-제약)

---

## 수집 방식이 Selenium이 아닌 이유

원래 요구사항은 Playwright/Selenium + BeautifulSoup을 전제했는데,
**실제 사이트 구조가 그 전제와 다릅니다.**

문피아는 React SPA로 전환되어 작품 페이지 HTML에 데이터가 하나도 없습니다:

```html
<body><div id="root"></div></body>   <!-- www.munpia.com/novel/detail/166 의 전체 body -->
```

모든 데이터는 SPA가 호출하는 내부 JSON API에서 옵니다. 그래서 이 파이프라인은
**브라우저 DOM 파싱 대신 그 JSON API를 직접 호출합니다.** 엔드포인트와 파라미터는
배포된 프런트 번들(`cdn1.munpia.com/v2/pc-novel/.../index.js`)에 정의된 실물을 확인해서
맞췄고, 전부 실제 응답으로 검증했습니다.

이 편이 나은 이유:

| | DOM 스크래핑 | JSON API (채택) |
|---|---|---|
| 정확도 | 렌더된 문자열 재파싱 (`"조회 1,234"` → 1234) | 숫자를 숫자 그대로 |
| 속도 | 회차당 브라우저 렌더 수 초 | 요청당 수백 ms |
| 안정성 | CSS 클래스 바뀌면 전부 깨짐 | 스키마 변경 전까지 안정 |
| 결측 | 지연 로딩 놓치면 조용히 누락 | 페이지네이션 `total`로 검증 가능 |

Playwright는 **로그인 세션 쿠키 확보 용도로만** 남겨뒀습니다 (아래 "유료 회차 댓글" 참고).

---

## 수집 가능 범위 (실측)

`robots.txt`와 실제 응답을 확인한 결과입니다.

| 항목 | 상태 |
|---|---|
| 작품 메타데이터 | ✅ 비로그인 수집 가능 |
| 회차별 조회/추천/댓글수/게시일시 | ✅ 비로그인 수집 가능 |
| 회차 반응 버튼(최고/웃김/놀람/응원/감동) | ✅ 비로그인 (`--entry-detail`) |
| 독자 성별·연령 분포 | ✅ 비로그인 수집 가능 |
| **무료 회차 댓글** | ✅ **비로그인 수집 가능** |
| **유료 회차 댓글** | ⚠️ 로그인 + **해당 회차 구매** 필요 (아래 참고) |
| 회차 본문 전문 | ❌ 수집하지 않음 (사유 아래) |

### 💰 유료 회차는 "계정이 유료면 전부"가 아닙니다

문피아 결제 구조를 API로 확인한 결과입니다:

```json
"purchase": { "originPerPrice": 100, "salePerPrice": 100, "enableSelective": "ENABLE" }
```

**회차당 100골드(≈100원) 결제 방식**이고, 무제한 정액제가 아닙니다. 회차마다
`purchased` / `rented` 플래그가 따로 있습니다. 즉 유료 계정으로 로그인해도
**실제로 구매·대여한 회차의 댓글만** 열립니다.

규모를 가늠해 보면 — 445화짜리 작품에서 무료 25화를 빼면 420화 × 100원 = **약 42,000원**,
작품 하나 기준입니다. 25개 작품이면 백만 원 단위가 됩니다.

그래서 크롤러는 이렇게 동작합니다:

- **구매하지 않은 유료 회차는 요청조차 하지 않습니다** — `purchased`/`rented`
  플래그로 미리 걸러내므로 권한 오류로 시간을 낭비하지 않습니다
- **결제 API는 호출하지 않습니다.** `/api/v1/pc/paid/order/...` 계열은 코드에
  존재하지 않습니다. 조회 전용이라 수집 중 돈이 빠져나갈 일이 없습니다
- 무료 계정이든 유료 계정이든 **접근 가능한 범위까지 자동으로** 수집합니다

> 문피아에 별도의 무제한 구독 상품이 있는지는 결제 API만으로는 확인할 수 없었습니다.
> 위 내용은 일반 유료 연재작의 회차별 판매 정보에 근거한 것입니다.

### 본문 전문을 수집하지 않는 이유

문피아 `robots.txt`가 명시적으로 막고 있습니다:

```
User-agent: *
Allow: /novel/detail/      ← 이 파이프라인이 쓰는 경로
Disallow: /novel/viewer/   ← 본문 뷰어
```

게다가 본문은 `/api/key-exchange` + `/api/novels/{id}/stream` 으로 오는 암호화 스트림이라
사실상 DRM입니다. 대신 **분량 지표**를 대체 제공합니다:

- `episodes.pages` — 문피아 표기 쪽수 (원본 값)
- `episodes.char_estimate` — `pages × 900자` 추정치
- `novels.total_characters` — 작품 전체 글자 수 (API 제공 원본 값)

이탈률 분석에 필요한 것은 "이 회차가 평소보다 짧았나/길었나"이고, `pages`로 충분히
잡힙니다. 감정 분석은 애초에 본문이 아니라 댓글이 대상이므로 영향이 없습니다.

---

## 빠른 시작 — 그냥 실행하기

**Windows**: `run.bat` 을 더블클릭하세요.
**macOS / Linux**: `chmod +x run.sh && ./run.sh`

처음 실행하면 가상환경과 필요한 패키지를 알아서 설치합니다(1~2분). 그다음부터는
바로 시작됩니다. 질문에 답하기만 하면 **수집 → 피처 → (선택)감정 분석 → 모델 학습**
까지 끝납니다.

> 감정 분석은 `torch`·`transformers`가, 모델 학습은 `scikit-learn`이 있어야 합니다.
> 없으면 설치 방법만 안내하고 건너뜁니다 — 이미 모은 데이터는 그대로 남습니다.

```
==================================================================
  문피아 웹소설 데이터 수집기
==================================================================
  독자 이탈률 · 고정 팬층 분석용 데이터를 모읍니다.
  질문에 답하기만 하면 됩니다. 엔터를 누르면 [기본값]이 선택됩니다.

[1/4] 로그인
------------------------------------------------------------------
  유료 회차 댓글을 수집하려면 로그인이 필요합니다.
    로그인하시겠습니까? (Y/n): y
  문피아 아이디: myaccount
  비밀번호 (입력해도 화면에 보이지 않습니다):
  로그인 중... 성공

[2/4] 수집할 작품 선택
------------------------------------------------------------------
  1) 인기·최신 작품을 자동으로 찾기
  2) 작품 주소(URL) 또는 번호 직접 입력
  3) 파일에서 목록 불러오기
  선택 [1]: 1
  몇 개 작품을 수집할까요? [20]: 20

[3/4] 수집 옵션
[4/4] 수집 시작
------------------------------------------------------------------
  [1/20] 재활만 했는데 드래곤의 힘을 얻었다 — 회차 153건, 댓글 904건
  [2/20] 신내림 받고 이혼함 — 회차 119건, 댓글 1102건
  ...
```

Python이 없으면 설치 방법을 안내합니다. Python 3.8 이상이면 됩니다.

> 중간에 `Ctrl+C`로 멈춰도 그때까지 모은 데이터는 저장되고, 다시 실행하면
> 남은 작품부터 이어서 수집합니다.

명령줄로 세밀하게 제어하려면 아래 [사용법](#사용법)을 보세요.

---

## 설치 (수동)

```bash
pip install -r requirements.txt

# 브라우저 로그인이 필요할 때만
playwright install chromium
```

Python 3.8+ 에서 동작합니다.

---

## 사용법

### 1. 지정한 작품 수집

```bash
python -m munpia.cli crawl --novel-ids 587273 582918 --out data/raw
```

작품 ID 대신 URL을 그대로 넣어도 됩니다:

```bash
python -m munpia.cli crawl --novel-ids https://www.munpia.com/novel/detail/587273
```

ID 목록 파일로도 가능합니다:

```bash
python -m munpia.cli crawl --id-file novel_ids.txt --out data/raw
```

### 2. 대상 작품 자동 탐색 + 수집

```bash
python -m munpia.cli discover --limit 50 --crawl --out data/raw
python -m munpia.cli discover --genres FANTASY,HEROISM --free-only --crawl
```

### 3. 학습용 피처 생성

```bash
python -m munpia.cli features --in data/raw --out data/features
```

### 4. 댓글 감정 분석 (KOTE)

```bash
pip install torch transformers
python -m munpia.cli sentiment --raw data/raw --out data/features
```

회차 단위 '내용' 신호를 만듭니다. 자세한 내용은 [댓글 감정 분석](#댓글-감정-분석-sentiment-명령) 참고.

### 5. 이탈 예측 모델 학습 · 요인 순위

```bash
# 요인 순위를 볼 때 (권장 설정)
python -m munpia.cli train --task episode --max-episode 25 \
    --label-mode within_novel --fixed-effect

# 새 작품 예측력을 볼 때
python -m munpia.cli train --task all --max-episode 25
```

자세한 내용은 [이탈 예측 모델](#이탈-예측-모델-train-명령) 참고.

### 6. (선택) 유료 회차 댓글용 로그인

`.env.example`을 복사해 자격증명을 채웁니다:

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```

```ini
MUNPIA_ID=your_id
MUNPIA_PW=your_password
```

```bash
python -m munpia.cli login                          # .env 읽어 자동 로그인
python -m munpia.cli check-login --novel-id 479065  # 유료 댓글이 실제로 열리는지 확인
python -m munpia.cli crawl --id-file ids.txt --cookies data/cookies.json --comment-scope all
```

**동작 방식**

문피아 로그인 폼(`nssl.munpia.com/login`)에 `_csrf` 토큰과 함께 자격증명을 POST 하고
세션 쿠키만 `data/cookies.json`에 저장합니다. **비밀번호는 저장되지도, 로그에 남지도
않습니다** — 요청 본문에만 쓰입니다. 아이디는 로그에 `mu********` 형태로 마스킹됩니다.

`.env`는 `os.environ`에 주입하지 않습니다. 하위 프로세스로 자격증명이 새는 걸 막기
위해서이고, 테스트로 검증돼 있습니다.

**캡차가 뜨는 경우**

로그인 페이지의 reCAPTCHA는 평소엔 비활성이지만 로그인 실패가 누적되면 켜집니다.
그러면 자동 로그인을 중단하고 브라우저 방식으로 안내합니다:

```bash
pip install playwright && playwright install chromium
python -m munpia.cli login --browser   # 창에서 직접 로그인, 쿠키만 저장
```

SNS 로그인이나 2단계 인증을 쓰는 계정도 `--browser`를 사용하세요.

> `.env`와 `cookies.json`은 `.gitignore`에 등록돼 있어 커밋되지 않습니다.

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--comment-scope` | `free` | `free`=무료회차만, `all`=전체(로그인 필요), `none`=댓글 생략 |
| `--min-delay` / `--max-delay` | 0.8 / 1.8 | 요청 간 랜덤 딜레이(초) |
| `--retries` | 3 | 5xx·네트워크 오류 재시도 횟수 (지수 백오프) |
| `--entry-detail` | off | 회차 상세를 추가 호출해 반응 버튼 수집 (요청 2배) |
| `--max-episodes` | 전체 | 작품당 최대 회차 수 (테스트용) |
| `--format` | `csv` | `csv` 또는 `jsonl` |
| `--no-resume` | off | 완료 목록 무시하고 재수집 |

---

## 출력 스키마

```
data/raw/
├── novels.csv        작품 1행
├── episodes.csv      회차 1행
├── comments.csv      댓글 1행
├── _completed.txt    수집 완료한 작품 ID (재개용)
├── _errors.log       작품별 실패 내역
└── crawl.log         전체 실행 로그
```

### 조인 관계

```
novels.novel_id  ──1:N──▶  episodes.novel_id
episodes.episode_uid  ──1:N──▶  comments.episode_uid
comments.parent_id  ──▶  comments.comment_id   (대댓글 self-join)
```

`episode_uid` = `"{novel_id}_{entry_id}"`, `comment_uid` = `"{novel_id}_{entry_id}_{comment_id}"`

### novels.csv (33컬럼) — 요구사항 3.1

| 컬럼 | 설명 |
|---|---|
| `novel_id` `title` `author_name` | 작품 ID / 제목 / 작가명 |
| `genre_main` `genres` | 대표 장르 / 전체 장르(`\|` 구분) |
| `introduction` | 시놉시스 본문 |
| `total_view_count` `total_like_count` `preference_count` | 총 조회수 / 총 추천수 / **총 선작수** |
| `serial_status` `status_code` | 연재중·완결·휴재 / 0·1·2 |
| `chapter_count` `free_chapter_count` `total_characters` | 총 회차 / 무료 회차 / 총 글자수 |
| `is_free` `is_adult` `is_paid_serial` `is_exclusive` `is_contest` `is_ebook` | 0/1 플래그 |
| `created_at` `updated_at` `created_ts` `updated_ts` | 일시 문자열 / 유닉스 타임스탬프 |
| `reader_male_count` `reader_female_count` `reader_age10s_pct` … `reader_age50s_pct` | 독자 인구통계 |

### episodes.csv (25컬럼) — 요구사항 3.2

| 컬럼 | 설명 |
|---|---|
| `episode_uid` `novel_id` `entry_id` `episode_num` `title` | 식별자 / 회차 번호 / 회차 제목 |
| `published_at` `published_ts` | 게시 일시 |
| `view_count` `like_count` `comment_count` | 회차별 조회수 / 추천수 / 댓글 수 |
| `comment_collected` `comment_status` | 실제 수집된 댓글 수 / `ok`·`skipped`·`permission`·`error` |
| `pages` `char_estimate` | 분량(쪽) / 추정 글자수 |
| `is_free` `is_adult` `is_notice` | 0/1 플래그 |
| `is_purchased` `is_rented` | 구매·대여 여부 (로그인 시에만 채워짐) |
| `reaction_total` `reaction_best` `reaction_funny` `reaction_amazing` `reaction_cheer` `reaction_impressed` | 반응 버튼 (`--entry-detail` 필요) |
| `author_comment` | 작가의 말 |

### comments.csv (24컬럼) — 요구사항 3.3

| 컬럼 | 설명 |
|---|---|
| `comment_uid` `episode_uid` `novel_id` `entry_id` `episode_num` `comment_id` | 식별자 |
| `user_key` | **작성자 안정 식별자** — 유저별 연속 작성/유지율 추적용 |
| `nickname` `blog_url` | 표시 닉네임 / 회원 슬러그 |
| `body` | **댓글 본문 (감정 분석 입력)** |
| `content_type` `sticker_url` `body_char_len` | `TEXT`/`STICKER` / 스티커 URL / 본문 길이 |
| `created_at` `created_ts` | 작성 일시 |
| `like_count` `dislike_count` | 추천수 / 비추천수 |
| `parent_id` `reply_level` `is_reply` `reply_count` | 대댓글 여부 및 개수 |
| `is_secret` `is_blocked` | 0/1 플래그 |

> `user_key`는 `blogUrl`(회원 고정 슬러그) 기반으로 `u_` 접두사를 붙입니다. 닉네임은
> 변경 가능하므로 신뢰하지 않고, `blogUrl`이 없을 때만 닉네임 해시(`n_` 접두사)로
> 대체합니다. 접두사로 어느 쪽인지 구분되니 필터링이 가능합니다.

---

## 전처리 규칙 (요구사항 3.4)

`munpia/preprocess.py` 한 곳에서만 값을 가공합니다.

**텍스트** (`clean_text`)
- HTML 태그·엔티티 제거, `<br>` → 개행
- 제어문자·제로폭 문자 제거, 탭·중복 공백 정리
- 유니코드 NFC 정규화 (자모 분리 상태 닉네임 통일)
- `ㅋㅋㅋㅋㅋㅋ` → `ㅋㅋㅋㅋ` (4자 초과 반복만 축약)

> **이모지와 `ㅋㅋ`·`ㅠㅠ`는 지우지 않습니다.** 감정 분석에서는 이것들이 그 자체로
> 가장 강한 라벨 신호입니다. 과도한 반복만 길이 폭주 방지 차원에서 잘라냅니다.

**숫자** (`to_int`) — `"1,234"` → `1234`, `"1.2만"` → `12000`, 파싱 실패 시 `0`

한 필드가 깨졌다고 회차 전체를 버리지 않습니다. 결측을 0으로 두고 수집 성공률을
지키는 쪽이 낫습니다.

**불리언** (`to_bool_int`) — 모두 0/1 정수. `True`/`False` 문자열이 CSV에 섞이지 않습니다.

**일시** — `created_at`(문자열)과 `created_ts`(유닉스 초) 두 형태로 저장. 시계열
모델에는 `_ts`를 그대로 쓰면 됩니다.

---

## 파생 피처 (`features` 명령)

### episode_features.csv — 이탈률

| 컬럼 | 의미 |
|---|---|
| `retention_from_first` | 1화 대비 잔존율. 작품 간 비교용 정규화 값 |
| `retention_step` | 직전 회차 대비 잔존율 |
| `churn_step` | `1 - retention_step` = 해당 회차의 원시 이탈률 |
| `churn_step_ex_paywall` | 페이월 경계 ±1화를 제외한 이탈률 |
| **`churn_step_clean`** | **페이월 + 미성숙 회차까지 제외한 실질 이탈률 (학습 권장)** |
| `is_paywall_boundary` | 무료→유료 전환 첫 회차면 1 |
| `last_free_episode` `episodes_from_paywall` | 마지막 무료 회차 / 경계까지의 거리 |
| `age_days` `is_mature` | 게시 후 경과일 / 7일 이상이면 1 |
| `days_since_prev` | 직전 회차와의 연재 간격(일). 공백↔이탈 가설 검증용 |
| `view_ma5` `view_vs_ma5` | 5회차 이동평균 대비 편차. 특정 회차의 급락 탐지 |
| `like_per_view` `comment_per_view` `reaction_per_view` | 조회 대비 참여 강도 |
| `reaction_best_pct` … `reaction_impressed_pct` | 반응 버튼 5종 **구성비** |
| `d_reaction_*_pct` | 직전 회차 대비 구성비 변화량 |
| `reaction_entropy` `d_reaction_entropy` | 반응 다양성 (0=한 종류 쏠림, 1=고르게 분산) |

> #### 반응 버튼에서 쓸 수 있는 것은 '구성비'뿐입니다
>
> 실측에서 **`reaction_total`은 `like_count`와 100% 같은 값**이었습니다. 총합은
> 추천수의 별칭이라 새 정보가 없습니다.
>
> 남는 것은 다섯 종류의 비율인데, 여기도 **'웃김'이 95.9%** 를 차지합니다
> (최고 2.0% · 응원 1.5% · 놀람 0.3% · 감동 0.3%). 문피아 UI 특성으로 보입니다.
> 그래도 작품 **내부** 변동이 작품 **간** 변동보다 크므로(비율 1.35) 회차별 상대
> 비교에는 쓸 수 있습니다. 절대 구성비는 작품 성향에 좌우되니 `d_*` 변화량을
> 함께 보세요.

### 이탈률을 무엇으로 잴 것인가 (`--churn-basis`)

```bash
python -m munpia.cli features --in data/raw --out data/features --churn-basis like
```

| | `view` (기본) | `like` |
|---|---|---|
| 기준 | 조회수 | 추천수 |
| 페이월 경계에서 튀는 정도 | 16.7배 | **3.8배** |
| 무료/유료 평균 이탈률 차이 | 0.0414 | **0.0187** |
| 이탈률이 음수인 회차 | 무료 14.3% · 유료 20.8% | 무료 33.0% · 유료 35.5% |
| **모델 AUC (같은 1~25화)** | **0.896 ± 0.007** | 0.836 ± 0.018 |

**척도 일관성은 추천수가 확실히 낫지만, 라벨로서는 조회수가 낫습니다.**
추천수는 절대 규모가 작아 회차 간 변동이 크고, 이탈률이 음수로 나오는 회차가
3분의 1이나 됩니다. 그 노이즈 손해가 척도 일관성 이득보다 컸습니다 — 같은
1~25화 구간에서 AUC가 0.060 낮습니다.

`like`는 **유료 구간을 반드시 봐야 할 때의 차선책**으로 쓰세요. 거기서는
조회수 기준 자체가 성립하지 않으므로 노이즈가 있어도 유일한 선택지입니다.
기본 분석은 `view` + `--max-episode 25`가 낫습니다.

> 초반 이탈 패턴은 두 기준 모두에서 보존됩니다(2화가 최대, 이후 단조 감소).
> 순위가 바뀌는 게 아니라 신호 대 잡음비가 달라지는 것입니다.

> #### ⚠️ `churn_step`을 그대로 쓰면 안 되는 이유 (실측)
>
> 수집 데이터에서 이탈률 상위를 뽑으면 **전부 25~26화**가 나옵니다. 작품이 갑자기
> 재미없어진 게 아니라, 문피아 유료 연재가 **앞 25화만 무료**이기 때문입니다.
> 그 지점의 조회수 급락은 이탈이 아니라 **결제 장벽**입니다.
>
> 그다음으로 높은 건 **각 작품의 최신 회차**입니다. 조회수가 누적값이라 방금 올라온
> 회차는 필연적으로 낮습니다.
>
> 이 두 노이즈를 걷어내면 평균 이탈률이 이렇게 움직입니다:
>
> ```
> 원본             0.0183
> 페이월 경계 제외   0.0132
> + 미성숙 회차 제외 0.0083   ← churn_step_clean
> ```
>
> 그러고 나서야 진짜 신호가 보입니다 — **2~5화 초반 이탈**(최대 66%). 웹소설
> 이탈 분석의 핵심 구간이고, 노이즈에 완전히 가려져 있던 부분입니다.

### episode_features.csv — 팬층 (댓글 있을 때)

| 컬럼 | 의미 |
|---|---|
| **`returning_commenter_ratio`** | 직전 회차에도 댓글을 단 사람 비율 = **고정 팬층 두께** |
| `new_commenter_ratio` | 신규 유입 비율 |
| `commenter_count` `new_commenter_count` `returning_commenter_count` | 원시 카운트 |

### user_features.csv — 독자 개인별 충성도

| 컬럼 | 의미 |
|---|---|
| `episodes_commented` `episode_span` | 참여 회차 수 / 첫~마지막 구간 |
| **`max_consecutive_episodes`** | **연속 작성 최대 길이** |
| `engagement_density` | 자기 구간 중 실제 참여 비율. 1에 가까울수록 매 회차 따라오는 독자 |
| `loyalty_ratio` | 작품 전체 회차 대비 참여 비율 |
| **`is_core_fan`** | 3회차 이상 참여 + 연속 2회 이상 → 1 |

`is_core_fan`은 **시작 기준선**이지 정답이 아닙니다. 작품 길이와 댓글 밀도에 따라
임계값을 조정하세요 (`munpia/features.py`).

---

## 댓글 감정 분석 (`sentiment` 명령)

```bash
pip install torch transformers
python -m munpia.cli sentiment --raw data/raw --out data/features
```

본문은 `robots.txt`가 막고 있어 수집하지 않습니다. 그래서 "이 회차가 어땠나"를
직접 잴 수 없고 독자 반응으로 대리합니다. 반응 버튼은 5종뿐인 데다 **'웃김'이
95.9%** 를 차지해 해상도가 낮습니다. 남는 것이 댓글 본문이고, 이걸 44개 감정으로
펼치는 것이 [KOTE](https://huggingface.co/searle-j/kote_for_easygoing_people)입니다.

### 44개를 그대로 쓰지 않는 이유

작품 수십 개 규모에서 회차당 44개 피처는 과적합의 지름길입니다. 그리고 이탈
분석에서 44개가 같은 무게를 갖지 않습니다.

```
"재미없음", "지긋지긋"  → 이탈 예고. 독자가 떠나기 직전의 말이다.
"슬픔", "공포", "절망"  → 이탈 신호가 아니다. 긴장감 있는 전개에 몰입한 반응이고,
                          오히려 작품이 잘 작동하고 있다는 증거일 수 있다.
```

이 둘을 "부정 감정"으로 뭉뚱그리면 신호가 상쇄됩니다. 그래서 **이탈과 어떤
관계인가**를 기준으로 9개 축으로 묶습니다.

| 축 | 뜻 | 예시 라벨 |
|---|---|---|
| `sent_boredom` | 이탈 직전의 말 | 재미없음 · 귀찮음 · 지긋지긋 |
| `sent_complaint` | 불만은 있지만 아직 읽는 중 | 불평/불만 · 짜증 · 실망 |
| `sent_hostility` | 작품·작가를 향한 적대 | 화남/분노 · 증오/혐오 |
| `sent_enjoyment` | 재미있게 읽는 중 | 즐거움/신남 · 기쁨 · 행복 |
| `sent_anticipation` | 다음 화를 부르는 힘 | 기대감 · 신기함/관심 · 놀람 |
| `sent_attachment` | 고정 팬층의 언어 | 아껴주는 · 존경 · 고마움 |
| **`sent_tension`** | **몰입형 부정 정서 — 이탈 아님** | 슬픔 · 공포 · 절망 · 불안 |
| `sent_confusion` | 전개를 못 따라가는 상태 | 당황/난처 · 의심/불신 |
| `sent_neutral` | 감정 없음 | 없음 |

`sent_churn_index` = (boredom + complaint + hostility) − (enjoyment + anticipation
+ attachment). **tension은 일부러 넣지 않습니다.**

원본 44개 점수도 `comment_sentiment.csv`에 남으므로, 축 정의를 바꿔 재집계할 때
모델을 다시 돌릴 필요가 없습니다. 이어서 실행하면 이미 채점한 댓글은 건너뜁니다.

### 실측 검증

댓글 95,125건 중 78,754건을 채점했습니다(스티커·초단문·작가 댓글 제외).
감정 축과 이탈률의 상관은 부호 구조가 전부 이론과 맞습니다:

```
sent_confusion    +0.106      sent_anticipation  +0.003
sent_churn_index  +0.090      sent_attachment    -0.103
sent_hostility    +0.085      sent_enjoyment     -0.106
sent_complaint    +0.084
sent_boredom      +0.064
```

> **다만 이 상관이 예측력으로 이어지지는 않았습니다.** 이탈 예측 피처로서의
> 기여는 확인되지 않았습니다 —
> [자세한 검증](#댓글-감정은-이탈-예측에-추가-기여를-하지-않습니다-검증-완료)

> **작가 본인 댓글은 자동으로 제외됩니다.** 한 작품은 댓글의 **49.9%가 작가**였습니다
> (작가가 모든 댓글에 답글). 작가는 독자 반응이 아니고 매 회차 등장하므로, 그대로
> 두면 회차별 감정이 작가의 인사말로 덮이고 `returning_commenter_ratio`도 부풀려집니다.

> **댓글 3건 미만 회차는 값을 비웁니다** (`--min-comments`). 댓글 1건으로 만든 감정
> 평균을 "이 회차의 분위기"라고 부를 수 없습니다. 버리지 않고 NaN으로 두는 이유는
> 대치 여부를 모델 쪽에서 정하고 결측 지시자로도 쓸 수 있게 하기 위해서입니다.

---

## 이탈 예측 모델 (`train` 명령)

`churn_step_clean`은 **계산된 관측값**이지 추론이 아닙니다. 조회수 두 개를 나눈
산술 결과라서 이미 일어난 이탈을 사후에 기술할 뿐, "이 회차는 왜 떨어졌나",
"다음 회차는 위험한가"에 답하지 못합니다. `train`이 그 위를 덮는 추론 레이어입니다.

```bash
python -m munpia.cli train --features data/features --raw data/raw --out data/models
```

### 세 가지 이탈을 따로 풉니다

| task | 표본 하나 = | 라벨 | 묻는 질문 |
|---|---|---|---|
| `episode` | 회차 | `churn_step_clean`이 상위 분위수를 넘는가 | 어떤 **회차**가 위험한가 |
| `user` | 독자 × 등장 | 이 회차를 끝으로 다시 댓글이 없는가 | 어떤 **독자**가 떠나는가 |
| `novel` | **작품** | 초반 25화 평균 이탈률이 상위인가 | 왜 이 **작품**이 이탈이 큰가 |

`user` 쪽이 이탈의 정의로는 가장 곧습니다 — 조회수는 대리 지표지만 독자 재등장은
관측된 사실입니다. 대신 **댓글을 쓴 독자만 보인다**는 한계가 있습니다.

`novel`은 **작품 하나가 표본 하나**입니다. 회차를 아무리 많이 모아도 이 표본은
늘지 않습니다. 작품 10개로 돌리면 로지스틱 AUC가 1.0으로 나오는데, 폴드당 검증
표본이 2개라서 나오는 숫자일 뿐 의미가 없습니다. **최소 200개는 모으세요.**

`--task all`이 셋 다, `both`가 episode+user를 돌립니다.

### 비교하는 모델

로지스틱 회귀가 주력입니다. 계수가 곧 해석이라 "연재 공백이 1 표준편차 늘면
이탈 오즈가 N배" 같은 문장을 그대로 뽑을 수 있습니다.

`logistic` · **`logistic_l1`** · `random_forest` · `hist_gbm` · `dummy`(기저율 하한선)

`logistic_l1`은 계수를 0으로 밀어 피처를 스스로 고릅니다. 표본 수백 행에 피처
수십 개인 지금 상황에서 "무엇이 실제로 남는가"를 보는 데 L2보다 곧습니다 —
실측에서 L2가 AUC 0.616일 때 L1은 0.701이었습니다.

교차검증은 항상 다섯 다 돌리고, `--model`로 지정한 것만 `.joblib`으로 저장합니다.

### ⚠️ 두 가지 평가 모드 — 숫자를 섞어 읽지 마세요

`--fixed-effect`를 켜면 교차검증 방식이 **자동으로 바뀝니다.** 묻는 질문이 달라지기
때문입니다.

| 모드 | 분할 | 묻는 질문 |
|---|---|---|
| 기본 | `GroupKFold` (작품 단위) | 처음 보는 **작품**의 이탈을 맞힐 수 있는가 |
| `--fixed-effect` | `StratifiedKFold` (회차 단위) | 한 작품 **안에서** 어떤 회차가 유독 빠지는가 |

작품 고정효과는 작품 더미를 넣어 작품 간 차이를 통째로 흡수합니다. 그러면 남는
변동이 순수하게 회차 간 차이가 되어 **요인 순위를 볼 때 맞는 설정**입니다.
다만 `GroupKFold`와는 같이 쓸 수 없습니다 — 검증 폴드의 작품 더미는 학습에서
본 적이 없어 무용지물이 되기 때문입니다. 그래서 자동으로 전환합니다.

두 모드의 AUC를 나란히 비교하는 것은 의미가 없습니다.

### ⚠️ 라벨 누수 — 이 모듈이 가장 신경 쓴 부분

`episode_features.csv`를 그대로 `LogisticRegression`에 넣으면 ROC-AUC가 0.99를
넘습니다. **전부 가짜입니다.** 라벨이

```
churn_step_clean = 1 - view_count / prev_view_count
```

이므로 피처에 `view_count`나 `retention_step`, `view_vs_ma5`가 남아 있으면 모델이
라벨을 그냥 역산합니다. `like_per_view`·`comment_per_view`도 분모가 `view_count`라
같은 경로로 새어듭니다.

그래서 이렇게 막았습니다:

- **피처는 화이트리스트로 고정** (`munpia/model.py`의 `EPISODE_STATIC_FEATURES`).
  블랙리스트가 아니라서 새 컬럼이 생겨도 자동으로 딸려 들어가지 않습니다.
- **참여 지표는 한 칸 밀어서만** 사용 — `prev_returning_commenter_ratio` 처럼
  직전 회차 값으로 넣습니다. 현재 회차 지표는 라벨과 같은 시점의 관측입니다.
- **`_assert_no_leakage`가 실행 시점에 재확인** — 금지 컬럼이 섞이면 학습 전에
  예외로 터집니다. 조용히 통과시키면 AUC 0.99를 보고 잘 됐다고 착각하게 됩니다.
- **검증은 `GroupKFold`** — 무작위 K-Fold는 같은 작품의 회차를 학습·검증에 나눠
  넣어서 작품별 조회수 수준을 외우게 합니다. `episode`는 `novel_id`,
  `user`는 `user_key`로 묶습니다.
- **전처리를 파이프라인 안에** — 스케일러를 밖에서 미리 fit 하면 검증 폴드의
  평균·분산이 새어듭니다.

### 우측 절단 (`user` task)

독자의 마지막 댓글이 최신 회차에 있으면, 떠난 게 아니라 **다음 회차가 아직 안
나온 것**일 수 있습니다. 그대로 이탈로 라벨링하면 안 되므로 작품의 마지막 3개
회차에서의 등장은 학습에서 제외합니다 (`censor_last_n`). 조회수 쪽 `is_mature`와
같은 취지입니다.

### 출력

```
data/models/
├── episode_churn_cv_scores.csv     모델별 ROC-AUC / PR-AUC / Brier
├── episode_churn_factor_rank.csv   ★ 이탈 요인 순위 (아래 참고)
├── episode_churn_explain.csv       계수와 오즈비 (트리 계열은 순열 중요도)
├── episode_churn_predictions.csv   회차별 이탈 확률 — 위험 회차 랭킹
├── episode_churn_summary.json      라벨 정의 · 피처 목록 · 표본 수
├── episode_churn_model.joblib      학습된 파이프라인
├── user_churn_*                    (독자 단위)
└── novel_churn_*                   (작품 단위)
```

`pr_auc_lift`는 PR-AUC를 기저율로 나눈 값입니다. 불균형 라벨에서 ROC-AUC는
후하게 나오므로 이쪽을 함께 보세요. 1.0이면 무작위와 같습니다.

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--task` | `both` | `episode` / `user` / `both` |
| `--model` | `logistic` | 최종 저장할 모델 |
| `--label-mode` | `quantile` | `quantile`=상위 분위수, `absolute`=절대 이탈률 |
| `--threshold` | `0.75` | 분위수(0~1) 또는 절대 임계값 |
| `--folds` | `5` | GroupKFold 폴드 수 (그룹 수보다 크면 자동 축소) |
| `--class-weight` | `none` | `balanced`는 확률 보정을 깨뜨립니다 (아래 참고) |

작품마다 이탈률 수준이 달라서 기본값은 분위수 기준입니다. 절대 기준이 필요하면
`--label-mode absolute --threshold 0.05` 처럼 쓰세요.

> **`--class-weight balanced`를 기본으로 쓰지 않는 이유**
>
> 두 태스크 모두 양성률이 25~50%라 불균형 보정이 필요 없습니다. 그런데 `balanced`를
> 걸면 예측 확률이 위로 밀려 보정이 깨집니다 — 실측에서 로지스틱 Brier가
> **0.181 → 0.240** 으로, 기저율만 내뱉는 `dummy`(0.189)보다도 나빠졌습니다.
> 순위(랭킹)만 쓸 거면 영향이 없지만, `churn_prob` 값을 확률로 읽을 거면 켜지 마세요.

### 표본이 부족할 때

`train`이 "학습 가능한 회차가 없습니다"로 멈추면 `churn_step_clean`이 전부 비어
있다는 뜻입니다. 수집 회차가 적거나 전부 게시 7일 미만(미성숙)인 경우입니다.
교차검증은 그룹(작품 또는 독자)이 2개 이상 필요하고, **작품 수가 곧 유효 표본
수**입니다. 회차를 늘리는 것보다 작품을 늘리는 쪽이 일반화에 훨씬 중요합니다.

---

## 이탈 요인 순위

`train`이 `*_factor_rank.csv`로 내놓는 결과입니다. "이탈률이 무엇에 의해
결정되는가"에 직접 답하기 위한 산출물입니다.

중요도는 **그 요인을 섞었을 때 ROC-AUC가 얼마나 떨어지는가**입니다. 클수록 그
요인 없이는 못 맞힌다는 뜻입니다. `0과구분`은 중요도가 표준편차의 2배를 넘는지
— 즉 우연이 아니라고 말할 수 있는지입니다.

### 회차 이탈 — 작품 298개 · 회차 5,923건 (CV AUC 0.897)

```
 순위          요인          컬럼수     중요도    표준편차  0과구분
  1    직전 회차 성과          8   0.0698   0.0071     O
  2       연재 위치           1   0.0256   0.0045     O
  3  작품 인기도(선작)         1   0.0131   0.0045     O
  4    독자 반응 구성         12   0.0094   0.0036     O
  5      페이월 구조          5   0.0074   0.0022     O
  6   연재 리듬(공백)          2   0.0062   0.0030     O
  7 댓글 감정(KOTE)         19   0.0041   0.0025     -
  8          장르           1   0.0005   0.0007     -
  9    작품 규모·상태         2   0.0002   0.0004     -
 10       회차 분량          1  -0.0002   0.0005     -
```

**직전 회차의 성과가 압도적입니다.** 2위 연재 위치(초반일수록 위험)의 2.7배이고,
나머지를 다 합친 것보다 큽니다. 이탈은 갑자기 오지 않고 **직전 회차에서 이미
징후가 보인다**는 뜻입니다.

### 독자 이탈 — 독자 16,546명 · 등장 89,885건 (CV AUC 0.844)

```
 순위       요인      컬럼수     중요도    표준편차  0과구분
  1 독자 참여 이력      4   0.2655   0.0059     O
  2    연재 위치       2   0.0942   0.0031     O
  3 댓글 작성 양상      8   0.0183   0.0021     O
  4 독자 이탈 징후      1   0.0010   0.0002     O
```

독자가 떠날지는 **그 독자가 지금까지 얼마나 참여했는가**로 거의 결정됩니다.
많이 쓴 독자, 연속으로 따라온 독자는 안 떠납니다. 2위(연재 위치)의 2.8배입니다.

### 작품 간 비교 — 작품 260개 (CV AUC 0.838)

```
 순위          요인       컬럼수     중요도    표준편차  0과구분
  1 댓글 감정(KOTE)      10   0.0415   0.0392     -
  2    독자 반응 구성       6   0.0396   0.0371     -
  3  작품 인기도(선작)      1   0.0277   0.0316     -
  ...  (이하 전부 0 부근)
```

**여기서는 아직 아무것도 유의하지 않습니다.** 모델은 AUC 0.838로 잘 맞히는데
(±0.064), 어느 요인 때문인지 특정할 수 없습니다. 작품 260개에 피처 33개라
개별 요인의 기여를 분리하기에는 여전히 부족합니다. 순위 자체는 참고만 하세요.
**작품 수만이 이걸 풉니다** — 회차나 댓글을 더 모아도 이 표본은 늘지 않습니다.

### 설계에서 다르게 한 두 가지

**그룹 단위로 섞습니다.** 개별 피처를 하나씩 섞으면 `sent_boredom`과
`sent_complaint`처럼 상관이 높은 피처는 서로가 서로를 대신해 **둘 다 "중요하지
않다"** 고 나옵니다. 같은 요인은 함께 섞어야 그 요인 전체의 기여가 보입니다.
표본이 수백 행인데 피처가 수십 개인 상황에서 추정 안정성도 훨씬 낫습니다.

**검증 폴드에서 섞습니다.** 학습 데이터에서 섞으면 모델이 외운 것을 재게 됩니다.
RandomForest는 in-sample AUC가 0.97까지 올라가는데, 그 위에서 잰 순위는
일반화되는 요인의 순위가 아닙니다. 폴드마다 학습은 train으로, 순열과 측정은
test로 합니다. 그래서 `기준성능`이 교차검증 AUC와 같은 값으로 나옵니다.

### 댓글 감정은 이탈 예측에 추가 기여를 하지 않습니다 (검증 완료)

이 프로젝트에서 가장 오래 붙들었던 질문이고, 결론은 **부정적**입니다.
커버리지를 세 단계로 올려가며 같은 검사를 반복했는데 결과가 바뀌지 않았습니다.

| 조건 | 표본 | 감정 커버리지 | 감정 중요도 |
|---|---|---|---|
| 초기 (댓글 대량 수집 전) | 219행 · 작품 10 | 3.3% | -0.0001 |
| 댓글 보유 작품만 | 219행 · 작품 10 | 87.7% | -0.0107 ± 0.0289 |
| **댓글 9.5만 건 수집 후** | **3,664행 · 작품 166** | **94.0%** | **0.0023 ± 0.0066** |

커버리지를 3.3% → 94%로 올려도 0과 구분되지 않습니다.

**중복 가설도 직접 검증했습니다.** 1위 요인인 '직전 회차 성과'에는
`prev_returning_commenter_ratio` 같은 댓글 기반 지표가 들어 있어, 감정이 이미
설명된 부분을 반복하는 것 아니냐는 가설이었습니다. 그 요인을 통째로 빼고 다시 쟀습니다:

```
전체 피처 (52개)          감정 6위 · 0.0023 ± 0.0066   유의하지 않음
직전 회차 성과 제외 (44개)  감정 5위 · 0.0078 ± 0.0086   유의하지 않음  (3.3배 상승)
```

경쟁 요인을 없애면 3.3배 올라가므로 **중복은 실재합니다.** 다만 그것만으로
설명되지 않습니다 — 경쟁자가 사라져도 유의 수준에 못 미칩니다.

### 그렇다고 신호가 없는 것은 아닙니다

단순 상관은 표본 3,664행에서도 부호가 완벽하게 일관됩니다:

```
sent_confusion    +0.106      sent_anticipation  +0.003
sent_churn_index  +0.090      sent_attachment    -0.103
sent_hostility    +0.085      sent_enjoyment     -0.106
sent_complaint    +0.084
sent_boredom      +0.064
```

지루함·불만·적대는 이탈과 같이 가고, 즐거움·애착은 반대로 갑니다.
**감정은 이탈과 실재하는 연관을 갖지만, 다른 요인들이 이미 설명한 것 위에
새로 얹는 정보가 없습니다.**

그래서 KOTE 파이프라인은 남겨둡니다 — 감정 자체를 보는 분석(어떤 회차가 어떤
반응을 얻었나, 고정팬과 신규 독자의 감정이 다른가)에는 그대로 쓸 수 있습니다.
다만 **이탈 예측 피처로는 기대하지 마세요.**

### 표본을 늘리는 방법

요인 순위를 신뢰하려면 **작품 수**를 늘려야 합니다 — 회차가 아니라 작품입니다.
`--task novel`은 아예 작품이 표본이고, `episode`도 교차검증을 작품 단위로
나누므로 유효 표본이 작품 수에 묶여 있습니다.

```bash
# 요인 순위용 대량 수집 — 초반 30화만, 댓글 없이
python -m munpia.cli discover --genres FANTASY,NEWFANTASY,HEROISM,HISTORY,ROMANCE,SPORTS,GAME,MYSTERY,LIGHTNOVEL,FUSION,DRAMA --limit 400
python -m munpia.cli crawl --id-file data/raw/novel_ids.txt \
    --entry-detail --max-episodes 30 --comment-scope none
```

작품당 약 45초입니다. 실측으로 386개 중 385개를 약 4시간에 수집했습니다.
댓글을 빼는 이유는 댓글 페이지네이션이 전체 시간의 대부분을 차지하기 때문입니다.

표본이 늘면 요인 순위가 이렇게 달라집니다:

| | 작품 10개 | 작품 298개 |
|---|---|---|
| 0과 구분되는 요인 (회차 이탈) | **0개** | **6개** |
| 1위 요인 | 0.045 ± 0.051 | 0.070 ± 0.007 |
| CV AUC | 0.785 | 0.897 |

독자 단위는 더 극적입니다. 등장 6,161건 → **89,885건**이 되면서 4개 요인이
전부 유의해졌고, 1위 요인의 오차가 ±0.019 → **±0.006**으로 줄었습니다.

**단, 작품 간 비교(`--task novel`)만은 아무리 댓글을 모아도 나아지지 않습니다.**
거기서는 작품 하나가 표본 하나이기 때문입니다. 260개로는 부족하고, 이 프로젝트가
확보할 수 있는 상한(`discover` API가 약 386개)에 이미 닿아 있습니다.

---

## 예외 처리 · 크롤링 매너 (요구사항 4)

- **랜덤 딜레이** — 요청마다 `min_delay`~`max_delay` 사이 랜덤 대기. 직전 요청 경과
  시간을 빼서 실제 간격을 보장합니다.
- **지수 백오프 재시도** — 429·5xx·네트워크 오류는 `2^n + jitter` 초 후 재시도.
  애플리케이션 레벨 오류(권한 없음 등)는 재시도해도 결과가 같으므로 즉시 포기합니다.
- **회차 단위 격리** — 한 회차의 파싱 실패가 작품 전체를, 한 작품의 실패가 배치 전체를
  중단시키지 않습니다. 실패는 `_errors.log`에 남고 다음 대상으로 넘어갑니다.
- **권한 오류 회로차단기** — 유료 회차에서 권한 오류가 5회 연속되면 그 작품의 댓글
  수집을 포기합니다. 유료 연재는 회차 대부분이 막혀 있어 계속 두드리면 요청만 낭비됩니다.
- **증분 저장 + 재개** — 결과를 작품 단위로 즉시 append 하고 완료 ID를 기록합니다.
  중단 후 같은 명령을 다시 실행하면 남은 작품부터 이어갑니다.

문피아 `robots.txt`가 막은 `/novel/viewer/`는 건드리지 않습니다. 기본 딜레이는 사람이
읽는 속도보다 느리게 잡아뒀으니 굳이 더 줄이지 마세요.

---

## 구조

```
munpia/
├── client.py       HTTP 세션 · 레이트리밋 · 재시도 · 에러코드 매핑
├── schema.py       레코드 정의 + API 응답 → 레코드 정규화
├── preprocess.py   텍스트 정제 · 숫자 정수화 · 일시 파싱
├── crawler.py      작품→회차→댓글 순회 오케스트레이션
├── storage.py      3테이블 증분 저장 · 재개
├── features.py     이탈률 / 팬층 / 반응 구성비 파생 지표
├── sentiment.py    KOTE 댓글 감정 — 44개 라벨 → 9개 축 → 회차별 집계
├── model.py        이탈 예측 — 학습셋 구성 · 누수 차단 · 교차검증 · 요인 순위
├── auth.py         .env 폼 로그인 · 브라우저 로그인 · 세션 진단
├── wizard.py       대화형 실행 마법사 (run.bat / run.sh 가 호출)
└── cli.py          명령줄 진입점

run.bat / run.sh    원클릭 런처 (Python 확인 → venv → 패키지 설치 → 실행)
tests/              77개 테스트 (전처리 13 · 저장 6 · 인증 8 · 마법사 11
                                 · 피처 11 · 감정 9 · 모델 19)
scripts/explore.py  수집 결과 요약 리포트
```

테스트 실행:

```bash
python -m tests.run_all
```

네트워크 없이 도는 테스트입니다. 전처리 규칙, CSV/JSONL 저장, `.env` 파싱,
대화형 입력 처리, 이탈률 계산의 경계 조건, 학습셋의 라벨 누수를 검증합니다.
`scikit-learn`이 없으면 모델 테스트는 자동으로 건너뜁니다.

---

## 분석 예시

```python
import pandas as pd

ep = pd.read_csv("data/features/episode_features.csv")
users = pd.read_csv("data/features/user_features.csv")

# 이탈이 가장 큰 회차 — 페이월·신작 노이즈를 뺀 값으로 본다
worst = ep.dropna(subset=["churn_step_clean"]).nlargest(10, "churn_step_clean")[
    ["novel_id", "episode_num", "title", "churn_step_clean", "view_count", "days_since_prev"]]

# 초반 이탈 (웹소설에서 가장 중요한 구간)
early = ep[(ep.episode_num <= 10) & (ep.is_mature == 1)]

# 고정 팬층이 두꺼운 작품
core = (ep.groupby("novel_id")["returning_commenter_ratio"]
          .mean().sort_values(ascending=False))

# 감정 분석 입력 (KOTE 등) — 스티커·빈 댓글 제외
cm = pd.read_csv("data/raw/comments.csv")
texts = cm[(cm.content_type == "TEXT") & (cm.body_char_len > 0)]

# 핵심 팬의 반응만 따로
fan_keys = set(users[users.is_core_fan == 1].user_key)
fan_texts = texts[texts.user_key.isin(fan_keys)]

# 여러 작품에 걸쳐 활동하는 독자 — user_key가 blogUrl 기반이라 작품 간 추적이 된다
cross = (cm.groupby("user_key")
           .agg(novels=("novel_id", "nunique"), comments=("comment_id", "count"))
           .query("novels >= 3").sort_values("novels", ascending=False))
```

모델을 코드에서 직접 쓸 때:

```python
from munpia.model import (build_episode_training_frame, cross_validate,
                          fit_final, explain, predict_frame)

ep = pd.read_csv("data/features/episode_features.csv")
frame = build_episode_training_frame(ep, label_mode="quantile", threshold=0.75)

print(cross_validate(frame))          # 모델 비교 — dummy가 하한선
pipe = fit_final(frame, "logistic")
print(explain(pipe, frame).head(10))  # 무엇이 이탈을 끌어올리는가 (오즈비)

risky = predict_frame(pipe, frame).head(20)   # 이탈 확률 상위 회차
```

학습된 모델을 새 데이터에 적용:

```python
import joblib
pipe = joblib.load("data/models/episode_churn_model.joblib")
prob = pipe.predict_proba(frame.X)[:, 1]
```

---

## 알려진 제약

1. **유료 회차 댓글은 구매한 회차만 열립니다.** 문피아가 회차당 과금(100골드) 구조라
   유료 계정이어도 전량 수집은 되지 않습니다 (위 💰 항목 참고). 구매하지 않은 회차는
   자동으로 건너뛰므로 낭비는 없지만, 댓글 볼륨의 상한이 곧 구매량입니다.
2. **조회수는 누적값입니다.** 방금 올라온 회차는 조회수가 낮아 이탈률이 과대 계상되고,
   무료→유료 전환 지점에서도 급락합니다. `churn_step_clean`이 두 노이즈를 모두
   제거한 값이니 학습에는 그쪽을 쓰세요 (위 ⚠️ 항목 참고).
3. **무료연재는 댓글 밀도가 낮습니다.** 회차당 0~2건 수준이라 회차 단위 감정 분석에는
   표본이 부족할 수 있습니다. 작품 단위로 묶어서 보거나 유료작 로그인 수집을 권합니다.
4. **`char_estimate`는 추정치입니다.** `pages × 900`이라 절대값이 아니라 회차 간
   상대 비교에만 쓰세요.
5. **비공개·계약종료 작품이 섞여 있습니다.** 사이트맵 ID 상당수가 `A002_12002`로
   응답합니다. 정상이며 `_errors.log`에 기록되고 건너뜁니다.
6. **모델의 유효 표본은 회차 수가 아니라 작품 수입니다.** 교차검증을 작품 단위로
   나누기 때문에, 25개 작품이면 사실상 25개 표본으로 일반화를 평가하는 셈입니다.
   회차를 더 모으는 것보다 작품 수를 늘리는 쪽이 훨씬 중요합니다.
   실측으로 작품 10개 → 298개에서 0과 구분되는 요인이 0개 → 5개가 됐습니다.
   그럼에도 **작품 간 비교(`--task novel`)는 260개로도 아직 부족**합니다 —
   모델은 AUC 0.849로 맞히는데 어느 요인 때문인지 특정되지 않습니다.
7. **독자 단위 이탈은 댓글을 쓴 독자만 보입니다.** 조용히 읽다 떠나는 대다수는
   관측되지 않으므로, `user` task의 이탈률을 전체 독자의 이탈률로 읽으면 안 됩니다.
   회차 단위(`episode`)와 함께 봐야 합니다.
8. **⚠️ 유료 회차의 `view_count`는 무료 회차와 다른 것을 셉니다.** 실측 10개 작품
   전부에서, 페이월 직후 조회수가 직전의 **4.5%** 로 떨어지는 동안 추천수는 93%,
   댓글수는 100%가 유지됐습니다. 독자가 95% 사라졌다면 추천도 같이 사라져야 합니다.
   그렇지 않다는 것은 세는 대상이 바뀌었다는 뜻입니다. 결과적으로

   - 유료 구간의 평균 이탈률이 **음수**로 나오는 작품이 있고, 회차의 24.5%가
     "조회수 증가"로 잡힙니다
   - `like_per_view` 같은 비율 피처는 페이월에서 **34~49배 점프**합니다

   그래서 이탈 분석은 `--max-episode 25`로 **무료 구간에 한정하는 것을 권장**합니다.
   손실은 크지 않습니다 — 초반 이탈이 압도적이기 때문입니다:

   ```
   1-5화   0.0892      26-50화  -0.0100
   6-25화  0.0192      51-100화 -0.0026
   ```

   전 구간을 봐야 한다면 `features --churn-basis like`로 이탈 기준을 추천수로
   바꿀 수 있습니다. 다만 **기본 권장이 아닙니다** — 아래 측정 결과를 보세요.
9. **댓글이 무료 구간에만 있습니다.** 팬층 지표와 감정 피처의 관측 회차번호가
   최대 25입니다. 유료 구간(26화 이상)에는 **0행**입니다. 회차의 88.6%가 유료라
   내용 신호가 거기서 통째로 빕니다. 로그인 + `--comment-scope all`로 일부
   완화되지만, 구매한 회차만 열리므로 근본 해결은 아닙니다.

   > 참고로 **유료 구간에서 댓글이 오히려 늘어납니다** — 무료 구간 49.4건/회차
   > → 유료 26~50화 84.6건/회차(작품 469259). 목록 API가 비로그인에도 댓글
   > *개수*는 알려주므로 확인된 값입니다. 독자가 떠난 게 아니라는 또 하나의 증거입니다.
10. **로그인해도 조회수는 같은 값이 보입니다.** "유료 회차 조회수가 낮은 건
   비로그인이라 안 보이는 것 아닌가"를 확인했는데, 같은 회차를 로그인 상태로
   다시 받아 비교하니 무료·유료 모두 1.000~1.005배였습니다(자연 증가분).
   페이월에서의 조회수 붕괴는 **가시성 문제가 아니라 서버가 유료 회차에 대해
   애초에 다른 지표를 반환하는 것**입니다.



