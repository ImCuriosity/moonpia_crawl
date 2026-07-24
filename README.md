# 문피아 웹소설 데이터 수집 · 전처리 파이프라인

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey">
  <img alt="Tests" src="https://img.shields.io/badge/tests-38%20passing-brightgreen">
  <img alt="Status" src="https://img.shields.io/badge/status-verified%20on%20live%20site-success">
</p>

웹소설 **독자 이탈률 추정**과 **고정 팬층 반응 분석**을 위한 ML 학습 데이터셋을
만드는 도구입니다. 작품 메타데이터 · 회차별 시계열 지표 · 독자 댓글을 수집해
정수화·구조화된 CSV/JSON으로 내보내고, 이탈률과 충성도 파생 지표까지 생성합니다.

**실측 검증 완료** (2026-07-24, 25개 작품 대상):

```
작품 26건 · 회차 4,576건 · 댓글 10,110건 · 고유 댓글 작성자 3,069명
텍스트 댓글 8,034건 / 스티커 2,076건 · 수집 오류 0건
조인 무결성 100% (고아 레코드 0건, 중복 0건)
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
바로 시작됩니다. 질문에 답하기만 하면 수집부터 피처 생성까지 끝납니다.

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

### 4. (선택) 유료 회차 댓글용 로그인

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
├── features.py     이탈률 / 팬층 파생 지표
├── auth.py         .env 폼 로그인 · 브라우저 로그인 · 세션 진단
└── cli.py          명령줄 진입점

├── wizard.py       대화형 실행 마법사 (run.bat / run.sh 가 호출)
└── cli.py          명령줄 진입점

run.bat / run.sh    원클릭 런처 (Python 확인 → venv → 패키지 설치 → 실행)
tests/              38개 테스트 (전처리 13 · 저장 6 · 인증 8 · 마법사 11)
scripts/explore.py  수집 결과 요약 리포트
```

테스트 실행:

```bash
python -m tests.run_all
```

네트워크 없이 도는 테스트입니다. 전처리 규칙, CSV/JSONL 저장, `.env` 파싱,
대화형 입력 처리를 검증합니다.

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
