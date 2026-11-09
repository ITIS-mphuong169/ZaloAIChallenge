# Zalo AI Challenge 2026 — DataMate và LifeMate

Tài liệu chung cho cả hai bài: cách làm, cài đặt, cách chạy, Docker, huấn luyện và nộp bài.
Cập nhật ngày 09/11/2026.

Mỗi bước trong tài liệu mang một trong ba nhãn:

- **[Đã kiểm chứng]** đã chạy trên máy phát triển, kết quả ghi kèm.
- **[Chưa chạy thử]** đã viết xong nhưng chưa chạy được, vì máy phát triển không có GPU và không có Docker.
- **[Chưa làm]** mới là kế hoạch.

Mục lục:

1. [Tổng quan hai bài](#1-tổng-quan-hai-bài)
2. [Thư mục và máy](#2-thư-mục-và-máy)
3. [Chuẩn bị server](#3-chuẩn-bị-server-làm-một-lần)
4. [DataMate](#4-datamate)
5. [LifeMate](#5-lifemate)
6. [Đóng gói Docker để nộp](#6-đóng-gói-docker-để-nộp)
7. [Hạn và quy tắc nộp bài](#7-hạn-và-quy-tắc-nộp-bài)
8. [Sự cố đã gặp](#8-sự-cố-đã-gặp)
9. [Câu hỏi còn mở](#9-câu-hỏi-còn-mở)

---

## 1. Tổng quan hai bài

| | DataMate | LifeMate |
|---|---|---|
| Bài toán | Trả lời câu hỏi tiếng Việt của chủ shop trên sổ sách của shop | Trợ lý giọng nói: nghe yêu cầu rồi gọi đúng lệnh của các app |
| Đầu vào | Câu hỏi dạng chữ, file Excel / CSV / SQLite của shop | File âm thanh, lịch sử hội thoại, danh sách app đã cài |
| Đầu ra | Một đáp án cho mỗi câu (số, %, ngày, tên, danh sách, bảng) | Chuỗi lệnh gọi tool và mã trạng thái cuối |
| Cách chấm | Đúng +1, sai −1, bỏ trống 0 | So trạng thái cuối với đáp án, và F1 trên các lệnh gọi |
| Cách làm | Chương trình luật cố định, tính bằng pandas | Mô hình nghe → mô hình ngôn ngữ → thư viện tool của ban tổ chức |
| Mô hình | Không có (0 tham số) | Bắt buộc có: một mô hình nghe và một mô hình ngôn ngữ |
| GPU | Không cần | Cần |
| Huấn luyện | Không có | Cần tinh chỉnh mô hình nghe; mô hình ngôn ngữ thì tuỳ kết quả |
| Kết quả | Train 3.000/3.000; public 1,0 | Chưa có file nộp |
| Còn thiếu | Build và chạy thử Docker thật (mục 6.2) | Mọi bước cần GPU; Docker chờ có mô hình (mục 6.3) |

Vì sao hai bài làm khác nhau: ở DataMate câu hỏi là chữ và mọi phép tính đều có luật rõ trong rulebook, nên viết luật là chính xác nhất.
Ở LifeMate đầu vào là âm thanh, và 11 app của tập public không hề có trong tập train (410/634 mẫu public có cài ít nhất một app lạ), nên không thể viết luật trước cho từng app.

---

## 2. Thư mục và máy

```text
ZALOAI/
├── README.md                 tài liệu này
├── DataMate/
│   ├── datamate/             mã nguồn (khoảng 6.000 dòng Python)
│   ├── dataset/              train/ và public_test/ của ban tổ chức
│   ├── result/               các file đã nộp; submission_v7.jsonl là bản được 1,0
│   ├── Dockerfile, predict.sh, requirements.txt, .dockerignore     bộ file Docker
│   ├── pyproject.toml, uv.lock
│   └── README.md             bản nộp kèm lời giải (có mục Models)
└── LifeMate/
    ├── lifemate/             mã nguồn
    ├── Dataset/              train+public_test.zip và bản đã giải nén
    ├── result/               lời thoại đã nhận dạng, file dự đoán
    ├── models/               mô hình tải về (không cần chép sang server)
    ├── Dockerfile, predict.sh, .dockerignore                       bộ file Docker
    ├── pyproject.toml
    └── README.md             ghi chú ngắn riêng của LifeMate
```

| Máy | Cấu hình | Dùng cho |
|---|---|---|
| Máy phát triển | Intel i3-14100 (4 nhân), 15 GB RAM, không GPU, không Docker | Viết code, chạy DataMate, thử nhỏ LifeMate |
| Server | RTX 3090 (24 GB) | Mọi bước cần GPU của LifeMate, và thử Docker cho cả hai bài |

Khi chép sang server:

- **Cần chép:** `README.md`, `DataMate/` (trừ `.venv`), `LifeMate/lifemate/`, `LifeMate/Dockerfile`, `LifeMate/predict.sh`, `LifeMate/.dockerignore`, `LifeMate/pyproject.toml`, `LifeMate/README.md`, `LifeMate/result/`, `LifeMate/Dataset/train+public_test.zip`.
- **Không cần chép:** mọi thư mục `.venv` (không dùng lại được trên máy khác), `LifeMate/models/` (6,9 GB, tải lại trên server nhanh hơn), `LifeMate/tools/`, `LifeMate/Dataset/extracted/` (giải nén lại).

Ví dụ một lệnh chép (thay `user@server`):

```bash
rsync -av --exclude .venv --exclude 'LifeMate/models' --exclude 'LifeMate/tools' --exclude 'LifeMate/Dataset/extracted' \
  ~/Documents/ZALOAI/ user@server:~/ZALOAI/
```

---

## 3. Chuẩn bị server (làm một lần)

**[Chưa chạy thử]** Các lệnh dưới đây là lệnh cài đặt chuẩn của từng công cụ, chưa chạy trên server này.

### 3.1. Kiểm tra

```bash
nvidia-smi            # phải thấy RTX 3090 và phiên bản driver
uv --version          # trình quản lý môi trường Python
docker --version      # cần cho bước Docker
df -h ~               # cần khoảng 80 GB trống: mô hình, môi trường, image Docker
```

### 3.2. Cài uv (nếu chưa có)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

uv tự tải Python 3.12 khi tạo môi trường, không cần cài Python riêng.

### 3.3. Cài Docker (nếu chưa có)

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"      # đăng xuất rồi đăng nhập lại để có hiệu lực
docker run --rm hello-world
```

Riêng LifeMate cần GPU bên trong Docker, tức là cần thêm NVIDIA Container Toolkit.
Cài theo hướng dẫn chính thức của NVIDIA (tìm "NVIDIA Container Toolkit install guide"), rồi kiểm tra:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Lệnh này phải in ra đúng bảng như `nvidia-smi` ở ngoài. DataMate không cần bước này.

### 3.4. Làm việc với Claude Code trên server

Mở thư mục `ZALOAI` trên server bằng VSCode Remote-SSH rồi bật Claude Code.
Phiên Claude trên server là phiên mới, không biết các cuộc trò chuyện trước, nên câu đầu tiên là: "đọc README.md rồi làm tiếp".

---

## 4. DataMate

### 4.1. Bài toán

Mỗi shop là một thư mục chứa file Excel / CSV / SQLite (đơn bán, nhập hàng, công nợ ...) và một `README.md` giải thích các cột.
Mỗi câu hỏi gồm mã câu, mã shop, câu hỏi tiếng Việt và `as_of_date` (ngày được coi là "hôm nay").
Kèm theo có `calendar.csv` (lịch âm, ngày lễ) và `rulebook.md` (luật tính của ban tổ chức).

- Train: 30 shop, 3.000 câu, có đáp án. Public: 14 shop, 1.000 câu, không có đáp án.
- Đáp án thuộc một trong 8 kiểu: `number`, `percent`, `date`, `boolean`, `string`, `list`, `table`, `unanswerable`. Có thể bỏ trống bằng `abstain`.
- Chấm: đúng +1, sai −1, bỏ trống 0. Điểm = (số đúng − số sai) / tổng số câu.
- Phải khớp tuyệt đối: tiền đúng từng đồng, phần trăm đúng 2 chữ số thập phân, sai định dạng là sai.

### 4.2. Cách làm

Không dùng mô hình nào. Toàn bộ là chương trình xác định: cùng dữ liệu vào thì luôn ra cùng đáp án.

| Bước | File | Việc làm |
|---|---|---|
| 1. Đọc dữ liệu | `raw.py`, `readme.py` | Đọc mọi file / sheet / bảng, tìm dòng tiêu đề, gộp các sheet cùng cấu trúc, đọc ý nghĩa cột trong README của shop |
| 2. Chuẩn hoá | `textnorm.py` | Số (`560,000`, `25.000đ`, `25k`), ngày (`01/02/2025`, số serial Excel, `1 thg 6 2025`) |
| 3. Dựng bảng bán hàng | `model.py` | Nhận diện vai trò từng cột, dựng bảng đơn và bảng dòng hàng, phân loại đơn (hoàn thành / huỷ / trả / đang giao), bỏ dòng tổng phụ và dòng trùng |
| 4. Danh mục | `enrich.py` | Danh mục hàng, nhóm hàng, nhóm khách, quy đổi về đơn vị cơ bản, gộp các cách viết tên |
| 5. Sổ phụ | `imports.py`, `debts.py`, `lots.py`, `ops.py` | Phiếu nhập, công nợ, lô và hạn dùng, giao hàng, bảng giá, chi phí |
| 6. Hiểu câu hỏi | `question.py` | Bỏ dấu, sửa lỗi telex, viết tắt và teencode, xác định kỳ thời gian theo `as_of_date` |
| 7. Trả lời | `engine.py`, `extra.py` | Nhận diện thực thể (mặt hàng, nhóm, kênh ...) theo từ điển riêng của shop, chọn ý định, tính theo rulebook, định dạng đáp án |
| 8. Theo ngành | `phone.py`, `baby.py`, `repair.py` | Cửa hàng điện thoại, mẹ và bé, tiệm sửa xe |
| Lịch | `calendar.py` | Lịch âm dương từ `calendar.csv` (câu hỏi về Tết) |
| Chấm | `score.py` | Chấm theo đúng luật so khớp của ban tổ chức |

Thứ tự xử lý một câu hỏi trong `engine.py`: các bộ xử lý chuyên ngành được thử trước (điện thoại, mẹ và bé, công nợ, kho, sửa xe), sau đó mới đến bộ xử lý chung về doanh thu, số đơn, xếp hạng.

Nguyên tắc an toàn, vì sai bị trừ điểm:

- Trong câu hỏi còn từ nào chương trình không hiểu thì trả `abstain`, không đoán.
- Trước khi bỏ cuộc, chương trình thử sửa chính tả các từ lạ (dính chữ, lẫn l/n, d/gi/v, sai một chữ cái). Chỉ nhận đáp án khi mọi cách sửa đều cho cùng một kết quả.
- Dữ liệu của shop không đủ để trả lời (kỳ nằm ngoài khoảng dữ liệu, shop không ghi loại thông tin đó) thì trả `unanswerable`. Đây là một đáp án thật và được chấm điểm.

### 4.3. Cài đặt và chạy

**[Đã kiểm chứng]** trên máy phát triển, Python 3.12.3, uv 0.11.19.

```bash
cd DataMate
uv sync          # tạo .venv theo uv.lock: pandas 3.1.1, numpy 2.5.4, python-calamine 0.9.1

# Chấm trên train (cần answers.jsonl). --show wrong in ra các câu sai.
uv run python -m datamate.cli eval --data dataset/train --show wrong

# Tạo file nộp cho public test
uv run python -m datamate.cli predict --data dataset/public_test \
    --out result/submission.jsonl --log result/run_log.jsonl
```

Kết quả chạy lại ngày 06/11/2026:

| Lệnh | Kết quả | Thời gian | RAM cao nhất |
|---|---|---|---|
| `eval` trên train | 3.000 đúng, 0 sai, 0 bỏ trống, điểm 1,0 | 290 giây | 3,3 GB |
| `predict` trên public | 1.000 câu có đáp án, 0 bỏ trống; giống hệt từng byte với `result/submission_v7.jsonl` | 196 giây | 1,6 GB |

`run_log.jsonl` ghi câu hỏi, đáp án và lý do của từng câu `abstain`. Đây là file để đọc khi cần biết vì sao một câu bị bỏ trống.

### 4.4. Kiểm chứng trước khi nộp

Mỗi lần sửa code phải làm đủ ba việc:

1. `eval` trên train phải giữ nguyên 0 câu sai.
2. `predict` trên public, rồi so với file đã có điểm: `cmp result/submission.jsonl result/submission_v7.jsonl`. Câu nào khác thì phải giải thích được.
3. So chéo với các file cũ đã biết số câu sai: v1 sai 50, v2 sai 7, v3 sai 10, v6 sai 9. Chấm file cũ với file mới làm "đáp án" bằng `datamate.score.score_one`. Nếu file mới đúng hết thì số câu lệch phải ra đúng các con số đó, sau khi bỏ các chỗ chỉ khác cách ghi (ví dụ ghi mã thay cho tên).

### 4.5. Docker

Bộ file gồm `Dockerfile`, `predict.sh`, `requirements.txt` và `.dockerignore`. Các bước build, chạy thử và gửi nằm ở mục 6.2.

Thiết kế:

- Image nền `python:3.12-slim`. Chỉ cài 5 thư viện, ghi cứng phiên bản trong `requirements.txt` đúng như bản đã chấm 3.000/3.000: `numpy==2.5.4`, `pandas==3.1.1`, `python-calamine==0.9.1`, `python-dateutil==4.0.0`, `tzdata==2026.6`.
- Trong image chỉ có thư mục `datamate/` và `predict.sh`. `.dockerignore` giữ dữ liệu, kết quả và `.venv` ở ngoài.
- Ban tổ chức gắn tập test ở `/data` (chỉ đọc) và thư mục kết quả ở `/result`.
- `predict.sh` chạy `python -m datamate.cli predict --data /data --out /result/submission.jsonl --log /result/logs/run_log.jsonl`.
- Không cần mạng lúc chạy, không cần GPU. Một câu lỗi không làm dừng cả lượt: câu đó thành `abstain`.

**[Chưa chạy thử]** Image chưa được build bằng Docker thật, vì máy phát triển không có Docker.

**[Đã kiểm chứng]** Giả lập trên máy phát triển ngày 09/11/2026, sát với lúc chấm nhất có thể khi không có Docker:

- môi trường Python mới, chỉ cài đúng `requirements.txt`;
- thư mục chương trình chỉ có `datamate/` và `predict.sh`, như trong image;
- không có mạng; biến môi trường chỉ gồm các biến ghi trong `Dockerfile`, cộng hai biến chỉ đường dẫn dữ liệu và kết quả.

| Tập gắn vào `/data` | Kết quả | Thời gian | RAM cao nhất |
|---|---|---|---|
| public test | 1.000 câu có đáp án; giống hệt từng byte với `result/submission_v7.jsonl`; thư mục dữ liệu không bị ghi gì | 160 giây | 1,6 GB |
| train | 3.000 câu có đáp án; chấm file sinh ra được 3.000 đúng, 0 sai, 0 bỏ trống | 295 giây | 3,2 GB |

Giả lập không kiểm được hai thứ: bản thân image nền `python:3.12-slim` và lệnh `docker build`.

### 4.6. Huấn luyện

Không có bước huấn luyện, vì không có mô hình.
Tập train chỉ dùng để viết luật và để kiểm chứng (mục 4.4).

Tập private có thể có shop mới và cách hỏi mới. Luật viết tay gặp cách hỏi chưa thấy thì trả `abstain`, tức mất điểm câu đó chứ không bị trừ.
Sau khi nộp Docker thì không sửa được nữa, nên mọi cải tiến phải xong trước hạn.

### 4.7. Nộp bài

- Nộp ở hạng **SLM** (tổng tham số ≤ 15 tỉ; lời giải này có 0 tham số).
- Mô tả bài nộp, bằng tiếng Anh:

```text
Rule-based program. No machine learning model (0 parameters), CPU only, no internet.

1. Read every file and sheet of each shop, find the header row, and use the shop's README to learn what each column means.
2. Clean numbers and dates, then build one table of orders and one table of order lines.
3. Clean the question (missing accents, telex typing, slang, typos) and work out the time period from as_of_date.
4. Match the question to a known question type and compute the answer with pandas, following the rulebook.
5. Leave a question blank when it is not fully understood, instead of guessing.
```

Lịch sử điểm public: 0,363 → 0,467 → 0,653 → 0,972 → **1,0** (file `result/submission_v7.jsonl`, nộp ngày 06/11/2026).

### 4.8. Việc còn lại

1. Build và chạy thử bằng Docker thật theo mục 6.2. Hạn nộp Docker là 23:59 ngày 16/11/2026.
2. Đối chiếu hướng dẫn Docker của ban tổ chức (mục 9) rồi gửi image theo mục 6.4.

---

## 5. LifeMate

### 5.1. Bài toán

Mỗi mẫu gồm:

- `audio`: file wav 16 kHz của yêu cầu cuối, bằng tiếng Việt. Đây là yêu cầu được chấm.
- `history`: các lượt trước đó. Lượt của người dùng là âm thanh; lượt của trợ lý có lời đáp, các lệnh đã gọi và kết quả của chúng.
- `context`: giờ hiện tại, múi giờ, vị trí, hồ sơ người dùng (tên, nhà, nơi làm việc, thành phố), và các app đã cài kèm danh sách tool.
- `state.json`: dữ liệu của từng app tại thời điểm đó. Tác nhân không đọc file này trực tiếp mà xem qua tool.

Việc phải làm: gọi đúng chuỗi tool qua thư viện `zaic_tools` của ban tổ chức, rồi ghi vào `submission.jsonl` hai thứ: `tool_calls` (mọi lệnh đã gọi) và `state_hash` (mã của trạng thái cuối).

Ví dụ từ train. Yêu cầu: "Nếu mai mưa thì dời chạy bộ muộn 1 tiếng, nhắn Sếp Dũng giờ mới; không mưa thì thôi." Đáp án:

```text
weather.get_forecast   {"date": "2026-11-29"}                        -> mai có mưa
calendar.list_events   {}                                            -> tìm ra evt_01 "chạy bộ" lúc 08:00
calendar.update_event  {"event_id": "evt_01", "start": "...T09:00:00+07:00", "end": "...T10:00:00+07:00"}
contacts.search        {"query": "Sếp Dũng"}                         -> tìm ra c_08
sms.send               {"contact_id": "c_08", "content": "mai chạy bộ dời sang 09:00 nhé"}
```

Số liệu của gói dữ liệu (bản v3.2.0):

| | Train | Public test |
|---|---|---|
| Số mẫu | 1.545 | 634 |
| Có lời thoại và đáp án | Có | Không, chỉ có âm thanh |
| Số clip âm thanh | 5.036 (410 phút) | 1.933 (166 phút) |
| Số app / số tool | 46 / 147 | 36 / 122 |
| App không có ở tập kia | 21 | 11 |

Các dạng đáp án trong train: 1.093 mẫu có lệnh ghi, 124 mẫu chỉ đọc, 217 mẫu kết thúc bằng hỏi lại người dùng (`sys.ask_missing_info`), 111 mẫu không gọi lệnh nào. Đáp án dài nhất có 6 lệnh.

Cách chấm: ban tổ chức chạy lại `tool_calls` trên `state.json` ban đầu bằng chính `zaic_tools`, so trạng thái với đáp án và tính F1 trên các lệnh gọi.

- Khi so trạng thái, các trường văn bản tự do (nội dung tin nhắn, tiêu đề, ghi chú, địa chỉ ...) bị bỏ ra; chúng được chấm gần đúng trên tham số của lệnh gọi.
- Dữ liệu của app thông tin (thời tiết, giá vàng, giá xăng ...) không nằm trong trạng thái được so.
- Công thức gộp hai phần không có trong gói dữ liệu.

Nếu không làm gì cả (không gọi lệnh nào), trên train sẽ khớp trạng thái 24,5% số mẫu và F1 lệnh gọi là 0,072. Đây là mức sàn để so sánh.

### 5.2. Cách làm

```text
file wav ──► mô hình nghe ──► chữ ──► mô hình ngôn ngữ ──► lệnh gọi ──► zaic_tools ──► kết quả
                                            ▲                                            │
                                            └──────────── đọc kết quả, gọi tiếp ─────────┘
```

**Mô hình nghe** ([asr.py](LifeMate/lifemate/asr.py)). Chuyển mọi clip (yêu cầu cuối và các lượt người dùng trong lịch sử) thành chữ.
Lời thoại được lưu đệm trong một file JSONL nên dừng rồi chạy tiếp được, và không clip nào phải nhận dạng hai lần.

**Lời nhắc cho mô hình ngôn ngữ** ([prompt.py](LifeMate/lifemate/prompt.py)) gồm:

- Luật làm việc: chỉ dùng tool của app đã cài; app cần dùng mà chưa cài thì không gọi gì; lấy ID bằng lệnh đọc, không tự đặt; thiếu thông tin hoặc có nhiều kết quả trùng tên thì làm phần làm được rồi gọi `sys.ask_missing_info`; câu điều kiện thì kiểm tra điều kiện rồi mới làm; người dùng tự sửa lời thì theo ý cuối.
- Giờ hiện tại kèm thứ trong tuần, và lịch bốn tuần (tuần trước, tuần này, tuần sau, tuần sau nữa) để mô hình không phải tự tính "thứ Ba tuần sau" là ngày nào.
- Hồ sơ người dùng.
- Mô tả từng tool của các app đã cài, viết gọn từ catalog (tham số, kiểu, giá trị cho phép, tham số bắt buộc).
- Vài mẫu train đã giải có câu chữ giống yêu cầu hiện tại, làm ví dụ ([retrieve.py](LifeMate/lifemate/retrieve.py), chọn bằng TF-IDF nên không cần mô hình).
- Lịch sử hội thoại, kể cả lệnh đã gọi và kết quả.

**Giao thức.** Mỗi lượt mô hình chỉ trả một đối tượng JSON, nên dùng được với mọi mô hình trò chuyện và mọi máy chủ:

```text
{"call": "<app.tool>", "args": {...}}      gọi một tool; lượt sau nhận "RESULT {...}"
{"reply": "<câu trả lời ngắn>"}            xong việc
```

**Vòng lặp gọi tool** ([agent.py](LifeMate/lifemate/agent.py)) có ba rào chắn, vì mọi lệnh đã gọi đều tính vào bài nộp:

- Kiểm tra schema trước khi gọi. Lệnh sai schema không được gửi đi mà báo lỗi lại cho mô hình.
- Lệnh đọc đã gọi rồi thì trả lại kết quả cũ, không gọi lần hai.
- Dừng ngay sau `sys.ask_missing_info`, và dừng sau tối đa 12 bước.

**Quy ước rút ra từ đáp án train** (đã đo trên cả 1.545 mẫu):

- ID trong lệnh ghi luôn lấy từ một lệnh đọc trước đó trong cùng đáp án (964 lần) hoặc từ lịch sử hội thoại (313 lần); chỉ 1 lần không thuộc hai nguồn này.
- Thông tin đã có trong lịch sử thì không đọc lại. Ví dụ đã hỏi thời tiết ngày mai ở lượt trước thì câu "nếu mai mưa ..." không gọi lại `weather.get_forecast`.
- `sys.ask_missing_info` luôn là lệnh cuối (217/217 mẫu).
- Lệnh đọc thường gọi không kèm bộ lọc (`calendar.list_events {}`: 87 lần, có khoảng ngày: 18 lần).

**Đã kiểm chứng** trên máy phát triển:

| Việc | Kết quả |
|---|---|
| Chạy lại 1.545 đáp án train qua `zaic_tools` | 0 kết quả tool lệch, 0 trạng thái cuối lệch |
| Chạy lại 3 dòng trong file nộp mẫu của ban tổ chức | `state_hash` khớp cả 3 |
| Vòng lặp với "mô hình giả" đọc lại đáp án train, 8 luồng song song | 1.545/1.545, chạy tiếp sau khi dừng cũng đúng |
| Dựng lời nhắc cho 634 mẫu public | Không lỗi. Lời nhắc dài khoảng 12.000 ký tự; trên train, kèm 4 ví dụ, khoảng 15.000 ký tự |
| Chạy công cụ kiểm tra schema của ban tổ chức trên một file 634 dòng | 634 dòng, 0 lỗi schema |
| Chạy với thư mục dữ liệu không có `train/` (như trong Docker) | Được, khi đặt `LIFEMATE_TRAIN_JSONL` |
| Các tình huống lỗi: máy chủ mô hình chết giữa chừng, không khởi động được, một clip không đọc được | Mẫu đó vẫn có dòng trong file nộp (giữ các lệnh đã gọi, hoặc đáp án "không làm gì"); lượt chạy không dừng |
| `predict.sh` chạy trọn ba bước ngoài Docker, 3 mẫu public, mô hình nghe thật và máy chủ mô hình giả | Ra 3 dòng, công cụ kiểm tra của ban tổ chức báo 0 lỗi |

**Chưa có:** mô hình ngôn ngữ chưa chạy lần nào, nên chưa có file nộp và chưa có điểm.

### 5.3. Cài đặt trên server

**[Chưa chạy thử]** trên server. Các lệnh `lifemate.cli` thì đã chạy trên máy phát triển.

```bash
cd ~/ZALOAI/LifeMate
rm -rf .venv
uv venv --python 3.12
uv pip install torch "transformers>=5" "faster-whisper>=1.1"

# Giải nén dữ liệu
mkdir -p Dataset/extracted
unzip -q "Dataset/train+public_test.zip" -x '__MACOSX/*' '*.DS_Store' -d Dataset/extracted

# Kiểm tra: phải in "0 tool results differ, 0 final app states differ"
uv run --no-project python -m lifemate.cli check-labels
```

- Luôn chạy bằng `uv run --no-project python -m lifemate.cli ...` từ thư mục `LifeMate`.
- Dữ liệu mặc định ở `Dataset/extracted/train+public_test`. Đổi bằng biến môi trường `LIFEMATE_DATA`.
- Mỗi tiến trình chỉ làm một split, vì mỗi split có bản `zaic_tools` riêng.

Tập kiểm tra cố định: mẫu train thứ 1, 11, 21 ... (cứ 10 mẫu lấy 1) là **phần dev**, gồm 155 mẫu và 519 clip. 1.390 mẫu còn lại là **phần fit**, gồm 4.517 clip.
Phần dev không bao giờ được dùng để huấn luyện, nên điểm đo trên nó là điểm thật. Trên dòng lệnh, `--every 10` chọn đúng phần dev.

### 5.4. Bước 1: chuyển âm thanh thành chữ

Lệnh `asr` có hai cách chạy mô hình:

| `--backend` | Dùng cho | Trạng thái |
|---|---|---|
| `ct2` (faster-whisper) | Tên mô hình như `large-v3-turbo`, hoặc thư mục có `model.bin` | **[Đã kiểm chứng]** trên CPU; chưa chạy trên GPU |
| `hf` (transformers) | Mã mô hình trên Hugging Face, hoặc thư mục do bước tinh chỉnh ghi ra | **[Chưa chạy thử]** |

Nếu không ghi `--backend`, chương trình tự chọn theo nội dung thư mục mô hình.

```bash
# Phần dev của train, để đo tỉ lệ sai từ
uv run --no-project python -m lifemate.cli asr --split train --every 10 \
    --model openai/whisper-large-v3-turbo --backend hf --cache result/asr_dev_turbo.jsonl --show 10

# Toàn bộ public test
uv run --no-project python -m lifemate.cli asr --split public_test \
    --model openai/whisper-large-v3-turbo --backend hf --cache result/asr_public_turbo.jsonl
```

Kết quả đã đo trên máy phát triển, chưa tinh chỉnh, 30 mẫu train lấy cách đều (`--every 51 --limit 30`), CPU, int8, beam 5:

| Mô hình | Sai từ ở yêu cầu cuối (30 clip) | Yêu cầu không sai từ nào | Tốc độ trên CPU |
|---|---|---|---|
| Whisper large-v3-turbo | 28,1% | 1/30 | 7,2 giây/clip |
| PhoWhisper-large | 26,6% | 2/30 | 10,8 giây/clip |
| PhoWhisper-medium | 27,6% | 1/30 | 6,7 giây/clip |

Riêng Whisper turbo còn đo trên 63 clip lịch sử của 30 mẫu đó: sai 35,5%.

Nhận xét:

- Ba mô hình ngang nhau. Chênh 1–2 điểm trên 30 clip là không đáng kể. Đổi mô hình không giải quyết được, phải tinh chỉnh (mục 5.7).
- Lỗi là nghe sai thật: "Zalo" thành "giao lưu", "SJC" thành "stj", "Sếp Dũng" thành "sức dũng". Clip nhiễu còn làm Whisper bịa cả câu.
- Một phần nhỏ của con số là do khác cách viết số ("chín giờ" so với "9 giờ").
- Nguyên nhân: một phần clip sạch, một phần bị thêm tiếng ồn (tỉ lệ tín hiệu trên nhiễu khoảng 13–17 dB) và cắt dải tần kiểu điện thoại. Lời thoại có từ địa phương (rứa, mô, ni, nghen, hông) và tên riêng nước ngoài.

Nếu `--backend ct2` trên GPU báo thiếu thư viện CUDA (cuBLAS, cuDNN) thì dùng `--backend hf`, vì cách này dùng thẳng CUDA của torch.

### 5.5. Bước 2: dựng máy chủ mô hình ngôn ngữ

**[Chưa chạy thử]** Mô hình ngôn ngữ chạy thành một máy chủ kiểu OpenAI trên chính server; `lifemate` gọi nó qua `--base-url`.

Ứng viên, đã kiểm tra là có trên Hugging Face và không khoá truy cập:

| Mô hình | Số tham số | Giấy phép | Vừa 24 GB không |
|---|---|---|---|
| `Qwen/Qwen3.5-9B` | 9,65 tỉ | Apache-2.0 | Vừa ở dạng 16 bit (khoảng 19 GB), còn ít chỗ cho ngữ cảnh |
| `google/gemma-4-12B-it` | 11,96 tỉ | Apache-2.0 | Không vừa ở 16 bit (khoảng 24 GB); phải dùng bản lượng tử hoá |
| `Qwen/Qwen3-14B` | 14,77 tỉ | Apache-2.0 | Phải lượng tử hoá; cộng với mô hình nghe sẽ vượt 15 tỉ |
| `Qwen/Qwen3.5-4B` | 4,66 tỉ | Apache-2.0 | Vừa thoải mái; dùng để thử nhanh |

Chưa mô hình nào được đo trên bài này. Nên bắt đầu bằng `Qwen/Qwen3.5-9B`, vì nó vừa card 3090 mà không cần lượng tử hoá và nằm trong giới hạn 15 tỉ kể cả khi cộng mô hình nghe (0,81 tỉ của Whisper turbo).

Dựng bằng vLLM, trong một môi trường riêng để không đụng thư viện của phần nghe:

```bash
cd ~/ZALOAI/LifeMate
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python vllm

.venv-vllm/bin/vllm serve Qwen/Qwen3.5-9B --served-model-name llm --port 8000 \
    --max-model-len 16384 --gpu-memory-utilization 0.92 \
    --reasoning-parser qwen3 --language-model-only

# Ở một cửa sổ khác: máy chủ sẵn sàng khi lệnh này trả về tên mô hình
curl -s http://127.0.0.1:8000/v1/models
```

- `--language-model-only` và `--reasoning-parser qwen3` lấy từ trang mô hình Qwen3.5: bỏ phần xử lý ảnh, và tách phần suy nghĩ khỏi câu trả lời.
- Qwen3.5 mặc định suy nghĩ trước khi trả lời. Khi để suy nghĩ thì đặt `--max-tokens 2048` ở lệnh `predict`. Muốn tắt: thêm `--extra '{"chat_template_kwargs": {"enable_thinking": false}}'`.
- Mô hình nghe và mô hình ngôn ngữ không nên nằm trên GPU cùng lúc: chạy xong bước `asr` rồi mới bật máy chủ.
- Nếu Claude Code từ chối chạy một chương trình tải từ ngoài (đã xảy ra với llama.cpp, mục 8), hãy tự gõ lệnh bật máy chủ trong một cửa sổ terminal. `lifemate` chỉ cần địa chỉ của máy chủ.

### 5.6. Bước 3: chạy tác nhân và chấm thử

**[Chưa chạy thử]** với mô hình thật. Chính các lệnh này đã chạy với "mô hình giả" (`--oracle`).

```bash
# A. Chỉ đo mô hình ngôn ngữ: dùng lời thoại chuẩn của train, trên phần dev
uv run --no-project python -m lifemate.cli predict --split train --every 10 --pool fit \
    --base-url http://127.0.0.1:8000/v1 --model llm --max-tokens 2048 --workers 8 \
    --out result/dev_gold.jsonl --trace result/dev_gold_trace.jsonl --fresh
uv run --no-project python -m lifemate.cli eval --pred result/dev_gold.jsonl --show 20

# B. Đo cả dây chuyền: dùng lời thoại do mô hình nghe tạo ra
uv run --no-project python -m lifemate.cli predict --split train --every 10 --pool fit \
    --asr result/asr_dev_turbo.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model llm --max-tokens 2048 --workers 8 \
    --out result/dev_asr.jsonl --trace result/dev_asr_trace.jsonl --fresh
uv run --no-project python -m lifemate.cli eval --pred result/dev_asr.jsonl --show 20
```

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--every 10` | Chỉ chạy phần dev |
| `--pool fit` | Ví dụ trong lời nhắc chỉ lấy từ phần fit, để điểm dev không bị đẹp giả. Khi chạy public thì dùng `--pool all` (mặc định) |
| `--shots N` | Số ví dụ đưa vào lời nhắc (mặc định 4; `0` là không dùng) |
| `--workers N` | Số mẫu chạy song song; máy chủ mô hình sẽ gom các yêu cầu lại |
| `--trace FILE` | Ghi từng bước của từng mẫu, để đọc khi cần biết vì sao sai |
| `--fresh` | Ghi đè file kết quả. Không có thì chạy tiếp các mẫu chưa xong |
| `--ids a,b` | Chỉ chạy các mẫu này |

`eval` in ba số: `state_match` (tỉ lệ mẫu khớp trạng thái cuối), `call_f1` (F1 trung bình trên lệnh gọi) và `both_perfect` (cả hai cùng đúng), kèm bảng theo số lệnh của đáp án.
Đây là ước lượng, vì công thức chính thức không có trong gói dữ liệu; riêng phần so văn bản tự do là luật tự đặt (trùng từ 50% trở lên thì coi là khớp).

Hiệu giữa A và B cho biết mất bao nhiêu điểm vì nghe sai. Hiệu đó lớn thì ưu tiên mục 5.7; A thấp thì ưu tiên sửa lời nhắc, đổi mô hình, hoặc mục 5.8.

### 5.7. Huấn luyện mô hình nghe

**[Chưa chạy thử]** Script [finetune_asr.py](LifeMate/lifemate/finetune_asr.py) đã viết theo giao diện của transformers 5.22 và đã kiểm tra phần chuẩn bị dữ liệu (4.517 clip huấn luyện, 519 clip dev, không trùng nhau). Phần huấn luyện cần GPU nên chưa chạy.

Cách hoạt động:

- Dữ liệu: mọi clip có lời thoại của phần fit. Clip của phần dev không được học.
- Nhãn là lời thoại nguyên văn của ban tổ chức (giữ dấu câu, chữ hoa, cách viết số).
- Tối ưu AdamW, tốc độ học tăng dần ở 10% bước đầu rồi giảm tuyến tính về 0, tính toán 16 bit, cắt gradient ở 1,0.
- In tỉ lệ sai từ trên dev trước khi học và sau mỗi vòng. Chỉ lưu lại bản có tỉ lệ thấp nhất.
- Kết quả là một thư mục mô hình dạng Hugging Face, kèm `metrics.json` ghi tỉ lệ sai từ từng vòng.

```bash
cd ~/ZALOAI/LifeMate

# 1. Chạy thử 20 bước để bắt lỗi sớm (vài phút)
uv run --no-project python -m lifemate.finetune_asr --base openai/whisper-large-v3-turbo \
    --out models/asr-try --max-steps 20 --dev-limit 32

# 2. Chạy thật với Whisper turbo
uv run --no-project python -m lifemate.finetune_asr --base openai/whisper-large-v3-turbo \
    --out models/asr-turbo-ft --epochs 8

# 3. So với PhoWhisper-large (mô hình lớn gấp đôi nên dùng batch nhỏ hơn)
uv run --no-project python -m lifemate.finetune_asr --base vinai/PhoWhisper-large \
    --out models/asr-pho-ft --epochs 8 --batch 4 --accum 4

# 4. Nhận dạng phần dev bằng mô hình tốt hơn, rồi chạy lại mục 5.6 B với file này
uv run --no-project python -m lifemate.cli asr --split train --every 10 \
    --model models/asr-turbo-ft --cache result/asr_dev_ft.jsonl --show 10
```

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--epochs` | 8 | Số vòng học qua toàn bộ dữ liệu |
| `--batch`, `--accum` | 8 và 2 | Số clip mỗi bước và số bước gộp; tích của chúng là batch thật (16) |
| `--lr` | 1e-5 | Tốc độ học |
| `--spec-augment` | tắt | Che ngẫu nhiên một số đoạn của đầu vào khi học, giúp đỡ học vẹt |
| `--grad-checkpoint` | tắt | Bật nếu GPU báo hết bộ nhớ (chậm hơn) |
| `--all-data` | tắt | Học cả phần dev; chỉ dùng cho bản cuối |

Cách chọn và các hướng thử khi kết quả chưa tốt:

- So `dev word error rate` của hai mô hình, chọn mô hình thấp hơn. Whisper turbo có 0,81 tỉ tham số; PhoWhisper-large theo kiến trúc Whisper large-v2, khoảng 1,5 tỉ.
- Tỉ lệ trên dev giảm rồi tăng lại là dấu hiệu học vẹt: giảm `--epochs` hoặc bật `--spec-augment`.
- Tỉ lệ giảm chậm: thử `--lr 2e-5`. Dao động mạnh: thử `--lr 5e-6`.
- Hết bộ nhớ: giảm `--batch` và tăng `--accum` để tích không đổi, hoặc bật `--grad-checkpoint`.

Bản cuối, sau khi đã chốt mô hình và số vòng, học trên toàn bộ 5.036 clip rồi nhận dạng public:

```bash
uv run --no-project python -m lifemate.finetune_asr --base openai/whisper-large-v3-turbo \
    --out models/asr-final --epochs <số vòng tốt nhất> --all-data
uv run --no-project python -m lifemate.cli asr --split public_test \
    --model models/asr-final --cache result/asr_public_final.jsonl
```

Với `--all-data`, con số trên dev không còn ý nghĩa vì mô hình đã học các clip đó.

Tuỳ chọn, chưa cần: muốn dùng faster-whisper cho nhanh thì chuyển mô hình sang CTranslate2:

```bash
uv run --no-project ct2-transformers-converter --model models/asr-final --output_dir models/asr-final-ct2 \
    --copy_files tokenizer.json preprocessor_config.json --quantization float16
```

### 5.8. Huấn luyện mô hình ngôn ngữ (chỉ khi cần)

Trước tiên chạy không huấn luyện (mục 5.6), vì lời nhắc đã kèm ví dụ lấy từ train. Chỉ tinh chỉnh khi điểm ở bước A còn thấp sau khi đã sửa lời nhắc.

**[Đã kiểm chứng]** Xuất dữ liệu huấn luyện:

```bash
uv run --no-project python -m lifemate.cli export-sft --out result/sft_fit.jsonl               # 1.390 mẫu phần fit
uv run --no-project python -m lifemate.cli export-sft --part dev --out result/sft_dev.jsonl   # 155 mẫu phần dev
```

Mỗi dòng là một mẫu train viết lại đúng như cuộc trò chuyện mà tác nhân thấy lúc chạy thật, theo sau là đáp án viết bằng giao thức ở mục 5.2:

```json
{"id": "train_0002", "n_context": 2,
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "Nếu mai mưa thì hủy lịch đá bóng... À mà thôi, ..."},
              {"role": "assistant", "content": "{\"call\": \"weather.get_forecast\", \"args\": {\"date\": \"2026-11-21\"}}"},
              {"role": "user", "content": "RESULT {...}"},
              {"role": "assistant", "content": "{\"reply\": \"…\"}"}]}
```

- `n_context` là vị trí bắt đầu của đáp án. Chỉ các tin nhắn `assistant` từ vị trí đó trở đi là thứ mô hình phải học; phần trước là ngữ cảnh.
- Phần fit cho 1.390 mẫu, khoảng 15.900 ký tự mỗi mẫu, tổng cộng 4.383 tin nhắn cần học.
- Thêm `--asr <file lời thoại>` để dùng lời thoại của mô hình nghe thay cho lời thoại chuẩn, giúp mô hình ngôn ngữ quen với lỗi nghe. Lưu ý: mô hình nghe đã tinh chỉnh sẽ nghe các clip phần fit tốt hơn thực tế, vì nó đã học chính các clip đó.

**[Chưa làm]** Script huấn luyện chưa viết, vì nó phụ thuộc mô hình được chọn. Cách làm dự kiến là LoRA:

1. Nạp mô hình ở dạng 4 bit để vừa 24 GB.
2. Chỉ tính sai số trên các tin nhắn `assistant` từ `n_context` trở đi.
3. Điểm xuất phát: hạng LoRA 16, tốc độ học 1e-4, 2–3 vòng, độ dài tối đa khoảng 8.000 token.
4. Sau mỗi vòng, chạy mục 5.6 A trên phần dev; giữ bản có điểm cao nhất.
5. Gộp LoRA vào mô hình gốc rồi dựng máy chủ như mục 5.5.

Phiên làm việc trên server phải xác định tên các lớp nhận LoRA của mô hình được chọn. Kiến trúc của Qwen3.5 khác các đời trước, nên không ghi sẵn ở đây.

### 5.9. Tạo file nộp public

**[Chưa chạy thử]** Chỉ làm khi điểm dev của cả dây chuyền (mục 5.6 B) đã ổn.

```bash
uv run --no-project python -m lifemate.cli predict --split public_test \
    --asr result/asr_public_final.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model llm --max-tokens 2048 --workers 8 \
    --out result/submission.jsonl --trace result/trace_public.jsonl --fresh

# Kiểm tra schema bằng công cụ của ban tổ chức
cd Dataset/extracted/train+public_test
PYTHONPATH=public_test python3 -m zaic_tools --split public_test --root . \
    --validate ../../../result/submission.jsonl      # phải in: 634 lines checked, 0 schema errors
```

File phải có đúng 634 dòng, mỗi mẫu một dòng. Lệnh `predict` tự xếp lại các dòng theo thứ tự của tập dữ liệu.

Luật của ban tổ chức cần nhớ: mỗi mẫu chỉ được mở phiên tool một lần, và mọi lệnh đã gọi đều tính vào bài nộp. Vì vậy không chạy một mẫu nhiều lần rồi chọn kết quả đẹp nhất.

### 5.10. Docker

Bộ file gồm `Dockerfile`, `predict.sh` và `.dockerignore`. Các bước đóng gói nằm ở mục 6.3.

Gói dữ liệu chỉ nói: vòng private chạy bằng Docker của đội, Docker phải tạo ra đúng `submission.jsonl`, và hướng dẫn đóng gói sẽ có tài liệu riêng. Tập private chỉ được phát hành sau khi nộp Docker.

Thiết kế:

- Image nền `vllm/vllm-openai:v0.33.0` (có trên Docker Hub, khoảng 11 GB khi nén). Nó có sẵn CUDA, torch, transformers và máy chủ vLLM, nên lúc build không phải cài gì từ mạng.
- Trong image: mã nguồn `lifemate/`, `train.jsonl` (để lấy ví dụ cho lời nhắc), mô hình nghe ở `/models/asr`, mô hình ngôn ngữ ở `/models/llm`.
- Không có trong image: `zaic_tools` và dữ liệu test. Chúng đi kèm tập test được gắn vào lúc chấm; mã nguồn tự nạp `zaic_tools` từ thư mục dữ liệu.
- Chạy không mạng: `HF_HUB_OFFLINE=1`, mọi mô hình nạp từ thư mục trong image.

`predict.sh` làm ba bước: nhận dạng giọng nói (xong thì GPU trống) → bật máy chủ mô hình ngôn ngữ và chờ nó sẵn sàng → chạy tác nhân, ghi `/result/submission.jsonl`.

- Tự tìm tên split trong `/data`, nên không cần biết trước tên tập private. Chấp nhận hai kiểu gắn: `/data/<split>/<split>.jsonl` hoặc `/data/<split>.jsonl`.
- Một bước hỏng không làm dừng lượt chạy. Clip không đọc được thì coi như yêu cầu rỗng. Máy chủ mô hình không lên thì các mẫu nhận đáp án "không làm gì". File nộp luôn được ghi đủ dòng.
- Log của từng bước nằm trong `/result/logs/`: `asr.log`, `llm_server.log`, `predict.log`, `trace.jsonl`.

Mọi thiết lập đổi được bằng biến môi trường (`docker run -e TÊN=giá_trị`):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DATA_DIR`, `RESULT_DIR` | `/data`, `/result` | Nơi gắn dữ liệu và nơi ghi kết quả |
| `SPLIT` | tự tìm | Tên tập cần chạy, ví dụ `public_test` |
| `ASR_MODEL`, `ASR_ARGS` | `/models/asr`, `--backend hf` | Mô hình nghe và tuỳ chọn của lệnh `asr` |
| `LLM_MODEL`, `LLM_ARGS` | `/models/llm`, `--max-model-len 16384 --gpu-memory-utilization 0.92` | Mô hình ngôn ngữ và tuỳ chọn của `vllm serve` |
| `LLM_WAIT` | `1800` | Số giây chờ máy chủ nạp xong mô hình |
| `PREDICT_ARGS` | `--max-tokens 2048 --workers 8` | Tuỳ chọn của lệnh `predict` |
| `SAMPLE_ARGS` | rỗng | Chọn mẫu để thử nhanh, ví dụ `--limit 20` |

**[Chưa chạy thử]** Image chưa build được, vì chưa có hai thư mục mô hình (`models/asr-final` và `models/llm`).

**[Đã kiểm chứng]** `predict.sh` chạy ngoài Docker trên máy phát triển, 3 mẫu public, mô hình nghe thật (Whisper turbo trên CPU) và một máy chủ mô hình giả:

| Tình huống | Kết quả |
|---|---|
| `/data` chứa thư mục `public_test/` | Tìm đúng split, ra 3 dòng, mã kết thúc 0 |
| `/data` chính là thư mục `public_test` | Như trên, đáp án giống hệt |
| Máy chủ mô hình không khởi động được | Phát hiện ngay, in log của máy chủ, vẫn ghi đủ 3 dòng |
| `/data` rỗng | Báo "no split found", mã kết thúc 2 |

Công cụ kiểm tra schema của ban tổ chức báo 0 lỗi trên file sinh ra.

### 5.11. Việc còn lại, theo thứ tự

1. Cài đặt trên server (mục 5.3), chạy `check-labels`.
2. Dựng máy chủ mô hình (mục 5.5), đo phần ngôn ngữ trên dev bằng lời thoại chuẩn (mục 5.6 A).
3. Đọc các mẫu sai trong file trace, sửa lời nhắc; so thêm một mô hình ứng viên.
4. Tinh chỉnh mô hình nghe (mục 5.7), đo lại cả dây chuyền (mục 5.6 B).
5. Nếu phần ngôn ngữ vẫn thấp: tinh chỉnh LoRA (mục 5.8).
6. Tạo file nộp public (mục 5.9) và nộp.
7. Đóng gói Docker (mục 6.3).

---

## 6. Đóng gói Docker để nộp

Làm trên server, vì máy phát triển không có Docker.
Thứ tự chung cho cả hai bài: build image → chạy thử đúng kiểu ban tổ chức chấm → kiểm tra kết quả → gửi image.

### 6.1. Hiện trạng bộ file Docker

| Bài | Bộ file | Đủ để build chưa | Đã kiểm chứng đến đâu |
|---|---|---|---|
| DataMate | `Dockerfile`, `predict.sh`, `requirements.txt`, `.dockerignore` | Đủ | Giả lập không cần Docker cho ra đúng file được 1,0 (mục 4.5). Chưa build bằng Docker thật |
| LifeMate | `Dockerfile`, `predict.sh`, `.dockerignore` | Chưa, còn thiếu hai thư mục mô hình | `predict.sh` chạy trọn ba bước ngoài Docker với máy chủ mô hình giả (mục 5.10). Chưa build |

Cả hai bộ file viết theo cách chấm đang biết: tập test gắn chỉ đọc ở `/data`, kết quả ghi vào `/result/submission.jsonl`, lúc chạy không có mạng.
Hướng dẫn Docker chính thức của ban tổ chức chưa có trong tay. Khi có, đối chiếu ba điểm: image nền, đường dẫn gắn dữ liệu, tên file kết quả. Đường dẫn đổi được mà không phải sửa code, bằng biến `DATA_DIR` và `RESULT_DIR`.

### 6.2. DataMate, từng bước

**[Chưa chạy thử]** bằng Docker thật. Mỗi bước ghi kèm kết quả phải thấy.

```bash
cd ~/ZALOAI/DataMate

# Bước 1. Build image
docker build -t datamate:v7 .
docker images datamate:v7                 # xem dung lượng image

# Bước 2. Kiểm tra thư viện trong image đúng bản đã ghi cứng. Phải in: 3.1.1 0.9.1
docker run --rm --network none --entrypoint python datamate:v7 \
    -c "import pandas; from importlib.metadata import version; print(pandas.__version__, version('python-calamine'))"

# Bước 3. Chạy như lúc chấm: không mạng, dữ liệu chỉ đọc. Ghi lại thời gian chạy.
sudo rm -rf result_docker && mkdir -p result_docker
time docker run --rm --network none \
    -v "$PWD/dataset/public_test:/data:ro" -v "$PWD/result_docker:/result" datamate:v7

# Bước 4. Kết quả phải giống hệt file đã được 1,0
wc -l result_docker/submission.jsonl      # phải là 1000
cmp result_docker/submission.jsonl result/submission_v7.jsonl && echo "OK: giong het ban da nop"

# Bước 5. Chạy trên train rồi chấm file do Docker sinh ra. Phải in: dung 3000 | sai 0 | bo trong 0
sudo rm -rf result_docker_train && mkdir -p result_docker_train
docker run --rm --network none \
    -v "$PWD/dataset/train:/data:ro" -v "$PWD/result_docker_train:/result" datamate:v7
uv run python -c "
from datamate.score import load_jsonl, score_one
ref = {r['id']: r for r in load_jsonl('dataset/train/answers.jsonl')}
s = [score_one(p, ref[p['id']]) for p in load_jsonl('result_docker_train/submission.jsonl')]
print('dung', s.count(1), '| sai', s.count(-1), '| bo trong', s.count(0), '| tong', len(s))"
```

Nếu bước 4 không in "OK": chạy `diff <(sort result_docker/submission.jsonl) <(sort result/submission_v7.jsonl) | head` để xem câu nào khác, rồi xem lý do trong `result_docker/logs/run_log.jsonl`.

File do Docker ghi vào `result_docker` thuộc quyền `root`, nên các lệnh trên dùng `sudo rm -rf` để dọn thư mục trước mỗi lần chạy.

### 6.3. LifeMate, từng bước

**[Chưa chạy thử]** Chỉ làm được sau khi đã có mô hình nghe tinh chỉnh (mục 5.7), đã chọn mô hình ngôn ngữ và đã có điểm dev ưng ý (mục 5.6).

**Bước 1. Đặt hai thư mục mô hình vào đúng chỗ.**

```bash
cd ~/ZALOAI/LifeMate
ls models/asr-final        # do mục 5.7 ghi ra: config.json, model.safetensors, tokenizer.json, preprocessor_config.json ...

# Mô hình ngôn ngữ: tải về thành thư mục thường. Nếu đã tinh chỉnh LoRA thì đây là thư mục mô hình đã gộp.
.venv/bin/hf download Qwen/Qwen3.5-9B --local-dir models/llm
du -sh models/asr-final models/llm
```

**Bước 2. Cho `predict.sh` dùng đúng các tuỳ chọn đã đo điểm.** Mở `predict.sh`, sửa giá trị mặc định của hai dòng:

- `LLM_ARGS`: giống hệt các tuỳ chọn của lệnh `vllm serve` ở mục 5.5. Với Qwen3.5 là `--max-model-len 16384 --gpu-memory-utilization 0.92 --reasoning-parser qwen3 --language-model-only`.
- `PREDICT_ARGS`: giống hệt các tuỳ chọn của lệnh `predict` đã cho điểm dev tốt nhất (`--max-tokens`, `--workers`, `--shots`, `--extra` nếu có).

Docker phải chạy đúng cấu hình đã đo. Đổi một tuỳ chọn mà không đo lại là nộp một lời giải chưa kiểm chứng.

**Bước 3. Thử `predict.sh` ngoài Docker trên 20 mẫu.** Bước này bắt lỗi nhanh, không phải chờ build.

```bash
DATA_DIR="$PWD/Dataset/extracted/train+public_test" RESULT_DIR="$PWD/result_script" \
PYTHON="$PWD/.venv/bin/python" PATH="$PWD/.venv-vllm/bin:$PATH" \
ASR_MODEL="$PWD/models/asr-final" LLM_MODEL="$PWD/models/llm" \
LIFEMATE_TRAIN_JSONL="$PWD/Dataset/extracted/train+public_test/train/train.jsonl" \
SAMPLE_ARGS="--limit 20" ./predict.sh
```

Phải thấy ba dòng `1/3`, `2/3`, `3/3`, rồi `done: 20 lines`. Nếu máy chủ mô hình không lên, script in luôn phần cuối của `result_script/logs/llm_server.log`.

**Bước 4. Build.** Lúc build Docker phải đọc hơn 20 GB mô hình nên mất vài phút.

```bash
docker build -t lifemate:v1 .
docker images lifemate:v1                 # sẽ là vài chục GB: image nền (11 GB khi nén) cộng hơn 20 GB mô hình
```

**Bước 5. Thử nhanh 20 mẫu trong Docker, không mạng.**

```bash
sudo rm -rf result_docker && mkdir -p result_docker
docker run --rm --gpus all --network none -e SAMPLE_ARGS="--limit 20" \
    -v "$PWD/Dataset/extracted/train+public_test:/data:ro" -v "$PWD/result_docker:/result" lifemate:v1
```

**Bước 6. Chạy đủ 634 mẫu public trong Docker, rồi kiểm tra.**

```bash
sudo rm -rf result_docker && mkdir -p result_docker
time docker run --rm --gpus all --network none \
    -v "$PWD/Dataset/extracted/train+public_test:/data:ro" -v "$PWD/result_docker:/result" lifemate:v1

wc -l result_docker/submission.jsonl      # phải là 634
(cd Dataset/extracted/train+public_test && PYTHONPATH=public_test python3 -m zaic_tools \
    --split public_test --root . --validate ../../../result_docker/submission.jsonl)   # 634 lines checked, 0 schema errors

# So với file đã nộp lên bảng xếp hạng (chạy ngoài Docker)
uv run --no-project python - result/submission.jsonl result_docker/submission.jsonl <<'EOF'
import json, sys
a = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()}
b = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()}
diff = [k for k in a if a[k] != b.get(k)]
print(f"{len(a)} dong / {len(b)} dong; khac nhau: {len(diff)}", diff[:5])
EOF
```

Khác với DataMate, hai lần chạy mô hình ngôn ngữ trên GPU có thể lệch nhau vài mẫu, nên không đòi giống hệt. Lệch nhiều (trên vài phần trăm số mẫu) là dấu hiệu Docker đang chạy khác cấu hình: xem lại bước 2.

**Bước 7. Chấm phần dev ngay trong Docker.** Điểm phải xấp xỉ điểm đã đo ngoài Docker ở mục 5.6 B.

```bash
sudo rm -rf result_docker_dev && mkdir -p result_docker_dev
docker run --rm --gpus all --network none -e SPLIT=train -e SAMPLE_ARGS="--every 10" \
    -e PREDICT_ARGS="--max-tokens 2048 --workers 8 --pool fit" \
    -v "$PWD/Dataset/extracted/train+public_test:/data:ro" -v "$PWD/result_docker_dev:/result" lifemate:v1
uv run --no-project python -m lifemate.cli eval --pred result_docker_dev/submission.jsonl
```

Ở lệnh này `PREDICT_ARGS` phải là các tuỳ chọn của bước 2 cộng thêm `--pool fit`.

Ghi chú về GPU trong Docker:

- `--gpus all` cần NVIDIA Container Toolkit (mục 3.3).
- vLLM dùng bộ nhớ chia sẻ. Nếu `llm_server.log` báo lỗi về shared memory, thêm `--ipc=host` vào lệnh `docker run`, và báo cho ban tổ chức biết image cần cờ này.

### 6.4. Gửi image cho ban tổ chức

Cách gửi phải theo hướng dẫn của ban tổ chức; hiện chưa biết họ nhận theo cách nào. Hai cách thường gặp:

**Cách 1, nộp file image.**

```bash
docker save datamate:v7 | gzip > datamate_v7.tar.gz
sha256sum datamate_v7.tar.gz              # gửi kèm mã này để bên nhận kiểm tra file không hỏng
ls -lh datamate_v7.tar.gz

# Thử lại đúng file sẽ gửi: xoá image, nạp lại từ file, chạy lại bước 3 và 4 của mục 6.2
docker rmi datamate:v7
docker load < datamate_v7.tar.gz
```

**Cách 2, đẩy lên kho chứa image.** Thay `<kho>` bằng địa chỉ ban tổ chức yêu cầu.

```bash
docker tag datamate:v7 <kho>/datamate:v7
docker login <kho>
docker push <kho>/datamate:v7
docker inspect --format '{{index .RepoDigests 0}}' <kho>/datamate:v7     # mã định danh của đúng bản đã đẩy
```

Với LifeMate thay `datamate:v7` bằng `lifemate:v1`. File image của LifeMate sẽ nặng vài chục GB, nên cần biết trước giới hạn dung lượng.

Thường phải gửi kèm:

- Mô tả lời giải có danh sách mô hình và số tham số. DataMate đã có ở `DataMate/README.md` (mục Models: 0 tham số). LifeMate phải viết khi chốt mô hình: tên và số tham số của mô hình nghe và mô hình ngôn ngữ.
- Lệnh chạy image, nếu khác lệnh mẫu của ban tổ chức (ví dụ cần `--gpus all` hoặc `--ipc=host`).

### 6.5. Danh sách kiểm tra trước khi gửi

Áp dụng cho cả hai bài:

- [ ] Build thành công từ một bản chép mới của thư mục dự án, không dựa vào file nào ngoài thư mục đó.
- [ ] Chạy được với `--network none`.
- [ ] Chạy được khi dữ liệu gắn `:ro`.
- [ ] File kết quả đúng tên `submission.jsonl`, đúng số dòng.
- [ ] Thời gian chạy và dung lượng image nằm trong giới hạn của ban tổ chức.
- [ ] Image nạp lại từ đúng file sẽ gửi (hoặc kéo về từ kho chứa) vẫn chạy ra cùng kết quả.
- [ ] Đã đối chiếu hướng dẫn Docker chính thức: image nền, đường dẫn gắn dữ liệu, tên file kết quả.

Riêng DataMate:

- [ ] Kết quả public giống hệt `result/submission_v7.jsonl`.
- [ ] File train do Docker sinh ra chấm được 3.000 đúng, 0 sai.

Riêng LifeMate:

- [ ] `predict.sh` dùng đúng các tuỳ chọn đã đo điểm (bước 2 của mục 6.3).
- [ ] Công cụ kiểm tra schema báo 0 lỗi trên 634 dòng.
- [ ] Điểm dev trong Docker xấp xỉ điểm dev ngoài Docker.
- [ ] Tổng số tham số của hai mô hình nằm trong giới hạn của hạng mục thi.
- [ ] Trong `result_docker/logs/predict.log`, số mẫu "stopped early by an error" bằng 0.

---

## 7. Hạn và quy tắc nộp bài

| Việc | Thời điểm | Ghi chú |
|---|---|---|
| Nộp Docker của DataMate | 23:59 ngày 16/11/2026 (giờ Việt Nam) | Docker chưa chạy thử |
| Vòng private của DataMate | 20–23/11/2026 | |
| Hạn của LifeMate | Chưa xác nhận | Kiểm tra trên cổng thi |

- Cổng thi cho 5 lượt nộp mỗi ngày. Kiểm trên train trước, gom các cải tiến lại rồi mới nộp một file.
- Hạng **SLM**: tổng tham số ≤ 15 tỉ.
- Mô tả bài nộp viết bằng tiếng Anh, ngắn, không dùng thuật ngữ.
- Bộ dữ liệu chỉ dành cho đội đã đăng ký và không được phát tán. Nếu đưa lên dịch vụ ngoài thì phải để chế độ riêng tư.

---

## 8. Sự cố đã gặp

| Sự cố | Cách xử lý |
|---|---|
| Máy phát triển tải từ PyPI rất chậm (50–150 KB/giây); Hugging Face và GitHub thì nhanh | Cài thư viện nặng trên server |
| `.venv` chép sang máy khác không chạy | Xoá và tạo lại bằng `uv venv` hoặc `uv sync` |
| faster-whisper 1.2.1 lỗi `open() got an unexpected keyword argument 'metadata_errors'` với bản PyAV mới | `lifemate/asr.py` tự đọc file wav, không dùng PyAV nữa |
| Whisper bịa câu trên clip nhiễu ("Hãy subscribe cho kênh ...") | Tinh chỉnh mô hình nghe (mục 5.7) |
| Claude Code từ chối chạy chương trình tải thẳng từ GitHub (llama.cpp) | Cài qua trình quản lý gói (`uv pip install vllm`), hoặc tự gõ lệnh bật máy chủ |
| Hai split của LifeMate có hai bản `zaic_tools` cùng tên | Mỗi tiến trình chỉ làm một split |
| Chấm DataMate trên train cần 3,3 GB RAM | Đóng bớt ứng dụng trên máy 15 GB, hoặc chạy trên server |

---

## 9. Câu hỏi còn mở

Cần câu trả lời từ trang cuộc thi hoặc ban tổ chức:

1. **LifeMate, giới hạn mô hình:** 15 tỉ tham số của hạng SLM tính tổng mô hình nghe và mô hình ngôn ngữ, hay chỉ mô hình ngôn ngữ? Có hạng nào khác không?
2. **LifeMate, cách chấm:** công thức gộp phần khớp trạng thái và F1; lệnh đọc có tính vào F1 không; văn bản tự do phải giống đến mức nào.
3. **LifeMate, Docker:** cấu hình GPU, giới hạn thời gian và dung lượng image, cách gắn dữ liệu, hạn nộp.
4. **DataMate, Docker:** image nền có bị quy định không, giới hạn thời gian, cách nộp image.
