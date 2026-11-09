# LifeMate — trợ lý giọng nói gọi tool (Zalo AI Challenge 2026)

Đây là ghi chú ngắn của riêng LifeMate. Hướng dẫn đầy đủ (cách làm, cài đặt trên server, huấn luyện, Docker) nằm ở [README.md](../README.md) của thư mục gốc, mục 5.

Dây chuyền: **âm thanh → chữ (mô hình nghe) → mô hình ngôn ngữ chọn tool → `zaic_tools` chạy tool**.

## Trạng thái (09/11/2026)

Đã kiểm chứng trên máy phát triển (không GPU):

- Chạy lại 1.545 đáp án train qua `zaic_tools`: 0 kết quả lệch, 0 trạng thái lệch.
- Chạy lại 3 dòng của file nộp mẫu: `state_hash` khớp cả 3.
- Vòng lặp gọi tool với "mô hình giả" đọc lại đáp án: 1.545/1.545, kể cả khi chạy 8 luồng song song.
- Ba mô hình nghe chưa tinh chỉnh đều sai 27–28% số từ trên 30 yêu cầu train (Whisper large-v3-turbo 28,1%; PhoWhisper-large 26,6%; PhoWhisper-medium 27,6%).
- `predict.sh` (điểm vào của Docker) chạy trọn ba bước ngoài Docker trên 3 mẫu public, với mô hình nghe thật và một máy chủ mô hình giả. Khi máy chủ mô hình hỏng hoặc một clip không đọc được, file nộp vẫn đủ dòng.

Chưa có: mô hình ngôn ngữ chưa chạy lần nào (chưa có file nộp, chưa có điểm); mô hình nghe chưa tinh chỉnh; image Docker chưa build được vì chưa có hai thư mục mô hình `models/asr-final` và `models/llm` (xem mục 6.3 của README gốc).

## Cài đặt

```bash
cd LifeMate
rm -rf .venv                       # môi trường chép từ máy khác không dùng lại được
uv venv --python 3.12
uv pip install torch "transformers>=5" "faster-whisper>=1.1"

mkdir -p Dataset/extracted         # nếu chưa giải nén
unzip -q "Dataset/train+public_test.zip" -x '__MACOSX/*' '*.DS_Store' -d Dataset/extracted

uv run --no-project python -m lifemate.cli check-labels   # phải in: 0 tool results differ, 0 final app states differ
```

## Lệnh

Mọi lệnh chạy từ thư mục `LifeMate` bằng `uv run --no-project python -m ...`.

| Lệnh | Việc làm |
|---|---|
| `lifemate.cli check-labels` | Chạy lại đáp án train để kiểm tra môi trường |
| `lifemate.cli asr` | Âm thanh thành chữ, có lưu đệm; trên train in thêm tỉ lệ sai từ |
| `lifemate.cli predict` | Chạy tác nhân, ghi file nộp |
| `lifemate.cli eval` | Chấm thử một file dự đoán trên train |
| `lifemate.cli export-sft` | Xuất dữ liệu tinh chỉnh mô hình ngôn ngữ |
| `lifemate.finetune_asr` | Tinh chỉnh mô hình nghe trên clip train (cần GPU, chưa chạy thử) |
| `./predict.sh` | Điểm vào của Docker: nhận dạng giọng nói, bật máy chủ mô hình, chạy tác nhân, ghi `submission.jsonl` |

Chọn mẫu, dùng chung cho `asr`, `predict`, `export-sft`: `--every K`, `--limit N`, `--ids a,b`.
`--every 10` trên train là phần dev (155 mẫu), không bao giờ dùng để huấn luyện.

## Cấu trúc

| File | Việc làm |
|---|---|
| `lifemate/data.py` | Đường dẫn dữ liệu, nạp mẫu, tách fit / dev, mở phiên `ToolEnv` |
| `lifemate/asr.py` | Hai cách chạy mô hình nghe (faster-whisper, transformers), đọc wav, lưu đệm, tỉ lệ sai từ |
| `lifemate/finetune_asr.py` | Tinh chỉnh mô hình nghe |
| `lifemate/prompt.py` | Lời nhắc: luật, lịch tuần, mô tả tool, lịch sử, ví dụ |
| `lifemate/retrieve.py` | Chọn ví dụ train giống yêu cầu (TF-IDF) |
| `lifemate/llm.py` | Gọi máy chủ mô hình kiểu OpenAI, đọc câu trả lời JSON |
| `lifemate/agent.py` | Vòng lặp gọi tool và các rào chắn |
| `lifemate/sft.py` | Dựng dữ liệu tinh chỉnh mô hình ngôn ngữ |
| `lifemate/score.py` | Chấm thử: khớp trạng thái và F1 trên lệnh gọi |
| `lifemate/cli.py` | Các lệnh |
| `Dockerfile`, `predict.sh`, `.dockerignore` | Bộ file Docker cho vòng private |

## Điều cần nhớ

- Mỗi mẫu chỉ được mở `ToolEnv` một lần; mọi lệnh đã gọi, kể cả lệnh bị từ chối, đều tính vào bài nộp.
- Chấm thử chỉ là ước lượng: công thức chính thức không có trong gói dữ liệu.
- Mô hình bên ngoài không dùng được cho vòng cuối: vòng private chạy bằng Docker của đội.
- Cổng nộp bài cho 5 lượt mỗi ngày: kiểm trên train trước, gom việc rồi mới nộp. Mô tả bài nộp viết bằng tiếng Anh, ngắn, không dùng thuật ngữ.
