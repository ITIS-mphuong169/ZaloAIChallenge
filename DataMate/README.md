# DataMate — tác nhân hỏi đáp dữ liệu bảng (Zalo AI Challenge 2026)

Hệ thống trả lời câu hỏi tiếng Việt trên dữ liệu CSV / Excel / SQLite của từng cửa hàng.
Phiên bản này **không dùng mô hình học máy** (0 tham số): toàn bộ là chương trình xác định, chạy bằng CPU, không cần mạng.
Câu nào chưa hiểu chắc thì trả `abstain` (0 điểm) thay vì đoán (sai bị −1).

## Models

Không sử dụng mô hình nào ở bước suy luận (tổng số tham số: 0).

## Cách chạy

Cần [uv](https://docs.astral.sh/uv/) và Python 3.12.

```bash
uv sync

# Tạo file nộp cho một tập (thư mục chứa questions.jsonl, calendar.csv, data/<business_id>/...)
uv run python -m datamate.cli predict --data dataset/public_test --out result/submission.jsonl --log result/run_log.jsonl

# Chấm trên tập train (cần answers.jsonl)
uv run python -m datamate.cli eval --data dataset/train --show wrong
```

Chạy bằng Docker theo đúng cách ban tổ chức chấm:

```bash
docker build -t datamate:v7 .
mkdir -p result_docker
docker run --rm --network none -v "$PWD/dataset/public_test:/data:ro" -v "$PWD/result_docker:/result" datamate:v7
```

## Kiến trúc

| Bước | File | Việc làm |
|---|---|---|
| Đọc dữ liệu | `datamate/raw.py`, `datamate/readme.py` | Đọc mọi file/sheet/bảng, tìm dòng tiêu đề, gộp các sheet/file cùng cấu trúc, đọc ý nghĩa cột trong README của cửa hàng |
| Chuẩn hoá | `datamate/textnorm.py` | Số (`560,000`, `25.000đ`, `25k`, `531 500`, `1 chai`), ngày (`01/02/2025`, số serial Excel, `1 thg 6 2025`) |
| Mô hình bán hàng | `datamate/model.py` | Nhận diện vai trò cột, dựng bảng dòng hàng và bảng đơn, phân loại trạng thái (hoàn thành / huỷ / trả / đang giao), bỏ dòng tổng phụ, dòng trùng, dòng voucher / phí ship |
| Danh mục | `datamate/enrich.py` | Danh mục hàng, nhóm hàng, nhóm khách, quy đổi về đơn vị cơ bản, gộp các cách viết tên |
| Câu hỏi | `datamate/question.py` | Bỏ dấu, sửa lỗi telex, viết tắt/teencode, nhận diện kỳ thời gian theo `as_of_date` |
| Trả lời | `datamate/engine.py` | Nhận diện thực thể và ý định, tính toán theo rulebook, định dạng đáp án |
| Chấm | `datamate/score.py` | Penalized Accuracy theo đúng luật so khớp của ban tổ chức |

## Các dạng câu hỏi đang trả lời

- **Bán hàng:** doanh thu, số đơn, đơn huỷ, tỉ lệ huỷ, giá trị trung bình một đơn, ngày / tháng cao nhất, tăng giảm % giữa hai kỳ
  (tháng trước, cùng kỳ, năm trước, trước Tết, ca chiều so ca sáng), doanh thu từng tháng, xu hướng tăng liên tục,
  số đơn / số ngày vượt ngưỡng, trung bình theo thứ trong tuần.
- **Lát cắt:** theo mặt hàng, nhóm hàng, nhóm khách, kênh bán, hình thức thanh toán, ca, nhân viên; tỉ trọng %.
- **Xếp hạng:** hàng bán chạy, hàng doanh thu cao nhất, hàng lãi nhất, khách mua nhiều nhất, nhân viên bán giỏi nhất,
  và các chỉ số có tên trong rulebook (khách sộp nhất, trung thành nhất, hay đổi ý nhất, hàng ưa chuộng nhất,
  ngày đắt khách nhất, giờ cao điểm, nhà cung cấp chính).
- **Khách và hàng:** số khách có từ N đơn, số mặt hàng có mặt trong từ N đơn, mặt hàng ế, số lượng bán theo đơn vị cơ bản.
- **Giá, lãi:** giá bán theo ngày, hiệu ứng tăng giá (kể cả tự tìm ngày đổi giá trong bảng giá), lãi gộp, lãi ròng.
- **Nhập hàng, kho:** tiền nhập, nhà cung cấp, giá nhập bình quân, số lượng nhập, nhập trừ bán, tồn kho (chẵn / lẻ),
  giá trị tồn theo giá vốn, hàng hết, lô và hạn dùng.
- **Công nợ:** còn nợ, số khách nợ, top khách nợ, thu nợ, nợ thêm / trả được, tuổi nợ, hạn thanh toán.
  Ba kiểu ghi nợ: bảng các lần trả tiền; cột còn nợ tính đến ngày cuối dữ liệu; ghi chú tự do trên đơn
  (`trả 259k, nợ 449.500đ`, `nợ cả đơn`). Hai kiểu sau chỉ trả lời được tại ngày cuối dữ liệu.
- **Theo ngành:** nhà thuốc (kê đơn, hoạt chất, lô), cà phê (size, topping ghi chung ô), vật liệu xây dựng (đợt giao hàng),
  mẹ và bé (voucher, điểm, phí ship, hãng, size / số sữa, thành viên mới), sửa xe (phiếu, thợ, bảo hành, hoa hồng,
  lượt xe theo việc đã làm, số chiếc xe theo dòng xe / biển số, dòng xe đem lại nhiều tiền nhất),
  điện thoại (ghi tắt, loại hàng, dòng máy, thu máy cũ, trả góp, bảo hành).
- **Không trả lời được:** kỳ ngoài khoảng dữ liệu, thông tin shop không ghi (giới tính, địa chỉ, lý do huỷ / trả, nhóm hàng ...),
  câu hỏi chủ quan.

## Chưa làm (đang trả `abstain`)

Các câu diễn đạt quá lạ mà bước sửa lỗi chính tả không khôi phục được; xếp hạng bị hoà ở vị trí cắt;
công nợ tại một ngày trước ngày cuối dữ liệu khi sổ chỉ ghi số còn nợ lúc cuối.

## Kết quả

Tập train (3.000 câu): 3.000 đúng, 0 sai, PA = 1,0.

Tập public (1.000 câu): bản nộp ngày 06/11/2026 (`result/submission_v7.jsonl`) trả lời đủ 1.000 câu, điểm 1,0.
Bản nộp trước đó (05/11/2026) trả lời 990 câu, điểm 0,972; 9 câu sai của bản đó đã được sửa (liệt kê tháng dùng
chung một năm `th7, th8, th9/2025`; `doanh thu bao nhieu` không dấu bị nhận nhầm là thứ Ba; công nợ ghi ở cột ghi chú).
