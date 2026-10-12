# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Báo cáo dựa trên lần chạy Waymo thực tế của repo này. Phần code và giải thích được Codex hỗ trợ soạn; học viên cần đọc, kiểm tra và giải thích được bài trong vấn đáp theo RULES.md.

## Thông tin học viên

- Họ tên: Nguyễn Văn Hồng
- MSSV: 2A202602800
- Email: 26ai.hongnv@vinuni.edu.vn
- Link repo (fork): https://github.com/hongneuk65/K4-L2L3-DAY23-NguyenVanHong-2A202602800-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): dùng HEAD sau commit CP6 khi nộp trên LMS; commit code và artifacts được kiểm chứng là `6cac944357d8816f0e72a6e9668bfbb3ce0c78c6`. Hash cuối cùng không tự ghi vào chính commit chứa báo cáo; lấy bằng lệnh `git rev-parse HEAD` sau khi hoàn tất.

## Tóm tắt kết quả

- `fusion_mode`: `compare`; `frames`: `[0, 198]` (199 frame/mode); `seed`: `0`.
- `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- `detection.precision`: 0.9700934579439252; `detection.recall`: 0.7004048582995951; `detection.tp/fp/fn`: 519/16/222.
- `tracking.lidar`: `rmse` = 0.15032270040636328; `matches` = 502; `sum_sq_err` = 11.343650957245547; `ghost_track_frames` = 0; `missed_gt_frames` = 239; `mean_confirmed_tracks` = 2.522613065326633.
- `tracking.fused`: `rmse` = 0.13586678029276522; `matches` = 502; `sum_sq_err` = 9.266810557535512; `ghost_track_frames` = 0; `missed_gt_frames` = 239; `mean_confirmed_tracks` = 2.522613065326633.
- `precision_track` LiDAR/fused: 1.000000000/1.000000000; `coverage` LiDAR/fused: 0.967244701/0.967244701.
- `rmse_fused - rmse_lidar`: -0.014455920 m; điều kiện không xấu hơn quá 0.05 m: đạt.

RMSE fused thấp hơn LiDAR 0.014455920 m. Hai mode có 502/502 cặp ghép, 0/0 ghost-track-frame và 239/239 missed-GT-frame. Vì vậy cần xem cả khả năng duy trì track và số cặp đánh giá cùng với sai số vị trí. Hai mode dùng cùng detector và cùng frame nên tổng detection giống nhau; khác biệt tracking xuất phát từ các update camera và ảnh hưởng tiếp theo lên trạng thái/gán đo. Đo 2D có nhiễu và tâm hộp ảnh không nhất thiết trùng phép chiếu tâm hộp 3D, nên camera không bảo đảm làm RMSE giảm. Đây là giải thích khả dĩ từ mô hình đo, không phải kết luận đã kiểm chứng nguyên nhân bằng thí nghiệm calibration.

RMSE là `sqrt(sum_sq_err/matches)` trên khoảng cách vị trí 3D, chỉ tính confirmed tracks được ghép một-một với GT trong gate XY 2.0 m. `precision_track = matches/(matches+ghost_track_frames)`; `coverage = matches/detection.tp` theo RUBRIC.md. Miss gồm cả giai đoạn track chưa confirmed và những nhãn không được confirmed track ghép; không thể quy toàn bộ miss cho detector.

`grade_run.log` có 398 record, đúng một record cho mỗi `(mode, frame)`. Đã đối chiếu các invariant `matches+ghosts==confirmed`, `matches+misses==valid_gt`, tổng số đếm, tổng sai số và trung bình confirmed với metrics. Các file metrics/log riêng từng mode cũng được kiểm tra và khớp file compare. Frame đầu có confirmed track là 4 ở mode LiDAR và 4 ở mode fused.

Camera dùng tâm hộp ground-truth FRONT có nhiễu seeded, không dùng camera detector. Kết quả này không đo chất lượng một hệ thống perception độc lập với ground truth.

### Tái lập lần chạy

Chạy từ gốc repo trên PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest student/tests -q --disable-warnings --basetemp .pytest_cache/lab-temp
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
.\.venv\Scripts\fusion-run-lab.exe --config student/config/paths.yaml --fusion compare --seed 0
.\.venv\Scripts\python.exe -X utf8 tools/check_submission.py
```

Cấu hình local (không commit):

```yaml
waymo_dir: ../data/Day23-Fusion/Waymo
weights_dir: ../data/Day23-Fusion/weights
segment: training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
frame_start: 0
frame_end: 198
```

Môi trường Python 3.12.10, PyTorch CPU. Venv dùng các thư viện khoa học có sẵn trên máy và cài các dependency còn thiếu của repo. Có thể cài lại theo README bằng `python -m pip install -e platform/third_party/waymo_reader -e platform -e student pytest` trong venv Python 3.12 mới. Số thread chỉ giới hạn tài nguyên CPU, không thay thuật toán hoặc tham số tracking.

Trên Windows, `-X utf8` tránh lỗi mã hóa khi công cụ kiểm tra in tiếng Việt. `--basetemp` đặt thư mục tạm của pytest trong repo, tránh lỗi quyền truy cập thư mục tạm mặc định; thư mục `.pytest_cache/` đã được Git bỏ qua. Lần kiểm tra lại nhận 128 passed, không failed/xfailed. Test có cảnh báo NumPy matrix và phép toán với tọa độ không hợp lệ trong các ca kiểm tra; sau khi kết thúc còn thông báo từ destructor của pyreadline3, nhưng pytest trả exit code 0.

### Đối chiếu checkpoint và bằng chứng

- CP0: cấu hình local trỏ đúng segment và weights; thông tin học viên đã điền.
- CP1: sáu hàm EKF trong `workspace/kalman.py`; commit `6b9f369`.
- CP2: ba hàm FOV, projection và measurement camera trong `workspace/camera_fusion.py`; commit `4e80557`.
- CP3: năm hàm gating và association trong `workspace/association.py`; commit `22ec6af`.
- CP4: ba hàm khởi tạo, score và xóa trong `workspace/track_management.py`; commit `b8b45f6`.
- CP5: sáu artifacts của lần chạy compare 0–198, seed 0; commit `6cac944`.
- CP6: báo cáo, khai báo AI và kiểm tra nộp bài; lấy HEAD cuối cùng sau commit để nộp LMS.

Thứ tự predict → LiDAR → camera nằm ở `platform/fusion_lab/scripts/run_lab.py`, dòng 200, 202 và 209. Công thức EKF nằm ở `workspace/kalman.py`, dòng 20–130; gating và vòng gán nằm ở `workspace/association.py`, từ dòng 25; điều kiện lifecycle nằm ở `workspace/track_management.py`, từ dòng 41. Các đường dẫn và dòng này là bằng chứng để đối chiếu khi vấn đáp.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Đo LiDAR 3D và camera 2D.** LiDAR có `z=[x_s,y_s,z_s]ᵀ`, đơn vị m, `R` 3×3 với phương sai mặc định 0.01 m²/trục; `h(x)` là vị trí đổi sang hệ sensor và `H` tuyến tính 3×6. Camera có `z=[u,v]ᵀ`, đơn vị pixel, `R=diag(25,25)` pixel². `h(x)` là pinhole nên phi tuyến; platform cung cấp Jacobian 2×6 tại trạng thái hiện tại. Cả hai cập nhật cùng trạng thái 6D. Trong `kalman.py`, `γ=z-h(x)`, `S=HPHᵀ+R` và `K=PHᵀS⁻¹`; code dùng giải hệ tuyến tính thay cho tính nghịch đảo trực tiếp.

2. **Gating Mahalanobis.** Khoảng cách bình phương `d²=γᵀS⁻¹γ` chuẩn hóa residual theo bất định của track và sensor. Cặp ngoài cổng χ² bị loại trước greedy matching, giảm nguy cơ gán đo xa hoặc không tương thích. `association.py` lấy ngưỡng `chi2.ppf(0.995, sensor.dim_meas)`, nên camera 2D và LiDAR 3D có ngưỡng khác nhau. Khi P lớn, cùng residual thường cho d² nhỏ hơn; Euclidean không thể hiện khác biệt này. Kiểm tra FOV trước Mahalanobis để không chiếu điểm sau camera hoặc độ sâu không hợp lệ.

3. **Track-then-fuse.** Chỉ một danh sách track: `run_lab.py` predict từng track một lần, gọi association LiDAR và lifecycle, rồi gọi association camera trên chính các track đó. Camera tinh chỉnh trạng thái đã được LiDAR cập nhật. Không predict thêm giữa hai sensor vì measurement cùng frame có cùng thời gian logic `t=frame_index*dt`. Log có các record `mode=lidar` và `mode=fused` cho frame 0–198 để so sánh kết quả; log tổng hợp không ghi từng thao tác predict/update, nên thứ tự được xác nhận bằng code tại `platform/fusion_lab/scripts/run_lab.py`, không suy ra riêng từ RMSE.

4. **Camera lệch calibration.** Extrinsic/intrinsic sai làm `h(x)` lệch so với pixel đo, gây innovation có xu hướng lệch có hệ thống hoặc tăng biên độ. Khi d² vượt gate, camera bị từ chối nhiều hơn; nếu vẫn trong gate, update có thể kéo trạng thái lệch và tăng RMSE. R lớn hoặc P lớn có thể làm gate chấp nhận residual đáng kể. Cần so sánh residual theo thời gian và thử nhiều mức lệch để kiểm chứng; bài này không thực hiện bonus calibration.

5. **Sensor tường minh ở frame rỗng.** `meas_list=[]` không chứa đối tượng nào để suy ra lượt hiện tại là LiDAR hay camera. Tham số `sensor` giúp luôn gọi `manager.manage_tracks(...)` đúng lượt: LiDAR rỗng vẫn trừ score các track không ghép trong FOV và đánh giá xóa; camera rỗng không thay lifecycle. LiDAR quyết định hit/miss, score, init và delete. Camera hit chỉ EKF update, không tạo track hay cộng score. `association.py` gọi `handle_updated_track(track, sensor)`; platform manager chỉ cập nhật score khi sensor là LiDAR.

6. **Xác nhận, giữ và xóa track.** Khởi tạo từ LiDAR có score `1/6`, state `initialized`, vận tốc 0. Position/covariance đo được đổi bằng `sens_to_veh`, covariance vị trí là `R_rotation*R_meas*R_rotationᵀ`; phương sai vận tốc ban đầu là `50²,50²,5²`. Hit cộng `1/6`, tối đa 1; track chưa confirmed có hit trở thành tentative, và được confirmed khi score **>0.8**. Miss trong FOV LiDAR trừ `1/6`, nhưng track đã confirmed không bị hạ state. Xóa khi `Pxx>9` hoặc `Pyy>9`, hoặc confirmed có score **<0.6**, hoặc chưa confirmed có score **<=0**. Các điều kiện là OR; tại đúng ngưỡng variance 9 và score confirmed 0.6 chưa xóa. Các điều kiện nằm trong `track_management.py`, được platform manager áp dụng chỉ ở lượt LiDAR.

## Bonus (không bắt buộc)

- Không. Hoàn thành phần bắt buộc; không thay calibration hoặc detector để tối ưu số liệu.

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): OpenAI Codex.
- Dùng cho phần nào (hàm, câu hỏi, debug): đọc tài liệu và mã repo; implement 17 hàm Part E–H; xử lý môi trường Windows; chạy test và Waymo; soạn phần giải thích và tổng hợp báo cáo từ artifacts thực tế; chuẩn bị commit và repo nộp.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Codex đã chạy bộ test gốc trong repo với **128 passed**, không failed/xfailed; chạy thử 21 frame rồi chạy chính thức compare 0–198, seed 0; kiểm tra metrics/log bằng validator của platform, đối chiếu cả artifacts riêng từng mode. Học viên cần tự đọc và xác nhận hiểu code, công thức, phần giải thích trước khi nộp/vấn đáp; báo cáo không khẳng định việc tự kiểm tra của học viên đã hoàn tất.

## Checklist nộp

- [x] Part E–H đã implement; 128 test pass, không failed/xfailed.
- [x] Giữ nguyên Part A–D, platform và bộ test.
- [x] Lần chạy chấm điểm compare, seed 0, frame_start 0, frame_end 198.
- [x] Đã commit đủ sáu artifacts do runner sinh; không sửa tay metrics/log.
- [x] Đã điền thông tin, kết quả, sáu câu giải thích và khai báo AI.
- [x] Dữ liệu Waymo, weights, paths.yaml, .env được giữ local và không commit.
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`; toàn bộ kiểm tra đã pass sau khi commit báo cáo và artifacts.
- [x] Repo GitHub đã có code và artifacts; commit remote main được kiểm tra trực tiếp. Sau mỗi sửa báo cáo, push và xác nhận HEAD cuối cùng trước khi nộp.
- [ ] Đã nộp link repo + commit hash trên LMS (chưa có bằng chứng nộp LMS).
