#!/usr/bin/env python3
"""
Complete grading script: Detect circled answers using YOLO and grade against answer key.
"""
from ultralytics import YOLO
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
model_path = "model_yolo26n/best.pt"
image_path = "/Users/donhuvy/Downloads/20250218_122310982_iOS.jpg"

# Đáp án đúng (chỉ 4 câu vì ảnh chỉ chụp trang 1, câu 5 chưa có đáp án trên ảnh)
# Dựa trên kiến thức Sinh học lớp 12:
# Câu 1: Đoạn ADN mang thông tin mã hoá... → C. Mã di truyền (Gen là cả đoạn ADN, mã di truyền = codon trên mARN mã hoá 1 aa)
# Câu 2: Quá trình nhân đôi ADN diễn ra ở → D. nhân tế bào  
# Câu 3: Câu nào sau đây đúng nhất → B. ADN chứa thông tin mã hoá cho việc gắn nối các aa...
# Câu 4: Dạng thông tin di truyền trực tiếp sử dụng trong tổng hợp prôtêin → B. mARN
ANSWER_KEY = {
    1: "C",
    2: "D",   
    3: "B",
    4: "B",
}

# ============================================================================
# LOAD MODEL & RUN INFERENCE
# ============================================================================
print("=" * 70)
print("   CHẤM ĐIỂM TRẮC NGHIỆM BẰNG AI (YOLO v26n)")
print("=" * 70)

model = YOLO(model_path)
results = model(image_path, conf=0.25, verbose=False)

# ============================================================================
# PARSE DETECTIONS & MAP TO QUESTIONS
# ============================================================================
detections = []
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        center_y = (xyxy[1] + xyxy[3]) / 2
        center_x = (xyxy[0] + xyxy[2]) / 2
        detections.append({
            "class": cls_name,
            "confidence": conf,
            "box": xyxy,
            "center_y": center_y,
            "center_x": center_x,
        })

# Sort by Y position (top to bottom) to map to question order
detections.sort(key=lambda d: d["center_y"])

# Remove duplicate detections (same question, overlapping boxes)
# If two detections are very close in Y position, keep the one with higher confidence
filtered = []
for d in detections:
    is_duplicate = False
    for f in filtered:
        if abs(d["center_y"] - f["center_y"]) < 100 and abs(d["center_x"] - f["center_x"]) < 100:
            # Same question area - keep higher confidence
            if d["confidence"] > f["confidence"]:
                filtered.remove(f)
                filtered.append(d)
            is_duplicate = True
            break
    if not is_duplicate:
        filtered.append(d)

filtered.sort(key=lambda d: d["center_y"])

# ============================================================================
# GRADE
# ============================================================================
print(f"\n📄 Bài thi: Đề cương HK1 - Sinh lớp 12")
print(f"👤 Học sinh: Ngô Việt Anh")
print(f"📝 Lớp: 12D3")
print(f"📅 Ngày: 12/02/2025")
print(f"\n{'─' * 70}")
print(f"{'Câu':^6} │ {'Đáp án HS':^12} │ {'Độ tin cậy':^12} │ {'Đáp án đúng':^12} │ {'Kết quả':^10}")
print(f"{'─' * 70}")

correct = 0
total_graded = 0

for i, det in enumerate(filtered):
    q_num = i + 1
    student_answer = det["class"]
    conf = det["confidence"]
    
    if q_num in ANSWER_KEY:
        correct_answer = ANSWER_KEY[q_num]
        is_correct = student_answer == correct_answer
        if is_correct:
            correct += 1
            result_str = "✅ Đúng"
        else:
            result_str = "❌ Sai"
        total_graded += 1
        print(f"  {q_num:^4} │ {student_answer:^12} │ {conf:^12.1%} │ {correct_answer:^12} │ {result_str:^10}")
    else:
        print(f"  {q_num:^4} │ {student_answer:^12} │ {conf:^12.1%} │ {'N/A':^12} │ {'—':^10}")

print(f"{'─' * 70}")

# Calculate score
if total_graded > 0:
    score_per_question = 10.0 / len(ANSWER_KEY)
    total_score = correct * score_per_question
    
    print(f"\n📊 KẾT QUẢ CHẤM ĐIỂM:")
    print(f"   • Số câu đã detect: {len(filtered)}")
    print(f"   • Số câu đúng: {correct}/{total_graded}")
    print(f"   • Điểm số: {total_score:.2f}/10.00")
    print(f"   • Đánh giá: {'✅ ĐẠT' if total_score >= 5.0 else '❌ CHƯA ĐẠT'}")
else:
    print("\n⚠️  Không có câu nào để chấm!")

print(f"\n{'=' * 70}")
print(f"⚠️  LƯU Ý: Ảnh chỉ chụp trang 1 (4 câu). Cần thêm ảnh các trang tiếp")
print(f"   theo để chấm điểm toàn bộ bài thi.")
print(f"{'=' * 70}")
