# Quy Trình Tấn Công FGSM Trực Quan

Đây là quy trình tấn công một ảnh bằng phương pháp FGSM, được trình bày một cách trực quan qua 4 bước.

### Bước 1: Ảnh Gốc & Dự Đoán Đúng

- Bắt đầu với một ảnh gốc từ tập dữ liệu (ví dụ: chữ số "3").
- Đưa ảnh này qua mô hình LeNet đã được huấn luyện.
- **Kết quả:** Mô hình nhận diện **ĐÚNG** ảnh là số "3".

![Ảnh gốc](https://i.imgur.com/L8eA2gH.png)

*Mô hình tự tin rằng đây là số 3.*

---

### Bước 2: Tính Toán & Tạo Nhiễu FGSM

- Dựa trên ảnh gốc, nhãn thật ("3"), và cấu trúc bên trong của mô hình, ta sử dụng thuật toán FGSM để tính toán một "mặt nạ nhiễu" (noise).
- Nhiễu này không phải ngẫu nhiên, mà được thiết kế để tối đa hóa lỗi của mô hình.
- **Bản chất:** Nhiễu là một ma trận các giá trị rất nhỏ (ví dụ: ±0.3), cho biết cần tăng hay giảm độ sáng của từng pixel trên ảnh gốc.

![Nhiễu FGSM](https://i.imgur.com/x8bF9zS.png)

*Đây là "vũ khí" của chúng ta. Các vùng màu xanh/đỏ tương ứng với việc giảm/tăng độ sáng pixel.*

---

### Bước 3: Tạo Ảnh Đối Nghịch (Adversarial Image)

- Lấy **Ảnh Gốc** cộng với **Nhiễu FGSM**.
- **Kết quả:** Một "ảnh đối nghịch" mới. Bằng mắt thường, ảnh này gần như không thể phân biệt được với ảnh gốc.

![Ảnh bị tấn công](https://i.imgur.com/3yS3J8R.png)

*Trông vẫn là số 3, phải không? Nhưng bên trong, giá trị của từng pixel đã bị thay đổi một chút.*

---

### Bước 4: Dự Đoán Sai

- Đưa ảnh đối nghịch vừa tạo vào lại mô hình.
- **Kết quả:** Mô hình bây giờ bị "lừa". Mặc dù ảnh trông vẫn là số "3", mô hình lại dự đoán nó là một số khác (ví dụ: "5") với độ chắc chắn cao.

![Dự đoán sai](https.imgur.com/3yS3J8R.png)

*Cuộc tấn công thành công! Mô hình đã bị đánh lừa hoàn toàn.*

---

### Tóm Tắt: So Sánh Trước và Sau

| Ảnh Gốc (Đúng) | + | Nhiễu FGSM | = | Ảnh Bị Tấn Công (Sai) |
| :---: | :-: | :---: | :-: | :---: |
| ![Original](https://i.imgur.com/L8eA2gH.png) | **+** | ![Noise](https://i.imgur.com/x8bF9zS.png) | **=** | ![Adversarial](https://i.imgur.com/3yS3J8R.png) |
| **Đoán: 3** ✅ | | | | **Đoán: 5** ❌ |

Quy trình này cho thấy các mô hình AI có thể "mong manh" như thế nào trước những thay đổi nhỏ và có chủ đích.

