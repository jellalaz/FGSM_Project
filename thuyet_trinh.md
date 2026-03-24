# KỊCH BẢN THUYẾT TRÌNH DEMO TẤN CÔNG FGSM

---

### **PHẦN 1: GIỚI THIỆU MÔ HÌNH & DEMO TRƯỚC TẤN CÔNG**
**Người trình bày:** Trường Giang

---

**[Slide 1: Lời chào & Giới thiệu]**

**Trường Giang:** "Em chào cô và các bạn. Em là Trường Giang, đại diện cho nhóm. Hôm nay, nhóm chúng em sẽ trình bày một demo thú vị về một khía cạnh quan trọng trong lĩnh vực Trí tuệ nhân tạo, đó là tính bảo mật và bền vững của các mô hình học máy."

"Bài demo của chúng em gồm 2 phần chính:"
1.  "**Phần 1:** Em sẽ giới thiệu mô hình nhận diện chữ số viết tay, cách nó được huấn luyện và demo khả năng nhận diện chính xác của nó trong điều kiện bình thường."
2.  "**Phần 2:** Bạn [Tên bạn tiếp theo] sẽ tiếp nối, trình bày về một kỹ thuật tấn công đơn giản nhưng rất hiệu quả có tên là FGSM, và cho thấy mô hình của chúng ta dễ bị 'đánh lừa' như thế nào."

---

**[Slide 2: Bài toán & Dữ liệu]**

**Trường Giang:** "Để cho dễ hình dung, chúng ta hãy bắt đầu với một bài toán rất quen thuộc: **Nhận diện chữ số viết tay**. Mục tiêu là dạy cho máy tính có thể 'đọc' được các chữ số từ 0 đến 9 do con người viết."

"Để 'dạy' cho máy, chúng ta cần dữ liệu. Ở đây, nhóm em sử dụng bộ dữ liệu kinh điển tên là **MNIST**. Cô và các bạn có thể hình dung nó như một bộ sách giáo khoa cho các mô hình AI về thị giác máy tính. Bộ dữ liệu này chứa:"
*   **60,000 ảnh** để huấn luyện (gọi là tập training).
*   **10,000 ảnh** để kiểm tra (gọi là tập test).

"Sau khi dùng 60,000 ảnh kia để 'dạy' cho mô hình, chúng ta sẽ có được một **file trọng số** - có thể coi là 'bộ não' của mô hình, chứa đựng toàn bộ kiến thức nó đã học."

---

**[Slide 2.5: Kiến trúc Mô hình (LeNet)]**

**Trường Giang:** "Vậy 'bộ não' mà chúng ta nói đến có hình dạng như thế nào? Nhóm em sử dụng một kiến trúc mạng nơ-ron (CNN) rất nổi tiếng tên là **mạng LeNet**. Đây chính là **khung sườn** cho bộ não của chúng ta."

"Cô và các bạn có thể hình dung LeNet như một dây chuyền xử lý ảnh:"
1.  "**Đầu vào:** Là ảnh chữ số 28x28 pixel."
2.  "**Các lớp Tích chập & Gộp:** Các lớp này sẽ quét qua ảnh để tìm các đặc trưng cơ bản như cạnh, đường cong, góc... rồi tóm tắt chúng lại."
3.  "**Các lớp Kết nối đầy đủ:** Các lớp cuối cùng này sẽ 'tổng hợp' thông tin và đưa ra quyết định cuối cùng: 'Đây là số mấy?'."

"Kiến trúc LeNet này định nghĩa **cách** mô hình sẽ xử lý thông tin. Nhưng chính xác thì nó lấy cái gì từ **file trọng số**?"

"File trọng số sẽ cung cấp các giá trị số đã được 'tôi luyện' qua quá trình huấn luyện, cụ thể là **weights (trọng số)** và **biases (thiên vị)** cho từng lớp trong kiến trúc:"
*   "**Với các lớp Tích chập:** File trọng số cung cấp giá trị cho các **bộ lọc (filter)**. Đây là những ma trận số đã học được cách nhận diện các đường nét, góc cạnh."
*   "**Với các lớp Kết nối đầy đủ:** File trọng số cung cấp giá trị cho **ma trận kết nối** khổng lồ, quyết định xem việc kết hợp các đặc trưng lại với nhau sẽ cho ra kết quả là số mấy."

"Nói cách khác, code Python định nghĩa **khung sườn**, còn file trọng số nạp **bộ não** vào. Bây giờ, chúng ta sẽ kiểm tra mô hình hoàn chỉnh này."

---

**[Slide 3: Demo trước tấn công]**

**Trường Giang:** "Và bây giờ là phần quan trọng nhất của Phần 1. Em sẽ dùng 'bộ não' vừa huấn luyện để nhận diện 10,000 ảnh trong tập test mà nó chưa từng thấy bao giờ."

**[Chuyển sang màn hình demo code]**

"Trên màn hình là môi trường code của chúng em. Em sẽ chạy đoạn code để nạp mô hình và bắt đầu quá trình nhận diện."

**[Chạy cell code demo TRƯỚC tấn công]**

"Như cô và các bạn có thể thấy trên màn hình:"
*   "Với mỗi ảnh chữ số (cột 'Thật'), mô hình đều đưa ra dự đoán (cột 'Đoán') một cách chính xác."
*   "Ví dụ ảnh số 7, mô hình đoán là 7. Ảnh số 2, mô hình đoán là 2. Tất cả đều có dấu tick xanh ✅, biểu thị cho việc đoán đúng."
*   "Trên thực tế, độ chính xác tổng thể của mô hình này trên toàn bộ 10,000 ảnh test là khoảng **99%**. Một con số rất ấn tượng, cho thấy mô hình hoạt động rất tốt trong điều kiện lý tưởng."

"Điều này chứng tỏ mô hình của chúng em đã được huấn luyện cẩn thận và có khả năng nhận diện chữ số với độ tin cậy rất cao."

"Phần trình bày của em đến đây là kết thúc. Nhưng, liệu mô hình có thực sự 'thông minh' và 'bền vững' như chúng ta nghĩ? Tiếp theo, em xin nhường lại sân khấu cho bạn **[Tên bạn tiếp theo]** để khám phá câu trả lời trong Phần 2."

---
---

### **PHẦN 2: TẤN CÔNG FGSM & ĐÁNH GIÁ KẾT QUẢ**
**Người trình bày:** [Tên bạn tiếp theo]

---

**[Slide 4: Giới thiệu Tấn công đối nghịch]**

**[Tên bạn tiếp theo]:** "Cảm ơn bạn Giang. Em chào cô và các bạn, em là [Tên bạn tiếp theo]. Như bạn Giang vừa trình bày, mô hình của chúng ta đang hoạt động rất tốt với độ chính xác gần như tuyệt đối. Nhưng liệu nó có an toàn trước những kẻ muốn phá hoại?"

"Câu trả lời là không. Các mô hình AI, dù mạnh mẽ đến đâu, cũng có những 'điểm mù', những lỗ hổng có thể bị khai thác. Kỹ thuật khai thác những lỗ hổng này được gọi là **Tấn công đối nghịch (Adversarial Attack)**."

"Ý tưởng rất đơn giản: Chúng ta sẽ tạo ra một loại **nhiễu loạn (noise)** đặc biệt, được tính toán một cách có chủ đích. Khi cộng nhiễu này vào ảnh gốc, mắt người chúng ta gần như không thể nhận ra sự khác biệt. Nhưng đối với mô hình AI, sự thay đổi nhỏ này lại khiến nó đưa ra dự đoán sai hoàn toàn."

"Phương pháp tấn công mà nhóm em sử dụng hôm nay là **Fast Gradient Sign Attack (FGSM)** - một trong những phương pháp đầu tiên và nổi tiếng nhất."

---

**[Slide 5: Demo sau tấn công]**

**[Tên bạn tiếp theo]:** "Nguyên lý của FGSM khá toán học, nhưng ý tưởng chính là: **thay vì chỉnh sửa trọng số để giảm sai sót (khi training), chúng ta lại chỉnh sửa ảnh đầu vào để tối đa hóa sai sót.**"

"Bây giờ, em sẽ tiến hành 'tấn công' vào những bức ảnh mà bạn Giang vừa demo."

**[Chuyển sang màn hình demo code]**

"Đầu tiên, em sẽ chạy code để tạo ra nhiễu FGSM từ chính những ảnh gốc đó."

**[Chạy cell code tạo nhiễu FGSM]**

"Đây là phần nhiễu được tạo ra. Nó trông giống như những chấm nhiễu ngẫu nhiên, nhưng thực chất mỗi chấm đều được tính toán để 'đẩy' dự đoán của mô hình ra xa khỏi sự thật."

"Tiếp theo, em sẽ cộng phần nhiễu này vào ảnh gốc và cho mô hình nhận diện lại."

**[Chạy cell code demo SAU tấn công]**

"Và đây là kết quả. Thật đáng kinh ngạc!"
*   "Vẫn là những ảnh đó, mắt chúng ta vẫn nhận ra rõ ràng là số 7, 2, 1... Nhưng bây giờ, mô hình lại 'đọc' chúng thành những con số hoàn toàn khác: số 7 giờ bị đoán là 9, số 2 bị đoán là 7, số 1 bị đoán là 8..."
*   "Hầu hết các dự đoán bây giờ đều sai, được đánh dấu bằng dấu chéo đỏ ❌."

---

**[Slide 6: So sánh & Đánh giá]**

**[Tên bạn tiếp theo]:** "Để thấy rõ hơn sự sụp đổ của mô hình, chúng ta hãy xem biểu đồ so sánh."

**[Chạy cell code vẽ biểu đồ]**

"Biểu đồ này cho thấy độ chính xác của mô hình giảm mạnh như thế nào khi mức độ nhiễu (epsilon) tăng lên."
*   "Khi không có nhiễu (epsilon=0), độ chính xác là **99%**."
*   "Chỉ với một chút nhiễu (epsilon=0.1), độ chính xác đã giảm xuống còn **87%**."
*   "Và với mức nhiễu cao hơn một chút (epsilon=0.3), độ chính xác sụp đổ chỉ còn khoảng **14%** - tức là không hơn gì đoán mò."

---

**[Slide 7: Kết luận]**

**[Tên bạn tiếp theo]:** "Qua bài demo này, nhóm chúng em muốn gửi đến cô và các bạn một thông điệp quan trọng: **Độ chính xác cao không đồng nghĩa với sự an toàn.**"

"Các mô hình AI có thể cực kỳ nhạy cảm với những thay đổi nhỏ, có chủ đích ở đầu vào. Điều này mở ra nhiều rủi ro bảo mật trong các ứng dụng thực tế như xe tự lái, nhận diện khuôn mặt hay chẩn đoán y tế."

"Việc nghiên cứu các phương pháp tấn công như FGSM không phải để phá hoại, mà là để hiểu rõ hơn về điểm yếu của mô hình, từ đó tìm ra các phương pháp phòng thủ hiệu quả, xây dựng những hệ thống AI không chỉ thông minh mà còn thực sự **bền vững và đáng tin cậy**."

"Phần trình bày của nhóm em đến đây là kết thúc. Em xin chân thành cảm ơn cô và các bạn đã chú ý lắng nghe."

