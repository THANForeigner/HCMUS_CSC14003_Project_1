# Ảnh Hưởng Của Scale Động (Dynamic Parameters) Trong Tối Ưu Hóa Liên Tục

Việc chuyển các tham số tĩnh (hardcoded) sang **Scale Động (phụ thuộc vào Số Chiều D và Miền Giá Trị)** mang lại tác động "sống còn" đối với các thuật toán mưu sinh dựa trên không gian như ABC, Firefly và Hill Climbing. 

Dưới đây là so sánh chi tiết giữa bản **Cũ (Tĩnh)** và **Mới (Động)**, cùng lý do tại sao bộ dữ liệu benchmark của bạn lại thay đổi ngoạn mục như vậy:

---

## 1. Thuật toán ABC (Giới hạn `limit`)
**`limit`**: Là số lần một con ong thợ cố gắng tìm kiếm quanh một nguồn mật (nghiệm) mà không thấy kết quả tốt hơn, trước khi từ bỏ và hóa thành "ong trinh sát" bay đi tìm nguồn mới ngẫu nhiên.

*   **Bản Cũ (Tĩnh: `limit = 20`)**: 
    *   *Tác động:* Ở bài toán 2D, 20 lần thử là quá đủ để kết luận nguồn mật đó đã cạn. **Nhưng ở bài toán 20D**, không gian tìm kiếm là một khối cầu khổng lồ 20 chiều, 20 lần bay ngẫu nhiên quanh nghiệm đó giống như "mò kim đáy biển". Ong thợ **chưa kịp tìm hết** các hướng tiềm năng đã quá vội vã vứt bỏ nghiệm đó để đi chỗ khác.
    *   *Hậu quả:* ABC biến thành Random Search (chỉ bay ngẫu nhiên), mất đi khả năng khai thác cục bộ (Exploitation), biểu đồ Success Rate rớt thê thảm.
*   **Bản Mới (Động: `limit = n_bees/2 * D`)**: 
    *   *Tác động:* Ở 20D, `limit = 15 * 20 = 300`. Bây giờ, ong thợ sẽ có 300 lần kiên nhẫn bay dò dẫm trong cái "biển" 20 chiều đó trước khi bỏ cuộc. 
    *   *Kết quả thực nghiệm:* Sự cân bằng giữa Khám phá (Exploration - tìm mỏ mới) và Khai thác (Exploitation - đào mỏ cũ) được phục hồi, ABC bám đuổi các hàm khó như Ackley bền bỉ hơn ở các Dimensionality cao.

## 2. Thuật toán Firefly (Hấp thụ ánh sáng `gamma`)
**`gamma`**: Mức độ suy giảm sức hút của một con đom đóm khi khoảng cách tăng lên.
$$ \beta = \beta_0 \cdot e^{-\gamma \cdot r^2} $$
Trong đó $r$ là khoảng cách giữa 2 con đom đóm trong không gian chuẩn Euclidean.

*   **Bản Cũ (Tĩnh: `gamma = 0.01`)**:
    *   *Tác động:* $0.01$ là con số phù hợp trên các hàm có miền giá trị cỡ $[-10, 10]$ (ví dụ Rastrigin). Lượng suy giảm là hợp lý. Nhưng với hàm **Griewank có miền $[-600, 600]$**, khoảng cách $r$ giữa hai con đom đóm có thể lên tới hàng nghìn.
    *   *Hậu quả:* $r^2$ sẽ thành một số cực lớn (vd: $1,000,000$). Công thức trở thành $e^{-0.01 \cdot 1000000} \approx 0$. Tức là **sức hút bằng 0 ngay lập tức**. Đom đóm **không nhìn thấy nhau** và tuyệt đối không bay về phía nhau. Các con đom đóm chỉ đứng im hoặc bay lung tung ngẫu nhiên tốn thời gian.
*   **Bản Mới (Động: `gamma = 1 / sqrt(ub - lb)`)**:
    *   *Tác động:* Với miền $[-600, 600] \Rightarrow gamma \approx 1/\sqrt{1200} \approx 0.028$. Với miền $[-5, 5] \Rightarrow gamma \approx 0.3$. 
    *   *Kết quả thực nghiệm:* Bất chấp cái hộp không gian (Domain) to bằng cái sân vận động hay bé cái lỗ kim, các con Đom Đóm luôn duy trì được "tầm nhìn" vừa đủ để kéo nhau về phía một điểm sáng.

## 3. Thuật toán Hill Climbing (Bước nhảy `step_size`)
**`step_size`**: Kích thước sải chân (bán kính vùng lân cận) mỗi khi thuật toán dò tìm hướng đi xuống núi.

*   **Bản Cũ (Tĩnh: `step_size = 0.5`)**:
    *   *Tác động:* Đối với hàm Sphere $[-5.12, 5.12]$, bước $0.5$ nhảy được $1/20$ toàn bản đồ. Thuật toán chạy mượt. Nhưng với hàm Griewank $[-600, 600]$, bước nhảy $0.5$ giống như một con kiến đi bộ từ Bắc vào Nam.
    *   *Hậu quả:* HC cần tới $1,000,000$ vòng lặp mới bò nổi xuống đáy thung lũng. Vì bị kẹt giới hạn vòng lặp sớm nên Fitness trả về tệ không khác gì Random Search.
*   **Bản Mới (Động: `step_size = 5% * (ub - lb)`)**:
    *   *Tác động:* Bước nhảy bây giờ là linh động: Căng ra to đùng ($60$) trên Griewank, và tự thu hẹp lại ($0.512$) trên Sphere. 
    *   *Kết quả thực nghiệm:* HC giờ đây lướt cực nhanh (theo đúng nghĩa leo núi) xuống vùng có chứa đáy thung lũng (dù bị mắc kẹt do đặc tính Local Optima), chứ không còn lặn hụp ở tận "đỉnh mây" khi hết thời gian nữa.
