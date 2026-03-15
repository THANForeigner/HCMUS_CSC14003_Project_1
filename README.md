# Dự án Thuật toán tìm kiếm (Search Algorithms Project)

Dự án này triển khai các thuật toán cổ điển và thuật toán lấy cảm hứng từ tự nhiên để giải quyết nhiều bài toán tối ưu hóa khác nhau, bao gồm: Bài toán Cái túi (Knapsack), Đường đi ngắn nhất (Shortest Path), Tô màu đồ thị (Graph Coloring), Người chào hàng (TSP) và Tối ưu hóa hàm liên tục.

## Cấu trúc Dự án

- `classical/`: Các thuật toán tìm kiếm cổ điển (Tìm kiếm có đối chiếu, Tìm kiếm mù, Tìm kiếm cục bộ).
- `nature_inspired/`: Các thuật toán lấy cảm hứng từ tự nhiên (Dựa trên sinh học, Tiến hóa, Con người, Vật lý).
- `problems/`: Định nghĩa các bài toán và các công cụ kiểm tra dữ liệu đầu vào.
- `Test/`: Các kịch bản đánh giá (benchmark) và trường hợp kiểm thử cho từng loại bài toán.
- `visualization/`: Công cụ trực quan hóa dựa trên GUI cho các thuật toán tối ưu hóa liên tục.
- `main.py`: Điểm bắt đầu chính của chương trình với menu giao diện dòng lệnh (CLI) tương tác.

## Điều kiện tiên quyết

- Python 3.8+
- Khuyến nghị: Sử dụng môi trường ảo (`venv`)

### Các thư viện phụ thuộc

Cài đặt các thư viện cần thiết bằng lệnh `pip`:

```bash
pip install -r requirements.txt
```

## Cách chạy

### 1. Main Menu
Cách dễ nhất để khám phá dự án là thông qua menu CLI (giao diện dòng lệnh) tương tác tại thư mục gốc. Menu này cho phép bạn chạy kiểm thử cho tất cả các bài toán và khởi chạy công cụ trực quan hóa.

```bash
python main.py
```

### 2. Chạy Kiểm tra đánh giá (Benchmarks)
Bạn có thể chạy các bài kiểm tra cụ thể thông qua menu `main.py` hoặc thực thi trực tiếp tệp `main.py` trong thư mục test của từng bài toán.

#### Thông qua Menu chính:
Chạy `python main.py` và chọn các tùy chọn từ 1-6.

#### Chạy thủ công:
Di chuyển đến thư mục test cụ thể và chạy tệp `main.py` tương ứng:
- **Knapsack:** `python Test/Knapsack/main.py`
- **Shortest Path:** `python Test/Shortest_Path/main.py`
- **Graph Coloring:** `python Test/Graph_Coloring/main.py`
- **TSP:** `python Test/Traveling_Sale_Man/main.py`
- **Continuous Optimization:** `python Test/Continuous_Optimization/main.py`

### 3. Chạy Trực quan hóa (Visualization)
Công cụ trực quan hóa trình diễn cách các thuật toán lấy cảm hứng từ tự nhiên hội tụ trên các hàm số liên tục 3D khác nhau (Sphere, Rastrigin, v.v.).

#### Thông qua Menu chính:
Chạy `python main.py` và chọn **7**.

#### Chạy thủ công:
Chạy trực tiếp kịch bản trực quan hóa từ thư mục gốc:
```bash
python visualization/main.py
```

## Các thuật toán được hỗ trợ

- **Cổ điển (Classical):** A*, BFS, DFS, UCS, Hill Climbing.
- **Lấy cảm hứng từ tự nhiên (Nature-Inspired)**
  - **Dựa trên sinh học (Biology-based):** Particle Swarm Optimization (PSO), Artificial Bee Colony (ABC), Firefly Algorithm (FA), Cuckoo Search (CS), Ant Colony Optimization (ACO).
  - **Dựa trên tiến hóa (Evolution-based):** Genetic Algorithm (GA), Differential Evolution (DE).
  - **Dựa trên con người (Human-based):** Teaching-Learning Based Optimization (TLBO).
  - **Dựa trên vật lý (Physics-based)** Simulated Annealing (SA).
