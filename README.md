🧠 8-PUZZLE SOLVER - TRÒ CHƠI GIẢI ĐỐ DÁN TRÍ TUỆ NHÂN TẠO

💡 Giới Thiệu

Trò chơi "8-Puzzle Solver" là một ứng dụng Python sử dụng thư viện pygame để mô phỏng và minh hoạ lý thuyết về các thuật toán tìm kiếm trong AI.

Người dùng có thể chọn nhiều thuật toán khác nhau và theo dõi quá trình giải bài toán xếp hình 8-Puzzle trực quan trên màn hình.

🌐 Mục Tiêu

Hiểu cách hoạt động của các thuật toán tìm kiếm trong AI.

So sánh đường đi, thời gian, số bước giải quyết của mỗi thuật toán.

Tạo trò chơi mang tính tương tác và thực hành.

🔍 Các Thuật Toán Hỗ Trợ

1. Tìm kiếm không thông tin:

BFS (Breadth-First Search)

DFS (Depth-First Search)

UCS (Uniform Cost Search)

IDDFS (Iterative Deepening DFS)

DLS (Depth-Limited Search)

2. Tìm kiếm có thông tin:

A* Search

IDA* (Iterative Deepening A*)

Greedy Best-First Search

3. Tìm kiếm leo dốc:

Simple Hill Climbing

Steepest Ascent Hill Climbing

Stochastic Hill Climbing

Simulated Annealing

4. Tìm kiếm beam & di truyền:

Local Beam Search

Genetic Algorithm

Min-Conflicts Search

5. Rà soát có forward checking:

Backtracking Search

Backtracking + Forward Checking

6. Học tăng cường:

Q-Learning

DQN (Deep Q-Network - giả lập)

SARSA

7. Tìm kiếm không quan sát:

PO-S (Partial Observation Search)

NO-S (No Observation Search)

🚀 Cách Chơi

Khởi động: Chạy file main.py

Giao diện hiển thị:

Bàn cờ 8-Puzzle (3x3), trong đó 0 là ô trống

Dòng trên có chữ "8-Puzzle Solver"

Nhiều nút thuật toán chia 3 hàng: cơ bản, heuristic, RL

Chọn thuật toán: Click vào nút tương ứng

Quan sát: Trò chơi sẽ highlight đường đi qua các ô

Reset / Thoát: Nút "RESET" và "X"

🎓 Thông Tin Về Bài Toán

INITIAL_STATE: ((4, 7, 5), (2, 1, 8), (3, 6, 0))

GOAL_STATE: ((1, 2, 3), (4, 5, 6), (7, 8, 0))

Mã trận là tuple 2 chiều bỏ trong heapq / visited

🌐 Cấu Trúc File

project_root/
├── Algorithms.py         # Toàn bộ thuật toán
├── main.py               # Giao diện pygame và logic chính
├── README.md             # File hướng dẫn

🔧 Cài Đặt

pip install pygame
python main.py

✨ Gợi Ý Mở Rộng

Chạy trên giao diện web (Flask/Streamlit)

Lưu lại video/animation đường đi

Cho phép người chơi xếp thử / đặt bài toán

Thêm giải thích của AI (giống explainable AI)
