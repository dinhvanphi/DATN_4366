# Hướng Dẫn Thuyết Trình Đồ Án: Tối Ưu Hóa Đèn Giao Thông Bằng PPO

Tài liệu này cung cấp dàn ý chi tiết và các luận điểm quan trọng nhất để bạn trình bày trước Giảng viên hướng dẫn (GVHD) và Hội đồng bảo vệ Đồ án Tốt nghiệp (DATN).

---

## Phần 1: Đặt Vấn Đề (Introduction)
*Hãy bắt đầu bằng việc nêu bật vấn đề thực tế để thu hút sự chú ý.*

*   **Vấn đề:** Các hệ thống đèn giao thông hiện tại hầu hết sử dụng **Fixed Timing (Chu kỳ thời gian cố định)**. Nhược điểm lớn nhất là sự cứng nhắc. Ví dụ, vào giờ cao điểm, trục đường chính đông nghẹt xe nhưng vẫn phải dừng chờ đèn đỏ, trong khi trục đường phụ trong hẻm trống không lại được bật đèn xanh. Điều này gây lãng phí thời gian và tăng kẹt xe.
*   **Giải pháp của Đồ án:** Áp dụng Trí tuệ Nhân tạo (AI), cụ thể là **Reinforcement Learning (Học Tăng Cường)** để tạo ra một "Cảnh sát giao thông AI". AI này có thể quan sát ngã tư theo thời gian thực và linh hoạt quyết định khi nào nên giữ đèn xanh, khi nào nên chuyển đèn đỏ để tối đa hóa lưu lượng thông xe.

---

## Phần 2: Cơ Sở Lý Thuyết (Cần nói thật chắc phần này)
*GVHD chắc chắn sẽ xoáy sâu vào việc bạn có hiểu bản chất của thuật toán không.*

### 2.1. Reinforcement Learning (Học Tăng Cường) là gì?
Giải thích bằng ngôn ngữ dễ hiểu: Giống như việc huấn luyện một chú chó. Khi nó làm đúng, bạn cho ăn (Thưởng/Reward). Khi làm sai, bạn mắng (Phạt/Penalty). Theo thời gian, nó sẽ tự tìm ra chiến thuật tối ưu nhất để nhận được nhiều thức ăn nhất.
**4 Thành phần chính của RL trong đồ án:**
1.  **Environment (Môi trường):** Ngã tư giao thông mô phỏng.
2.  **Agent (Tác tử/Người ra quyết định):** Bộ điều khiển đèn giao thông AI.
3.  **State (Trạng thái):** Góc nhìn của AI (Số lượng xe đang kẹt, thời gian xe đã chờ).
4.  **Action (Hành động):** Giữ đèn xanh hoặc Đổi pha đèn.

### 2.2. Tại sao chọn PPO thay vì DQN? (Cực kỳ quan trọng)
*Nếu bạn từng làm hoặc nhắc đến DQN, hãy dùng điểm này để so sánh:*
*   **DQN (Deep Q-Network):** Là thuật toán tính điểm số (Value-based) cho từng hành động. Yếu điểm của DQN là nó rất "nóng vội", dễ bị sai lệch khi đánh giá phần thưởng (overestimation) và dẫn đến các chính sách điều khiển thiếu ổn định (đèn nháy chớp tắt liên tục).
*   **PPO (Proximal Policy Optimization):** Là thuật toán tối ưu hóa chính sách cận kề, thuộc họ *Actor-Critic*. Đây là thuật toán hiện đại **State-of-the-Art (SOTA)** đang được OpenAI sử dụng cho ChatGPT. 
    *   *Từ khóa ghi điểm "Proximal" (Cận kề):* Nghĩa là thuật toán giới hạn biên độ cập nhật sau mỗi lần học. Tránh việc AI vô tình tìm ra một cách làm "có vẻ tốt" rồi thay đổi toàn bộ kiến thức cũ, khiến mô hình bị sụp đổ (catastrophic forgetting). Nhờ đó, PPO trong đồ án này hội tụ cực kỳ mượt mà và tìm ra chính sách điều khiển ổn định hơn hẳn DQN.

---

## Phần 3: Kiến Trúc Cốt Lõi Của Đồ Án (Project Architecture)
*Đây là phần thể hiện khối lượng công việc và chất xám bạn bỏ ra.*

### 3.1. Trạng thái (State) - AI nhìn thấy gì?
Không chỉ đếm số xe đơn thuần, em thiết kế một State Matrix gồm 17 thông số cực kỳ chi tiết:
*   Độ dài hàng đợi (Queue) ở cả 4 hướng.
*   **Đếm số xe đang chờ (Waiting counts) và Tổng thời gian chờ (Waiting time).**
*   Trạng thái pha đèn hiện tại và tỷ lệ thời gian đã trôi qua.
*   *Điểm sáng:* Tính toán **Áp lực giao thông (Pressure)** giữa trục Bắc-Nam và Đông-Tây để AI có cái nhìn tương quan toàn cục.

### 3.2. Hàm Phần thưởng (Reward Function) - Trái tim của Đồ án
*Đây là phần bạn đã phải sửa rất nhiều lần để AI khôn lên, hãy nhấn mạnh nó:*
*   **Phạt BÌNH PHƯƠNG số lượng xe chờ:** Thay vì phạt tuyến tính, em dùng phạt bình phương (`-(số_xe^2) * 0.01`). Giảng viên sẽ hỏi tại sao? Câu trả lời là: *Một hàng đợi dài 20 xe sẽ tồi tệ hơn rất nhiều so với hai hàng đợi mỗi hàng 10 xe. Hàm bình phương sẽ ÉP AI cảm thấy "vô cùng đau đớn" khi có một hướng bị kẹt cứng, buộc nó phải lập tức chuyển pha đèn để giải cứu hướng đó.*
*   **Phạt chuyển đèn (Switch Penalty):** Đổi đèn sẽ mất 14 frames đèn Vàng/Đỏ (không xe nào đi được). Do đó có một hình phạt nhỏ để ngăn AI đổi đèn liên tục (tránh hiện tượng flickering).

---

## Phần 4: Kịch Bản Thử Nghiệm và Kết Quả Đánh Giá
*Nói về tính thực tiễn của dự án.*

### Kịch bản 1: Lưu lượng đều đặn (Uniform Traffic)
*   **Kết quả:** PPO chỉ nhỉnh hơn Fixed Timing khoảng 2-4%.
*   **Giải thích:** Trong giao thông học, nếu xe ra đều và đối xứng hoàn hảo, chu kỳ cố định vốn đã gần chạm mốc tối ưu (Optimal). PPO chỉ có thể tối ưu thêm bằng cách vi chỉnh kéo dài đèn 1-2 giây cho chiếc xe cuối cùng thoát qua. Việc AI tự tìm ra được tính chất này đã là một sự thành công lớn.

### Kịch bản 2: Giờ Cao Điểm Bất Đối Xứng (Asymmetric Rush Hour) - ĐIỂM SÁNG!
*   **Cấu hình:** Trục đường chính Bắc-Nam có lưu lượng xe đông gấp **4.5 lần** trục đường phụ (hẻm) Đông-Tây.
*   **Kết quả:** 
    *   Fixed Timing thất bại hoàn toàn vì vẫn chia đều 140 frames cho hẻm vắng xe, khiến trục chính kẹt cứng hàng nghìn xe.
    *   **PPO:** AI tự động cắt ngắn đèn xanh của đường phụ chỉ còn ~45 frames và dành toàn bộ hơn 200 frames cho đường chính Bắc-Nam. 
    *   **Hiệu quả:** Lưu lượng thông xe (Throughput) tăng vọt, thời gian kẹt xe trung bình giảm mạnh (10-20%). Minh chứng rõ nét AI hoàn toàn **chủ động ra quyết định (Agent > 0%)** chứ không dựa dẫm vào luật ép buộc.

---

## Phần 5: Q&A - Dự phòng câu hỏi khó từ Giảng Viên

**Câu 1: "Tại sao không dùng các thuật toán cảm biến cứng (ví dụ: thấy 5 xe thì bật xanh) mà phải dùng Machine Learning/RL rườm rà?"**
> **Trả lời:** Các hệ thống cảm biến cứng (Rule-based) chỉ hoạt động tốt ở các ngã tư đơn lẻ và tầm nhìn thiển cận. Reinforcement Learning vượt trội ở chỗ nó tối ưu hóa phần thưởng **dài hạn (Long-term reward)**. Hơn nữa, RL có khả năng mở rộng (Scale) để điều khiển đồng bộ nhiều ngã tư cùng lúc trong tương lai (Multi-agent RL) bằng cách phối hợp "Làn sóng xanh" (Green Wave), điều mà rule-based không thể tự cấu hình được.

**Câu 2: "Làm sao em chứng minh được là AI tự quyết định đổi đèn chứ không phải do code của em cài đặt sẵn (If-else)?"**
> **Trả lời:** Em có thể chứng minh qua log mô phỏng. Khi chạy mô phỏng, giao diện hiển thị rõ ràng thông số `Agent %`. Hơn nữa, em đã nới lỏng mức giới hạn ép đổi đèn (`MAX_GREEN_TIME`) của hệ thống lên rất cao (300 frames). Do đó, những lần đổi đèn ở frame 80, 100 hay 150 hoàn toàn là do Mạng Nơ-ron (Neural Network) tính toán xác suất và ép Action = 1, không hề có bất kỳ câu lệnh if-else nào can thiệp trước ngưỡng 300 frames.

**Câu 3: "Dữ liệu huấn luyện (Training Data) em lấy từ đâu?"**
> **Trả lời:** Học tăng cường (RL) khác với Học có giám sát (Supervised Learning). Nó **không cần bộ dữ liệu (Dataset) thu thập sẵn**. RL tự động "học qua trải nghiệm" (Trial and Error) bằng cách tương tác trực tiếp với bộ mô phỏng (Simulation). Ở đồ án này, em đã cho AI chạy ngầm không giao diện (Headless Training) tới **500.000 đến 1.000.000 frames** để AI tự khám phá ra quy luật giao thông tối ưu nhất.

1. Nhóm File Cốt Lõi (Mô phỏng Vật lý & Giao thông)
Đây là phần tạo ra thế giới cho AI tương tác.

traffic_constants.py:
Nhiệm vụ: Chứa toàn bộ các hằng số cấu hình của hệ thống.
Chi tiết: Định nghĩa kích thước đường, kích thước xe, màu sắc hiển thị, và đặc biệt là Tỷ lệ sinh xe (SPAWN_PROBS). Việc cấu hình kịch bản Giờ Cao Điểm (Rush Hour) bất đối xứng được chỉnh sửa trực tiếp tại đây.
traffic_entities.py:
Nhiệm vụ: Định nghĩa các đối tượng vật lý (Entities).
Chi tiết: Chứa class Car (Xe cộ) quản lý vị trí, vận tốc, gia tốc, phanh xe, và logic tự động dừng lại khi gặp đèn đỏ hoặc gặp xe phía trước (car-following model). Chứa class TrafficLight để quản lý đèn vật lý.
traffic_simulation.py:
Nhiệm vụ: Trái tim của bộ mô phỏng (Simulation Engine).
Chi tiết: File lớn nhất dự án. Chứa vòng lặp thời gian thực (game loop), vẽ đồ họa bằng Matplotlib, liên tục cập nhật vị trí các xe, sinh xe mới, xóa xe khi ra khỏi ngã tư. Nó cũng chứa class SimulationBenchmark để chạy so sánh kết quả giữa các thuật toán.
2. Nhóm File Trí Tuệ Nhân Tạo (Reinforcement Learning)
Đây là "Bộ não" của dự án.

traffic_ppo.py:
Nhiệm vụ: Chứa thuật toán lõi Proximal Policy Optimization (PPO).
Chi tiết: Chứa cấu trúc Mạng Nơ-ron Nhân tạo (Neural Network) gồm 2 phần Actor (ra quyết định) và Critic (đánh giá điểm số). Xử lý toán học phức tạp như hàm mất mát (loss function), tính toán lợi thế (GAE - Generalized Advantage Estimation), và cơ chế "Clip" chống mất ổn định đặc trưng của PPO.
traffic_dqn.py:
Nhiệm vụ: Chứa thuật toán Deep Q-Network (DQN).
Chi tiết: File này đóng vai trò làm thuật toán dự phòng hoặc để so sánh (Baseline). DQN là thuật toán cũ hơn dựa trên bảng Q-Table được Deep Learning hóa, thường kém ổn định hơn PPO trong bài toán liên tục như đèn giao thông.
traffic_rl.py:
Nhiệm vụ: Cầu nối (Wrapper) giữa Môi trường (Simulation) và Bộ não (PPO).
Chi tiết: Đây là nơi định nghĩa bài toán RL. Nó chứa 3 hàm quan trọng bậc nhất:
get_state(): Tổng hợp dữ liệu (số xe, thời gian chờ, áp lực giao thông) thành ma trận 17 biến cho AI "nhìn".
calculate_reward(): Hàm tính toán điểm thưởng/phạt dựa trên bình phương số xe chờ.
step(): Giao tiếp hành động chuyển đèn từ AI vào bộ mô phỏng.
3. Nhóm File Chạy Đầu Cuối (Entry Points & Scripts)
Đây là các file dùng để khởi chạy chương trình.

traffic_sim.py:
Nhiệm vụ: Menu điều khiển chính (CLI).
Chi tiết: Cung cấp cho người dùng các lựa chọn từ 1 đến 8 (Chạy Fixed Timing, Train có UI, Test, Compare nhiều Seed...).
train_headless.py:
Nhiệm vụ: Kịch bản huấn luyện tốc độ cao (Headless Training).
Chi tiết: Khởi chạy mô phỏng nhưng TẮT toàn bộ giao diện đồ họa (GUI). Điều này cho phép CPU dồn toàn bộ sức mạnh để chạy hàng triệu khung hình (frames) chỉ trong thời gian ngắn, liên tục theo dõi và lưu lại mô hình tốt nhất (ppo_model.pth).
plot_benchmark.py:
Nhiệm vụ: Vẽ biểu đồ báo cáo.
Chi tiết: Chạy sau khi test xong để tạo ra file ảnh PNG chứa các biểu đồ trực quan (Lưu lượng, Thời gian chờ, Số lần đổi pha) so sánh giữa PPO và Fixed Timing, dùng để chèn thẳng vào báo cáo Word hoặc Slide.


Thuật toán PPO (Proximal Policy Optimization) được xây dựng dựa trên lý thuyết của các phương pháp Policy Gradient (Gradient chính sách) trong Học tăng cường (Reinforcement Learning).Cụ thể, nó dựa trên hai nền tảng chính:Trust Region Methods (Phương pháp Vùng tin cậy): Kế thừa ý tưởng từ thuật toán TRPO, PPO cố gắng đảm bảo mỗi bước cập nhật chính sách không quá lớn để tránh làm sụp đổ quá trình huấn luyện.Importance Sampling (Lấy mẫu quan trọng): Sử dụng tỉ lệ giữa chính sách mới và chính sách cũ để đánh giá mức độ thay đổi, giúp tái sử dụng dữ liệu hiệu quả hơn.Điểm cốt lõi của PPO:Thay vì giải các bài toán tối ưu phức tạp như TRPO, PPO sử dụng một hàm mục tiêu gọi là Clipped Surrogate Objective. Hàm này sẽ "cắt" (clip) các thay đổi quá mức của chính sách, giữ cho quá trình cập nhật luôn nằm trong một "vùng an toàn", giúp thuật toán vừa ổn định vừa dễ cài đặt hơn.