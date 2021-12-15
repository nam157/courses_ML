## CÁC KHÁI NIỆM CƠ BẢN VỀ NEURAL NETWORK

   ![image](https://user-images.githubusercontent.com/72034584/146119536-5084a1e4-56b0-4863-9650-877b8cf00fb3.png)
   
**Khái niệm**
- Shallow neural network: có thể hiểu đó là những thuật toán cở bản như là: Logistic regression hay perceptron learing algorithm. 
- Neural network có thể hiểu là các stack xếp chồng của hồi quy logistic
- Neural là 1 tính từ (nơ-ron),network chỉ là 1 cấu trúc, cách mà các nơ-ron đó liên kết với nhau, nên NN có thể tính toán lấy cảm hứng từ sự hoạt động của các nơ-ron hệ thần kinh
- Neural sẽ có các lớp như là Input layer, hidden layer, output layer và node (Số nơ-ron)

**Hàm kích hoạt**

- Hàm kích hoạt được sử dụng sau khi tính tổng linear trong neural network, các hàm kích hoạt này thường là các non-linear
- Hàm sigmoid có giá trị từ [0,1] và là hàm liên tục có đạo ở mọi điểm. Công thức là: a = 1 / (1 + np.exp(-z)) (z là ma trận đầu vào)
- Hàm tanh có giá trị nằm từ [-1,1]. Công thức là : a = np.tanh(z) 
- Hàm tanh được sử dụng ở lớp ẩn hơn vì nó thường đưa giá trị trung tâm, mean về gần 0 thay vì đưa về 0.5, căn giữa dữ liệu tốt cho lớp tiếp theo 
- Nhược điểm của hàm tanh và hàm sigmoid đó là: nếu giá trị đầu vào rất dương hay là rất âm thì đạo hàm của 2 hàm này sẽ gần với 0, điều này cùng nghĩa với những hệ số tương ứng với unit đang xét có thể không cập nhật được gì khi sử dụng gradient descent 
- Hàm kích hoạt phải là phi tuyến tính bởi vì nếu hàm kích hoạt tuyến tính ở 1 layer thì layer này hay layer tiếp theo cũng sẽ tuyến tính vì vậy thì có thể góp lại thành 1 layer. Vì hợp của hàm tuyến tính sẽ ra 1 hàm tuyến tính 
- Một hàm kích hoạt gần đây được sử dụng rộng rãi đó RELU: Hàm có công thức là: a = max(0,z) . Có đạo hàm 1 khi điểm dương và 0 khi điểm âm . Mặc dù có nhược điểm đạo bằng 0 khi điểm âm tuy nhiên có thể khắc phục bằng việc tăng số hidden unit lên.
- Đạo hàm hàm sigmoid: 
  - g(z) = 1 / (1 + np.exp(-z))
  - g(z)' = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
  - g(z)' = g(z) * (1 - g(z))
- Đạo hàm hàm tanh: 
  - g(z) = (e^z - e^-z) / (e^z + e^-z)
  - g(z)' =  1 - np.tanh(z)^2 = 1 - g(z)^2
- Đạo hàm hàm RELU: 
  -   g(z) = np.maximum(0,z)
  -   g(z)' = { 0 if z < 0 
               1 if z > 0}
**Tính toán GD**
- Lan truyền tiến
- Lan truyền ngược
**Khởi tạo trọng số**
- Trong mạng nơ-ron , chúng ta cần khởi tạo trọng số ngẫu nhiên, nếu chúng ta khởi tạo trọng số bằng 0 thì quá trình training nó sẽ không hoạt động bởi vì tất cả đơn vị ẩn giống hệt nhau, chính xác là tính toán các hàm như nhau, và tất cả đơn vị ẩn cập nhật như nhau ở từng lần lặp Gradient descent
- Tuy nhiên khởi tạo trọng số ngẫu nhiên quá lớn hoặc quá nhỏ thì ảnh hướng rất lớn tới vấn đế training và vấn vanishin/ exploding gradient thường thì khởi tạo trọng số theo công thứ He Initialization / Xavier Initialization 
- Khởi tạo được chọn tốt có thể: Tăng tốc độ hội tụ của gradient descent và Tăng tỷ lệ hội tụ gradient descent thành lỗi huấn luyện thấp hơn (và tổng quát hóa)
### REGULARIZATION 
### TỐI ƯU MÔ HÌNH
### HYPERPARAMETER TUNING
