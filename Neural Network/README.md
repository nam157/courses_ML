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
 
**Tính toán trong mạng neural**
- ![image](https://user-images.githubusercontent.com/72034584/147097722-2c13a579-6a1c-43c7-aed5-eb0d24529fe7.png)
- Hình dạng của các biến: nx1 (số lớp ẩn) = 4, nx2 (input) = 3
  - W1 là trọng số của lớp ẩn thứ 1 có dạng là (nx1,nx2) = (4,3)
  - b1 có dạng là (nx1,1) = (4,1)
  - z1 là kết quả của pt: W1 * X1 + b1 có dạng là (4,3)@(1,4) + (4,1) = (nx1,1) = (4,1)
  - a1 là kết quả của phương trình z1 có dạng là (nx1,1) = (4,1)
  - W2 là trọng số lớp ẩn thứ 2 có dạng là (1,nx1) = (1,4)
  - b2 có dạng (1,1)
  - z2 có dạng là (1,1)
  - a2 có dạng là (1,1)

**Tính toán GD**

- Lan truyền tiến:     ![feed](https://user-images.githubusercontent.com/72034584/147099578-6f048e90-49ea-4255-9a52-2915b4c28ff7.jpg)

   - Z1 = W1*A0 + B1 (A0 == X)
   - A1 = sigmoid(Z1)
   - Z2 = W2*A1 + B2
   - A2 = sigmoid(Z2)
- Lan truyền ngược:    ![back](https://user-images.githubusercontent.com/72034584/147099595-2326bf81-c909-4429-a8d3-1cf3df4bdede.jpg)

   - dZ2 = A2 - Y
   - dW2 =  (dZ2 * A1.T) / m
   - dB2 = sum(dZ2)/m
   - dZ1 = (W2.T @ dZ2) @ g'1(Z1)
   - dW1 = (dZ1 * A0.T) / m
   - dB1 = sum(dZ1)/m

**Khởi tạo trọng số**

- Trong mạng nơ-ron , chúng ta cần khởi tạo trọng số ngẫu nhiên, nếu chúng ta khởi tạo trọng số bằng 0 thì quá trình training nó sẽ không hoạt động bởi vì tất cả đơn vị ẩn giống hệt nhau, chính xác là tính toán các hàm như nhau, và tất cả đơn vị ẩn cập nhật như nhau ở từng lần lặp Gradient descent
- Tuy nhiên khởi tạo trọng số ngẫu nhiên quá lớn hoặc quá nhỏ thì ảnh hướng rất lớn tới vấn đế training và vấn vanishin/ exploding gradient thường thì khởi tạo trọng số theo công thứ He Initialization / Xavier Initialization 
- Khởi tạo được chọn tốt có thể: Tăng tốc độ hội tụ của gradient descent và Tăng tỷ lệ hội tụ gradient descent thành lỗi huấn luyện thấp hơn (và tổng quát hóa)


**Deep Neural Network**
- Mạng shallow neural network là mạng chỉ có 1 hoặc 2 lớp. Mạng DNN là có 3 lớp trở lên.
- Một số ký hiệu hay sử dung: L biểu thị số lớp trong mạng, n[l] số nơ-ron của 1 lớp cụ thể,g[l] là hàm kích hoạt,a[l] = g[l](z[l]) 
- Tính toán trong mạng noron ở DNN nó cũng giống ở shallow nhưng chỉ khác là nhiều số lớp ẩn hơn.
- Kích thước của W là (n[l],n[l-1]), b là (n[l],1), dW có shape tương tự với W, Z[l],A[l],dZ[l],dA[l] (n[l],m)
- Lan truyền xuôi:
   - Z[l] = W[l]A[l-1] +b[l]
   - A[l] = g[l](Z[l])
   - Output A[l],cache(Z[l])
- Lan truyền ngược:
   -  dZ[l]  = dA[l] * g'[l](Z[l])
   -  dW[l] = (dZ[l]A[l-1].T) / m
   -  dB[l] = sum(dZ[l]) / m
   -  dA[l-1] = W[l].T * dZ[l]
   -  Output dA[l-1],dW[l],dB[l]

- Các tham số cần tối ưu:
   - Learning rate
   - Số vòng lặp
   - Số lớp ẩn L
   - Số đơn vị ẩn n (số nơ-ron)
   - Hàm kích hoạt 

### REGULARIZATION
- Có các phương pháp regularization chính: L1-L2,dropout,early stopping, data augmentation
- Thêm yếu tố regualarization sẽ giúp mạng của không bị overfitting
- Chuẩn ma trận L1: ||w|| = sum(|w[i,j]|)
- Chuẩn ma trận L2: ||w||2 = sum(|w[i,j|^2)

### TỐI ƯU MÔ HÌNH
### HYPERPARAMETER TUNING
