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
- Một số ký hiệu hay sử dung: L biểu thị số lớp trong mạng, n[l] số nơ-ron của 1 lớp cụ thể,g[l] là hàm kích hoạt,a[l] = g[l] (z[l]) 
- Tính toán trong mạng noron ở DNN nó cũng giống ở shallow nhưng chỉ khác là nhiều số lớp ẩn hơn.
- Kích thước của W là (n[l],n[l-1]), b là (n[l],1), dW có shape tương tự với W, Z[l],A[l],dZ[l],dA[l] (n[l],m)
- Lan truyền xuôi:
   - Z[l] = W[l]A[l-1] +b[l]
   - A[l] = g[l] (Z[l])
   - Output A[l],cache(Z[l])
- Lan truyền ngược:
   -  dZ[l]  = dA[l] * g'[l] (Z[l])
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

**L1-L2 norm**
- Thêm yếu tố regualarization sẽ giúp mạng của không bị overfitting
- Chuẩn ma trận L1: ||w|| = sum(|w[i,j]|)
- Chuẩn ma trận L2: ||w||2 = sum(|w[i,j|^2)
- Phiên bản chuẩn hóa L1 thì khiến các trọng số giảm về 0 nên kích thước mô hình sẽ nhỏ hơn ban đầu
- Phiên chuẩn hóa L2 thì được sử dụng nhiều hơn

Tiêu chuẩn trong mạng nơ-ron:
- Hàm chi phí chuẩn mà chúng ta cần tối thiểu hóa: J(W1,b1...,WL,bL) = (1/m) * Sum(L(y(i),y'(i))).
- Phiên bản điều chuẩn L2: J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum((||W[l]||^2).
- Chúng ta xếp chồng ma trận thành một vectơ (mn,1) rồi áp dụng sqrt(w1^2 + w2^2.....).
- Thực hiện lan truyền ngược (cách cũ): dw[l] = (from back propagation).
- Cách mới: dw[l] = (from back propagation) + lambda/m * w[l].

**Câu hỏi: Tại số tiêu chuẩn có thể giúp model đỡ bị overfitting**
- Như chúng ta đã biết thì tham số tiêu chuẩn (lambda) càng lớn thì trọng số w có xung hướng gần bằng 0, điều này giúp cho mạng nơ-ron chúng ta trở nên đơn giản hơn
- Nếu như tham số tiêu chuẩn vừa đủ thì khiến trọng số w không quá lớn
- Nếu như trong mạng nơ-ron chúng ta sử dụng hàm kích hoạt tanh, thì lambda quá lớn -> w sẽ nhỏ -> chúng ta sẽ dùng các phần tuyến tính của hàm kích hoạt tanh nên hãy đi từ kích hoạt phi tuyến tính gần tới tuyến tính, làm cho mạng nơ-ron thành phân loại như phân tách tuyến tính


**Dropout**
- Chính là loại ngẫu nhiên các đơn vị ra khỏi mạng nơ-ron, sẽ như vậy ở mỗi vòng lặp, chúng ta sẽ làm việc với mạng nơ-ron nhỏ hơn, vậy nên sử dụng mạng nơ-ron nhỏ hơn hơn giống như tiêu chuẩn.
- Không thể dựa vào 1 đặc trưng nào cả mà phải trải rộng các trọng số
- Không thể chỉ ra dropout có tác động tương tự L2-norm
- Dropout còn có thể keep_prob khác ở mỗi lớp
- Dropout lớp đầu vào phải gần 1 vì chúng ta ko muốn loại nhiều đặc trưng
- Nếu như chúng ta sợ 1 số lớp quá khớp các lớp khác thì chúng ta có keep_prob thấp hơn vài lớp, nếu làm như lày thì khó kiểm định chéo.

**Data augmentation**
- Chúng ta sẽ biến đổi dữ liệu bằng cách lật , xoay các thức để thêm dữ liệu vào
- Dữ liệu mới thu được thì ko được tốt như dữ liệu độc lập thực tế, cần dùng kĩ thuật tiêu chuẩn

**Early Stopping**

![image](https://user-images.githubusercontent.com/72034584/147108490-83d9b1ba-e1f7-4667-979a-36442e81680c.png)


**Vanishing/exploding gradients**
- Chuẩn hóa đầu vào: Chúng ta sẽ chuẩn hóa đầu vào sẽ giúp model hoạt động tốt hơn và thời gian trainnig cũng nhanh hơn
- Nếu như không chuẩn hóa thì hàm chi phí sẽ sâu và hình dạng không đồng nhất, tốn thời gian tối ưu
- Vanishing gradient:
   - Khi chúng ta lan truyền ngược từ lớp đầu ra đến lớp đầu vào, các gradient thường sẽ nhỏ dần và nhỏ hơn và thấp chí tiến về 0. Điều này chúng ta sẽ thấy rằng các trọng số w ở các lớp đầu vào sẽ không thay đổi hoặc gần như không thay đổi. Hậu quả sẽ khiến mô hình không bao giờ hội tụ
- Exploding gradient:
   - Nó ngược lại vấn đề vanishing, khi lan truyền ngược thì gradinet có xu hướng lớn dần, đến lớp đầu vào thì chúng ta sẽ thu được trọng số w vô cùng lớn làm cho độ dốc nó phân kỳ. 
=> Để giải quyết vấn đề này thì chúng ta có thể sử dụng các kỹ thuật như: Batch-normalization,Khởi tạo trọng số ban đầu thích hợp,Chọn activation funtions không bị bão hòa

- Khởi tạo trọng số chúng ta có thể sử dụng công thức KT: He Initialization
### TỐI ƯU MÔ HÌNH
- Gradient Descent:
   - Batch Gradient Descent: 
      - Chúng ta tính toán và chạy gradient descent trên toàn bộ tập dữ liệu
      - Và qua mỗi vòng lặp thì hàm chí phí luôn luôn có xu hướng giảm
      - Đối với tập liệu khổng lồ thì việc đào tạo toàn 1 dữ liệu thì khiến thời gian training rất là lâu, khó tối ưu hóa
      - ![image](https://user-images.githubusercontent.com/72034584/147120026-83c5fc36-36d4-49fc-aff6-bceee7d17a4d.png)

   - Stochastics Gradient Descent:
     - Nó lấy ngẫu nhiên 1 điểm dữ liệu và chạy GD
     - Và qua mỗi vòng lặp + cộng với yếu tố ngẫu nhiên thì nó sẽ nhảy lung tung ở mọi điểm
     - Tuy nhiên về cơ bản nó luôn có xu hướng về điểm cực tiêu, nhưng nó sẽ không bao giờ hội tụ và chỉ dao động lên xuống ở điểm cực tiểu
     - Và chậm vector hóa vì nó chọn 1 điểm dữ liệu
     - ![image](https://user-images.githubusercontent.com/72034584/147120621-146ddfd3-8fe9-4c64-95c3-c69dda3c2183.png)
 
   - Mini-batch gradient descent:
     - Nó sẽ lấy mẫu trong từ 1 -> m, nhưng mẫu sẽ không quá nhỏ hoặc không quá lớn
     - Nó sẽ tính toán và học nó nhanh hơn
     - Tiếp tục tiến độ hội tụ mà không xử lý toàn bộ dữ liệu
     - Nó luôn có xung hướng về điểm cực tiêu tuy nhiên nó rất khó hội tụ, vì nó lấy từng mẫu để tính toán vì hàm chi phí luôn dao động lên xuống
     - Nếu có về gần điểm cực tiêu thì nó vẫn khó hội tụ và tiếp dao động
     - ![image](https://user-images.githubusercontent.com/72034584/147126064-4723c5d8-28d3-4664-a7a8-503688585273.png)
- Thuật toán cải tiển hoặc sự kết hợp GD:
   - Thuật toán GD theo momentum:
      - Thông thường GD là mini-bactch hoặc stochastics kết hợp với momentum
      - Y tưởng đơn giản đó là tính toán trung bình cộng theo số nhân rồi mới cập nhật trọng với các giá trị mới
      - Và có thêm 1 tham số mới đó beta, beta thường nằm ở 0.9 đến 0.98
        - Beta = 0.9 thì thường nó sẽ tính entry cuối,0.98 thì 50, 0.5 thì 2 entry 
      - Khi beta này cao thì nó sẽ làm phằng trung bình các điểm dữ liệu bị lệch. Do vậy điều này giảm dao động trong GD, khiến đường dẫn tới cực tiểu nhanh hơn.
      - Công thức tổng quát: vdw = beta * vdw + (1- beta)*dw, w = w - learning_rate * vdw  
   - Thuật toán RMSprop (root mean square prop): Đây cũng là 1 thuật tăng tốc độ GD
     - Về cơ bản thì RMSprop hoạt động khá tương tự Momentum
     - RMSprop có khiến hàm chi phí di chuyện chậm theo hướng dọc và nhanh hơn theo phương ngang
     - Công thức: sdw = beta * sdw + (1-beta)dw**2, w = w - learning_rate*(dw/sqrt(sdw)) thường chúng ta sẽ thêm tham số esp để không phải chia cho 0
   - Thuật toán Adam:
     - Thuật toán này là ra là sự kết hợp của 2 thuật toán ở trên momentum và RMSprop
     - Công thức:
       - vdw = (beta1 * vdw) + (1-beta1) * dw, vdb = (beta1 * vdb) + (1 - beta1) * db
       - sdw = (beta2 * sdw) + (1 - beta2) * dw**2 , sdb = (beta2 * sdb) + (1 - beta2) * db**2
       - vdw = vdw / (1-beta1^t),vdb = vdb / (1-beta1^t)
       - sdw = sdw / (1-beta2^t),sdb = sdb / (1-beta2^t)
       - w = w - learing_rate * vdw / (sqrt(sdw) + epsilon)
     - Beta1 = 0.9, beta2 = 0.99
- Batch-normalization:
   - Chúng ta thường chuẩn hóa đầu vào của bài toán, giúp cho tốc độ học nhanh hơn, hàm chi phí và đạt cực tiêu nhanh hơn 
   - Và bây giờ chúng cũng có thể chuẩn hóa đầu vào của các lớp ẩn a[l].
   - Mặc dụng chuẩn hóa batch ở các lớp ẩn có thể giống ở chuẩn hóa đầu vào đưa các giá trị mean = 0 và std , tùy nhiên trong các lớp ẩn chúng ta không mong muốn dữ liệu đều đưa về như vậy, mean std có thể khác đi để tận dụng phi tuyến tính trong activation function. Để kiểm soát vấn đề mean,std chúng ta có thể điều chỉnh 2 tham số gamma và beta
   - Công thức: ![image](https://user-images.githubusercontent.com/72034584/147117689-ccc00313-dcbc-4d40-bc4c-579994d01ef4.png)
   - VD: ![image](https://user-images.githubusercontent.com/72034584/147117435-1b51e88e-7a3d-4ff5-a931-c559090e9f21.png)
   - ![aa](https://user-images.githubusercontent.com/72034584/147115077-071548cd-e93a-4dc1-a94f-63fa0e841545.jpg)
   - Chuẩn hóa batch còn giảm vấn đề thay đổi (dịch chuyển) giá trị đầu vào
   - Mỗi mini-batch bị co giãn theo trung bình/phương sai đã tính của nó, mean = 0, std = 1
   - Điều này thêm nhiễu vào giá trị z[l] trong minibatch. Nó giống như dropout, điều này thêm nhiễu vào các kích hoạt của từng lớp ẩn
 
### HYPERPARAMETER TUNING
