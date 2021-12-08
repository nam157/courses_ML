## Kiến thức thông thường
#### Supervised and Unsupervised
- Supervised là dữ liệu đầu vào luôn được gán nhãn, và nhãn đây có thể hiểu là đầu ra mong muốn
- Unsupervised là dữ liệu đầu vào không được gán nhãn
#### Machine Learning and Deep Learning
- Machine Learning là tính năng của AI, cho phép các chuyên gia đào tạo cho AI để nó nhận biết các mẫu dữ liệu và dự đoán
- Deep Learning là kỷ thuật nhỏ của ML, cho phép máy có thể tự đào tạo chính mình, và các phép tính toán học phức tạp hơn
#### Supervised: Regression, Classification, DNN
- Regression: Kết quả đầu ra chính ra là dữ liệu liên tục
- Classification: Kết quả đầu ra chính là dữ liệu rời rạc (nhãn hoặc xác suất nhãn)
- Một thuật toán classification có thể dự đoán giá trị liên tục nhưng giá trị liên tục ở dạng xác suất đối với nhãn
- Một thuật toán Regression có thể dự đoán giá trị rời rạc nhưng giá trị rời rạc với đại lương nguyên
- DNN: Có thể là liên tục hoặc rời rạc (hoặc thuật toán Decision Tree)
- Điều quan trọng nhất chính là phép đánh giá: Classification và Regression
  - Dự đoán Classification có thể đánh giá bằng độ chính xác (accuracy)
  - Dự đoán Regression có thể đánh giá bằng root mean squared error
#### Unsupervised: Clustering,Auto Encoder Decoder, Word embedding
- Clustering: nhiệm vụ là chia dữ liệu vào cũng 1 nhóm, và các điểm trong nhóm đó giống nhau và khác với điểm dữ liệu trong nhóm khác.Về cơ bản là nó là tập hợp các đối tượng trên cơ sở giống nhau và không giống nhau giữa chúng
- Auto Encoder Decoder: Autoencoder là một mô hình mạng nơ-ron có thể được sử dụng để học cách biểu diễn dữ liệu thô được nén. Một bộ Autoencoder có 2 phần đó là encoder và decoder sub-models. Encoder cố gằng nén đầu vào và Decoder cố gắng tái tạo đầu vào [https://machinelearningmastery.com/autoencoder-for-classification/]

#### Train set/ validation set/ test set
- Train set: là tập dữ liệu để chạy thuật toán
- Validation set: Là tập dữ liệu được dùng để hiệu chỉnh các tham số, lựa chọnđặc trưng và quyết định các thay đổi liên quan đến thuật toán học. Đôi khi, nócòn được gọi là tập kiểm định chéo.
- Test set:  Là tập dữ liệu dùng để đánh giá chất lượng của thuật toán học,nhưng không được dùng để quyết định các thay đổi liên quan đến thuật toán họchay các tham số.

#### Cross-validation
- Trong nhiều trường hợp ta thiếu dữ liệu và chia tập training/validation không phù hợp. Nếu chia tập validation quá ít thì mô hình chưa thực sự tối ưu, còn chia tập validation nhiều dữ liệu thì sẽ gây thiếu cho dữ liệu training thì sẽ không đủ xây dựng mô hình. Cross-validation là cải tiến với dữ liệu validation nhỏ nhưng chất lượng mô hình sẽ đánh giá trên nhiều validation khác nhau. Chia tập training thành k tập không giao nhau và kích thước bằng nhau. Mỗi run thì ta sẽ lấy k làm tập validation còn k-1 làm tập training set. Cuối mỗi lần chạy đó ta thu các chi phí và sau đó đánh giá trung bình lỗi trên validation/traning.

#### Regularization
- Early Stopping: Là một kĩ thuật giúp model khi gặp vấn đề overfit, Nó sẽ đánh giá trên hàm mất mát thường thì giá trị hàm mất mát sẽ giảm dần khi tăng số vòng lặp lên. Bây giờ chia ra training  và validation. Trong khi huấn luyện, ta tính toán cả training error và validation error, nếu training error vẫn có xu hướng giảm nhưng validation error có xu hướng tăng lên thì ta dừng thuật toán

                ![Screenshot 2021-12-09 001224](https://user-images.githubusercontent.com/72034584/145252818-4890f677-ca1e-4c40-b983-bfcd7e240701.png)

- Thêm số hạng (L1,L2)

#### Gradient Descent
- Batch Gradient Descent:
  - Dùng tất cả dữ liệu trong training set cho mỗi lần thực hiện bước tính
đạo hàm (n)
- Stochastic Gradient Descent:
  - Chỉ dùng một dữ liệu ngẫu nhiên trong training set cho mỗi lần thực hiện
bước tính đạo hàm (1)
- Mini-batch gradient descent:
  - Dùng một phần dữ liệu trong training set cho mỗi lần thực hiện
bước tính đạo hàm. (1-n)

#### Bias and Variance
- Bias: nghĩa là độ lệch, biểu thị sự chênh lệch giữa giá trị trung bình mà mô hình dự đoán và giá trị
thực tế của dữ liệu
- Variance: nghĩa là phương sai, biểu thị độ phân tán của các giá trị mà mô hình dự đoán so với giá
trị thực tế.
![1_Y-yJiR0FzMgchPA-Fm5c1Q](https://user-images.githubusercontent.com/72034584/143260447-50f2e59e-892c-415f-8267-56485d1b0d22.jpeg)

Giá trị thật dữ liệu (ground truth) ở giữa tâm các đường tròn. Các dấu X là các giá trị dự đoán. Ta
thấy nếu high bias thì giá trị dự đoán rất xa tâm. Tuy nhiên nếu high variance thì các giá trị dự đoán
phân tán rộng dẫn đến việc ra giá trị thực tế. => Ta mong muốn low bias và low variance
## Data preprocessing
- Handling missing data (Missing data) according to contract MCAR, MAR, MNAR
-  Categorical Encoding
-  Handle outlier và scale data 
## Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
## Classification

