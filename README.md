## Basic knowledge
#### Probability & Statistics in Machine Learning
[Probability & Statistical](https://towardsdatascience.com/probability-vs-statistics-for-data-science-and-machine-learning-84f00bf67ce1)
- Trong xác suất, chúng ta sẽ bắt đầu với model mô tả khả năng của sự kiện sẽ xảy ra. Sau đó dựa đoán khả năng xảy ra của sự kiện. Tóm lại có thể hiểu rằng là xây dựng 1 cái model dự đoán khả năng xảy ra trong tương lại dựa trên mô hình không có dữ liệu thực tế
- Trong thông kê thì tương phản với xác suất, thông kê chúng ta sẽ suy luận từ data hoặc mô hình dự trên dữ liệu thực tế để quan sát 
- Xác suất là đi từ model sang data trong khi Thông kê là đi từ data sang model 

     ![image](https://user-images.githubusercontent.com/72034584/145321985-4e237c87-27fa-4053-96a1-5d7a079c046c.png)

#### Supervised and Unsupervised
- **Supervised** là thuật toán dựa đoán đầu ra của 1 hoặc nhiều mới dựa vào cặp (đầu vào, đầu ra) đã biết trước. Một tập biến đầu vào X= {X1,X2,...Xn} và tập đầu ra tương ứng Y = {Y1,Y2,...,Yn}
- **Unsupervised** là ngược lại với supervised chúng ta không biết kết quả đầu ra mà chỉ biết các vector đặc trưng đầu vào 

#### Machine Learning and Deep Learning
- **Machine Learning** là tính năng của AI, cho phép các chuyên gia đào tạo cho AI để nó nhận biết các mẫu dữ liệu và dự đoán
- **Deep Learning** là kỷ thuật nhỏ của ML, cho phép máy có thể tự đào tạo chính mình, và các phép tính toán học phức tạp hơn

#### Supervised: Regression, Classification, DNN
- **Regression**: Kết quả đầu ra chính ra là dữ liệu liên tục
- **Classification**: Kết quả đầu ra chính là dữ liệu rời rạc (nhãn hoặc xác suất nhãn)
- Một thuật toán classification có thể dự đoán giá trị liên tục nhưng giá trị liên tục ở dạng xác suất đối với nhãn
- Một thuật toán Regression có thể dự đoán giá trị rời rạc nhưng giá trị rời rạc với đại lương nguyên
- DNN: Có thể là liên tục hoặc rời rạc (hoặc thuật toán Decision Tree)
- Điều quan trọng nhất chính là phép đánh giá: Classification và Regression
  - Dự đoán Classification có thể đánh giá bằng độ chính xác (accuracy)
  - Dự đoán Regression có thể đánh giá bằng root mean squared error
#### Unsupervised: Clustering,Auto Encoder Decoder, Word embedding
- **Clustering**: nhiệm vụ là chia dữ liệu vào cũng 1 nhóm, và các điểm trong nhóm đó giống nhau và khác với điểm dữ liệu trong nhóm khác.Về cơ bản là nó là tập hợp các đối tượng trên cơ sở giống nhau và không giống nhau giữa chúng
- **Auto Encoder Decoder**: Autoencoder là một mô hình mạng nơ-ron có thể được sử dụng để học cách biểu diễn dữ liệu thô được nén. Một bộ Autoencoder có 2 phần đó là encoder và decoder sub-models. Encoder cố gằng nén đầu vào và Decoder cố gắng tái tạo đầu vào.[Autoencoder](https://machinelearningmastery.com/autoencoder-for-classification/)

#### Train set/ validation set/ test set
- **Train set**: là tập dữ liệu để chạy thuật toán
- **Validation set**: Là tập dữ liệu được dùng để hiệu chỉnh các tham số, lựa chọnđặc trưng và quyết định các thay đổi liên quan đến thuật toán học. Đôi khi, nócòn được gọi là tập kiểm định chéo.
- **Test set**:  Là tập dữ liệu dùng để đánh giá chất lượng của thuật toán học,nhưng không được dùng để quyết định các thay đổi liên quan đến thuật toán họchay các tham số.

#### Cross-validation
- Trong nhiều trường hợp ta thiếu dữ liệu và chia tập training/validation không phù hợp. Nếu chia tập validation quá ít thì mô hình chưa thực sự tối ưu, còn chia tập validation nhiều dữ liệu thì sẽ gây thiếu cho dữ liệu training thì sẽ không đủ xây dựng mô hình. Cross-validation là cải tiến với dữ liệu validation nhỏ nhưng chất lượng mô hình sẽ đánh giá trên nhiều validation khác nhau. Chia tập training thành k tập không giao nhau và kích thước bằng nhau. Mỗi run thì ta sẽ lấy k làm tập validation còn k-1 làm tập training set. Cuối mỗi lần chạy đó ta thu các chi phí và sau đó đánh giá trung bình lỗi trên validation/traning.

#### Regularization
- Early Stopping: Là một kĩ thuật giúp model khi gặp vấn đề overfit, Nó sẽ đánh giá trên hàm mất mát thường thì giá trị hàm mất mát sẽ giảm dần khi tăng số vòng lặp lên. Bây giờ chia ra training  và validation. Trong khi huấn luyện, ta tính toán cả training error và validation error, nếu training error vẫn có xu hướng giảm nhưng validation error có xu hướng tăng lên thì ta dừng thuật toán

     ![Screenshot 2021-12-09 001224](https://user-images.githubusercontent.com/72034584/145318350-bebddcca-e9de-47d8-ac8e-48ad67311299.png)

- Thêm số hạng (L1,L2): Là một kỹ thuật phổ biến hiện nay, ta sẽ 1 thêm số hạng vào hàm mất mát, số hạng này sẽ đánh giá độ phức tạp của mô hình, số hạng càng lớn thì thể hiện rằng mô hình phức tạp. Có 2 hàm regularization phổ biển đó là l1-norm and l2-norm. Khi sử dụng l1-norm là ||w1||1 thì nghiệm w có xu hướng giảm về 0. VD: Khi ta thêm l1-norm vào Linear Regression chúng ta thu được Lasso Regression. w = 0 là những đặc trưng không quan trọng và những w != 0 là những đặc trưng ảnh hưởng tới mô hình. Vì Lasso Regression thường dùng để nén mô hình hoặc extract features.  Khi sử dụng l2-norm thêm vào hàm mất mát thì ta sẽ thu được Rigde Regression. Hàm regularization sẽ giúp các hệ số w không quá lớn, giúp tránh việc phụ thuộc quá nhiều 1 đặc trưng nào đó.
Công thức chung : Lreg(θ) = L(θ) + λR(θ)

#### Các phép đánh giá mô hình cơ bản

![image](https://user-images.githubusercontent.com/72034584/145331108-f2ab80f4-b775-4e2c-998c-a46de6506e3f.png)

[Nguồn](https://www.miai.vn/2020/06/16/oanh-gia-model-ai-theo-cach-mi-an-lien-chuong-2-precision-recall-va-f-score/)

- Accuaracy: Nó đo lượng có bao nhiêu quan sát, cả tích cực và tiêu cực (positive and negative) đã được phân loại chính xác.
- ROC-AUC: Đường cong AUC - ROC là một phép đo hiệu suất cho các vấn đề phân loại ở các cài đặt ngưỡng khác nhau. ROC là một đường cong xác suất và AUC đại diện cho mức độ hoặc thước đo khả năng phân tách. Nó cho biết mô hình có khả năng phân biệt giữa các lớp như thế nào. AUC càng cao, mô hình càng tốt trong việc dự đoán 0 lớp là 0 và 1 lớp là 1. 

     ![image](https://user-images.githubusercontent.com/72034584/145335357-50f9be18-7152-4933-8e05-b52c25b7e9b6.png)
     
     - ROC:
          - Để vẽ được đường cong thì ta cần quan tâm tới 2 khái niệm nữa là:
               - True Positive Rate (TPR): Chính là Recall 
               - False Positive Rate (FPR): Tỷ lệ cảnh báo sai
           - => Thực chất là đường cong ROC biểu thị mối quan hệ giữa TPR,FPR khi chúng ta thay đổi ngưỡng model 
     - AUC: Chính là diện tích bên dưới đường cong, diện tích lớn là càng tốt

[Tham Khao ROC-AUC](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5?fbclid=IwAR3w8Mv4ojUnIYYiy3mXWTJ23wH9FP52S8u_jaeQRviUK6XJINeQf1Q4wUI)

![image](https://user-images.githubusercontent.com/72034584/145330819-5511c6d8-b5d6-483f-9200-a2a858037cf6.png)

[Nguồn](https://www.digital-mr.com/media/cache/5e/b4/5eb4dbc50024c306e5f707736fd79c1e.png)

   ![image](https://user-images.githubusercontent.com/72034584/145334234-0cc732ba-492f-4a8d-9ea7-277ff05035ca.png)

- Precision: Được tính là True positive (TP) chia cho tổng số sample được phân loại positive (TP + FP) và precision nó nằm trong khoảng 0 đến 1, Precision càng lớn có nghĩa là độ chính xác của các điểm tìm được càng cao. Precision sẽ cần được coi trọng hơn khi lựa chọn model với các bài toán cụ thể khi mà việc nhận nhầm False Positive mang lại kêt quả tồi tệ. Ví dụ với bài toán chặn Spam Mail chẳng hạn, khi đó việc nhận nhầm FP (nhầm 1 mail thường thành mail spam) sẽ làm ảnh hưởng đến công việc của người dùng vì miss một cái mail quan trọng (hợp đồng hàng nghìn tỷ đồng chẳng hạn).=> Thể hiện sự chuẩn xác của việc phát hiện các điểm Positive. Số này càng cao thì model nhận các điểm Positive càng chuẩn.

     ![image](https://user-images.githubusercontent.com/72034584/145334281-84b38f7c-73ea-4d7d-bb67-babfb4a4c189.png)

- Recall: Được tính là tỷ lệ điểm giữa các điểm thực đúng trên tổng positive thực. Khi Recall càng cao thì tỷ lệ bỏ sót càng ít. Recall nên được gán trọng số cao hơn khi cân nhắc lựa chọn model tốt nhất khi mà việc nhận nhầm các nhãn Positive thực thành False Negative mang lại hậu quả khôn lường. Ví dụ như bài toán ung thư bên trên kìa, việc nhận nhầm người Ung thư thành người bình thường và trả về nhà không điều trị sớm thì toi. => Thể hiện khả năng phát hiện tất cả các postivie, tỷ lệ này càng cao thì cho thấy khả năng bỏ sót các điểm Positive là thấp

    ![image](https://user-images.githubusercontent.com/72034584/145334325-c8a4e920-0459-478d-9ee4-7a3d1df493c6.png)

- F1-Score: Khi xây dựng model thì ta luôn Precision và Recall càng cao càng tốt tuy nhiên trong thực tế thì xây dựng model thì hay vấn để Precision cao thì Recall thấp và ngược lại vì vậy đề lựa chọn tốt thì ta quan tâm F1, F1 sẽ dung hòa 2 cái này lại và lựa chọn model căn cứ và F1 để chọn. F1 càng cao thì càng tốt. Khi lý tưởng nhất thì F1 = 1 (khi Recall = Precision=1).=> Là số dung hòa Recall và Precision giúp ta có căn cứ để lựa chọn model. F1 càng cao càng tốt


#### Overfitting and Underfitting
![image](https://user-images.githubusercontent.com/72034584/145583551-13af0e08-1f58-4351-aa26-b63dd5f396c8.png)

- Overfitting
  - Là trường hợp quá fit với tập dữ liệu trainning có nghĩa là những điểm nhiễu trong tập dữ liệu training cũng học, trường hợp overfit xảy ra khi tập dữ liệu training quá nhỏ hoặc model quá cao.
  - Chi phí lỗi training error thấp mà validation error/test error quá cao thì điều có nghĩa tập dữ liệu overfit
  - Để khắc phục vấn đề này thì có thể sử dụng: regurlarization, Validation,...

- Underfitting
  - Là trường hợp mô hình chưa khái quá hóa được dữ liệu traning cũng như chưa khái quan hóa tập dữ liệu mới
  - Một mô hình học máy không phù hợp không phải là một mô hình phù hợp và sẽ hiển nhiên vì nó sẽ có hiệu suất kém trên dữ liệu đào tạo.
  - Hoặc mô hình quá nhỏ
  - Chi phí lỗi traning error và validation error/test error đều cao thì có nghĩa là model đang bị underfit


#### Gradient Descent
- Là 1 thuật toán tối ưu tổng quát tốt nhất, có khả năng tìm nghiệm tối ưu cho rất nhiều bài toán. Ý tưởng chung, là liên tục điều chỉnh tham số để cực tiêu hóa chí phí. VD: Ta đứng trên 1 đỉnh núi và muốn xuống núi nhanh thì chúng ta phải đi xuống núi theo hướng dốc nhất.

![Screenshot 2021-12-10 210101](https://user-images.githubusercontent.com/72034584/145585739-50e2e882-3a31-4db8-8f34-2673d77fd30b.jpg)

- Từ VD đó ta thì bài toán GD cũng thực hiện như vậy, nó gradient cục bộ của hàm chi phí theo vector tham số, rồi đi ngược hướng với gradient đó, khi GD bằng 0 tức đó là điểm cực tiểu
- Trong GD có tham số rất qua trọng đó learning rate hay còn gọi tốc độ học, cần phải điều chỉnh phù hợp
- Và cần xem xét hàm lỗi hay không lỗi đều điều chỉnh số vòng lặp và learning rate cho phù hợp. Hàm lồi là hàm đoạn thắng nối 2 điểm bất kỳ trên 1 đường cong, không bao giờ cắt đường cong đó, điểm local cũng chính là điểm global 

- Batch Gradient Descent:
  - Dùng tất cả dữ liệu trong training set cho mỗi lần thực hiện bước tính
đạo hàm (n)
- Stochastic Gradient Descent:
  - Chỉ dùng một dữ liệu ngẫu nhiên trong training set cho mỗi lần thực hiện
bước tính đạo hàm (1)
- Mini-batch gradient descent:
  - Dùng một phần dữ liệu trong training set cho mỗi lần thực hiện
bước tính đạo hàm. (Khoảng từ 1 - n)

#### Loss function
- Regression Losses

  ![image](https://user-images.githubusercontent.com/72034584/145739806-ff8c7271-8d3f-43f8-aaa2-0d6d7241d526.png)
  
  - Mean square error: Là một phép đo trung bình phương giữa giá trị dự đoán và giá trị thực tế. Nó chỉ quan tâm đến mức độ lỗi trung bình bất kể hướng của chúng. Thêm vào đó MSE giúp tính toán hệ số dốc hiểu quả hơn. Kết quả luôn dương, bình phương có nghĩa là những sai lầm lớn dẫn đến nhiều lỗi hơn sai lầm nhỏ, có nghĩa là mô hình bị phạt nặng nếu mắc sai lầm lớn.
    - Ưu điểm: MSE rất tốt trong việc đảm bảo model được học không dự đoán outlier với chi phí lớn. vì ta đặt trọng số lớn vào các giá trị outlier để giảm chi phí xuống
    - Nhược điểm: Nếu model chúng ta đưa ra dự đoán rất tế thì phần bình phương chi phí lỗi sẽ phóng đại lên,nhạy cảm outlier,tuy nhiên trong thực tế chúng ta không quan tâm mấy đến outlier mà hướng tới một mô hình toàn diện hoạt động đủ tốt với đa số
   
  ![image](https://user-images.githubusercontent.com/72034584/145741309-a1c4b571-f071-41b0-bfd3-98da6c0afcc9.png)

  - Mean absolute error: Đo lường trung bình tổng chêch lệch tuyệt đối giữa giá trị dự đoán và giá trị thực tế, cũng giống như MSE, nó đo độ lớn lớn không cần xem xét hướng. MAE phức tạp hơn cần lập trình tuyến tính thì mới dễ dàng tìm được độ dốc, và MAE mạnh mẽ với các điểm outlier vì nó không bình phương. 
    - Ưu điểm: Chúng ta lấy giá trị tuyết đối, tất cả sai số sẽ được tính theo 1 thang đo tuyến tính, do đó không giống như MSE, không đặt quá nhiều trọng số vào outlier để giảm thang đó chung
    - Nhược điểm: Nếu chúng ta quan tâm tới các giá trị outlier thì MAE sẽ không hiệu quả. Nhưng khi các điểm ngoại lai cực hiếm gặp (như trong đường cong hình chuông), độ đo RMSE lại tốt hơn và được sử dụng phổ biến hơn.

  - Root mean square error: Nó cũng giống như các hàm loss MSE, tính toán thì căn bậc 2 của bình phương trung bình giữa giá trị thực tế và giá trị dự đoán. **Khi đánh giá mức độ phù hợp của 1 mô hình, chúng ta thường sử dụng RMSE bởi vì nó đo lường các đơn vị trong giống biến mục tiêu, MSE thì đo bằng đơn vị bình phương của biến mục tiêu.** 
[Tham khao](https://www.statology.org/mse-vs-rmse/)     
- Classification Losses
     
  ![image](https://user-images.githubusercontent.com/72034584/145745219-eb189a67-4f7e-4bd0-a013-67bf6db7914e.png)

  - Hinge Loss: được dùng để phân loại maximum-margin, đáng chú ý nhất là svm, là 1 hàm liên tục có đạo hàm mọi điểm trừ điểm có hoành độ bằng 1 và hàm lồi
  - Cross Entropy error: Cross Entropy là độ đo giữa hai phân bốp và q để đo lượng trung bình thông tin khi dùng mã hóa thông tin của phân bố q thay cho mã hóa thông tin phân bố p.

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
- Logistics Regression
- SVM
- Decision Tree
