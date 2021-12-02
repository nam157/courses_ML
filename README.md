### Gradient Descent
- Batch Gradient Descent:
  - Dùng tất cả dữ liệu trong training set cho mỗi lần thực hiện bước tính
đạo hàm
- Stochastic Gradient Descent:
  - Chỉ dùng một dữ liệu ngẫu nhiên (mẫu) trong training set cho mỗi lần thực hiện
bước tính đạo hàm
- Mini-batch gradient descent:
  - Dùng một phần dữ liệu trong training set cho mỗi lần thực hiện
bước tính đạo hàm.

### Bias and Variance
- Bias: nghĩa là độ lệch, biểu thị sự chênh lệch giữa giá trị trung bình mà mô hình dự đoán và giá trị
thực tế của dữ liệu
- Variance: nghĩa là phương sai, biểu thị độ phân tán của các giá trị mà mô hình dự đoán so với giá
trị thực tế.
![1_Y-yJiR0FzMgchPA-Fm5c1Q](https://user-images.githubusercontent.com/72034584/143260447-50f2e59e-892c-415f-8267-56485d1b0d22.jpeg)

Giá trị thật dữ liệu (ground truth) ở giữa tâm các đường tròn. Các dấu X là các giá trị dự đoán. Ta
thấy nếu high bias thì giá trị dự đoán rất xa tâm. Tuy nhiên nếu high variance thì các giá trị dự đoán
phân tán rộng dẫn đến việc ra giá trị thực tế. => Ta mong muốn low bias và low variance
### Data preprocessing
- Handling missing data (Missing data) according to contract MCAR, MAR, MNAR
-  Categorical Encoding
-  Handle outlier và scale data 
### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
### Classification

