## REGRESSION
- Nếu nhãn không chia thành các nhóm mà các giá trị thực (có thể vô hạn) thì bài toán đó được gọi là hồi quy. 
- Ví dụ : Ước lượng một căn nhà rộng x m2, có y phòng ngủ và cách trung tâm thành phố z km sẽ có giá khoảng bao nhiêu?
### LINEAR REGRESSION
- Linear regression là thuật toán supervised learning, ở đó có quan hệ đầu vào và đầu ra được mô tả bởi một hàm tuyến tính. 

 - Phương trình tổng quát:  <img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;\widehat{y}&space;=&space;f(x)&space;=&space;w_{1}*x_{1}&space;&plus;&space;w_{2}*x_{2}&space;&plus;&space;w_{3}*x_{3}&space;=&space;X^{T}*W}" title="{\color{Blue} \widehat{y} = f(x) = w_{1}*x_{1} + w_{2}*x_{2} + w_{3}*x_{3} = X^{T}*W}" />

      - Trong đó <img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;W&space;=&space;[w_{1},w_{2},w_{3}]^{T}}" title="{\color{Blue} W = [w_{1},w_{2},w_{3}]^{T}}" /> là trọng số. Đây chính là tham số chúng ta cần tìm kiếm. Và <img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;X&space;=&space;[X_{1},X_{2},X_{3}]^{T}}" title="{\color{Blue} X = [X_{1},X_{2},X_{3}]^{T}}" /> là vector cột chứa thông tin giá trị đầu vào. 
 
 - Các phương thức đo độ khớp của mô hình với tập dữ liệu: Mean squared error (MSE), Root mean squared error (RMSE), Mean absolute error (MAE): Đo lường khoảng cách giữa 2 vector, vector thực và vector dự đoán
   - Công thức: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{Blue}&space;MSE(X,W)&space;=&space;\frac{1}{m}&space;\sum_{i=1}^{m}&space;({y_{i}}&space;-&space;W^{T}&space;*&space;X_{i})^{2}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;MSE(X,W)&space;=&space;\frac{1}{m}&space;\sum_{i=1}^{m}&space;({y_{i}}&space;-&space;W^{T}&space;*&space;X_{i})^{2}}" title="{\color{Blue} MSE(X,W) = \frac{1}{m} \sum_{i=1}^{m} ({y_{i}} - W^{T} * X_{i})^{2}}" /></a>
   - Công thức: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{Blue}&space;RMSE(X,W)&space;=\sqrt{\frac{1}{m}&space;\sum_{i=1}^{m}&space;({y_{i}}&space;-&space;(W^{T}&space;*&space;X_{i}))^{2}}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;RMSE(X,W)&space;=\sqrt{\frac{1}{m}&space;\sum_{i=1}^{m}&space;({y_{i}}&space;-&space;(W^{T}&space;*&space;X_{i}))^{2}}}" title="{\color{Blue} RMSE(X,W) =\sqrt{\frac{1}{m} \sum_{i=1}^{m} ({y_{i}} - (W^{T} * X_{i}))^{2}}}" /></a>
   - MSE,RMSE: Chính là chuẩn Euclid (L2)
   - Công thức: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{Blue}&space;MAE(X,W)&space;=&space;\frac{1}{m}&space;\sum_{i=1}^{m}&space;|{y_{i}}&space;-&space;(W^{T}&space;*&space;X_{i})|}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;MAE(X,W)&space;=&space;\frac{1}{m}&space;\sum_{i=1}^{m}&space;|{y_{i}}&space;-&space;(W^{T}&space;*&space;X_{i})|}" title="{\color{Blue} MAE(X,W) = \frac{1}{m} \sum_{i=1}^{m} |{y_{i}} - (W^{T} * X_{i})|}" /></a>
   - MAE: Chính là chuẩn Manhattan (L1)
- Chỉ số chuẩn càng cao thì, chuẩn đó càng tập trung vào các giá trị lớn và bỏ qua các giá trị nhỏ, vì vậy RMSE lại nhạy cảm với dữ liệu outlier hơn MAE.
- Tìm trọng số W:
  - Phương trình pháp tuyến: Đạo hàm hàm mất mát theo W = 0: <img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;\hat{w}&space;=&space;(X^{T}*X)^{-1}&space;*&space;X^{T}*y}" title="{\color{Blue} \hat{w} = (X^{T}*X)^{-1} * X^{T}*y}" />
  - Gradient Descent: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{Blue}&space;w&space;=&space;w&space;-&space;\alpha&space;*&space;\frac{dL(w)}{dw}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;w&space;=&space;w&space;-&space;\alpha&space;*&space;\frac{dL(w)}{dw}}" title="{\color{Blue} w = w - \alpha * \frac{dL(w)}{dw}}" /></a>
- Hạn chế của Linear Regression là nhạy cảm với nhiễu, nó không biểu diễn được mô hình phức tạp 
### RIDGE REGRESSION
- Hồi quy ridge chính là bản tiêu chuẩn của hồi quy tuyến tính, thêm tham số tiêu chuẩn vào hàm mất mát và Hồi quy Ridge giúp chống lại vấn đề overfitting 
- Tham số tiêu chuẩn có thể gọi lambda: Khi lambda = 0 thì bài toán sẽ trở về Hồi quy tuyến tính, Khi lambda càng lớn thì các trọng số w có xu hướng không quá lớn. Vì vậy giúp ta có thể kiểm soát được bài toán và tránh đầu ra không quá phụ thuộc bất kỳ đặc trưng nào. Tham số tiêu chuẩn cho Ridge là l2-norm.
- Công thức: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{Blue}&space;J(X,W)&space;=&space;\frac{1}{m}&space;\sum_{i=1}^{m}&space;({y_{i}}&space;-&space;(W^{T}&space;*&space;X_{i}))^{2}&space;&plus;&space;\lambda&space;\frac{1}{2}\sum_{1}^{m}(w_{i})^{2}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;J(X,W)&space;=&space;\frac{1}{m}&space;\sum_{i=1}^{m}&space;({y_{i}}&space;-&space;(W^{T}&space;*&space;X_{i}))^{2}&space;&plus;&space;\lambda&space;\frac{1}{2}\sum_{1}^{m}(w_{i})^{2}}" title="{\color{Blue} J(X,W) = \frac{1}{m} \sum_{i=1}^{m} ({y_{i}} - (W^{T} * X_{i}))^{2} + \lambda \frac{1}{2}\sum_{1}^{m}(w_{i})^{2}}" /></a>
- Tìm trọng số W: Cũng có thể tìm kiếm bằng 2 cách:
  -  Phương trình pháp tuyến: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{Blue}&space;\hat{W}&space;=&space;(X^{T}*X&space;&plus;&space;\lambda&space;*&space;A)^{-1}&space;*&space;X^{T}y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;\hat{W}&space;=&space;(X^{T}*X&space;&plus;&space;\lambda&space;*&space;A)^{-1}&space;*&space;X^{T}y}" title="{\color{Blue} \hat{W} = (X^{T}*X + \lambda * A)^{-1} * X^{T}y}" /></a>
  -  Gradient Descent: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{Blue}&space;w&space;=&space;w&space;-&space;\alpha&space;*&space;\frac{dL(w)}{dw}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;w&space;=&space;w&space;-&space;\alpha&space;*&space;\frac{dL(w)}{dw}}" title="{\color{Blue} w = w - \alpha * \frac{dL(w)}{dw}}" /></a>
- Trước khi thực hiện hồi rigde cần phải chuấn hóa dữ liệu
### Polynormial Regression
- Nếu dữ liệu phi tuyến, ta có thể thấy lấy mô hình tuyến tính khớp với dữ liệu, Một cách đơn giản để làm việc này đó là lũy thừa dữ liệu lên, sau đó lấy mô hình huấn luyện với dữ liệu mới. 
### LASSO REGRESSION
- Hồi quy lasso chính là bản tiêu chuẩn của hồi quy tuyến tính, thêm tham số tiêu chuẩn vào hàm mất mát và Hồi quy Lasso giúp chống lại vấn đề overfitting 
- Tham số tiêu chuẩn có thể gọi lambda: Khi lambda = 0 thì bài toán sẽ trở về Hồi quy tuyến tính. Khi thêm tham số lambda thì w luôn có xu hướng bằng 0. Vì vậy model Lasso được hay dùng để nén mô hình hoặc lựa chọn đặc trưng. Những trọng số bằng 0 tương ứng đặc trưng đó không được coi trọng còn trọng số khác 0 thì tương ứng quan trọng đóng góp cho kết quả đầu ra. Chuẩn dùng trong Lasso là chuẩn 1(l1-norm). 
- Công thức: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{Blue}&space;J(X,W)&space;=\frac{1}{m}\sum_{i=1}^{m}({y_{i}}-(W^{T}*X_{i}))^{2}&plus;\lambda\frac{1}{2}\sum_{1}^{m}|w_{i}|}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{Blue}&space;J(X,W)&space;=\frac{1}{m}\sum_{i=1}^{m}({y_{i}}-(W^{T}*X_{i}))^{2}&plus;\lambda\frac{1}{2}\sum_{1}^{m}|w_{i}|}" title="{\color{Blue} J(X,W) =\frac{1}{m}\sum_{i=1}^{m}({y_{i}}-(W^{T}*X_{i}))^{2}+\lambda\frac{1}{2}\sum_{1}^{m}|w_{i}|}" /></a>
- Ta cũng có thể dễ thì thấy l1-regularization là đạo hàm của l1-norm không xác định tại 0 (Đạo hàm giá trị tuyến đối). Vì vậy thời gian tìm nghiệm sẽ mất thời gian hơn.
### SUPPORT VECTOR MACHINE
- Classification

   ![image](https://user-images.githubusercontent.com/72034584/146286777-17eb6122-9fdc-4c33-8a41-0bdf1a11f1de.png)

  - Tìm một mặt siêu mặt phẳng chia cắt hay phân tách ra 2 lớp hay nhiều lớp
  - Cần tìm khoảng cách ngắn nhất từ điểm gần nhất mỗi lớp (Support vector) tới mặt phân chia là như nhau, khoảng cách này đường gọi biên/lề (margin)
  - Margin càng rộng thì càng phân tách các lớp rõ ràng hơn trong tập dữ liệu hay tập dữ liệu mới
  - Bài toán tối ưu SVM chính là tìm đường phân chia sao cho margin lớn nhất nhưng cũng hạn chế các vi phạm biên tức là các mẫu nằm trên hoặ trong mặt phằng
  - Với cặp dữ liệu <a href="https://www.codecogs.com/eqnedit.php?latex={\color{Blue}&space;(X_{n},y_{n})}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{Blue}&space;(X_{n},y_{n})}" title="{\color{Blue} (X_{n},y_{n})}" /></a> bất kỳ tới mặt phẳng phân chia <a href="https://www.codecogs.com/eqnedit.php?latex={\color{Blue}&space;W^{T}X&space;&plus;&space;b}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{Blue}&space;W^{T}X&space;&plus;&space;b}" title="{\color{Blue} W^{T}X + b}" /></a> là 
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{Blue}&space;\frac{y_{n}(W^{T}X&plus;b)}{\left&space;\|&space;W&space;\right&space;\|_{2}}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{Blue}&space;\frac{y_{n}(W^{T}X&plus;b)}{\left&space;\|&space;W&space;\right&space;\|_{2}}}" title="{\color{Blue} \frac{y_{n}(W^{T}X+b)}{\left \| W \right \|_{2}}}" /></a>
  - Tìm w,b qua hàm đối ngẫu, điều KKT nó hơi đi sâu toán học sẽ rất khó hiểu với các điều kiệu ràng buộc
  - Chúng ta sẽ tìm hiểu Biên cứng và Biên mềm (hard margin and soft margin)
    - SVM thuần thường gọi là biên cứng, nếu chúng ta nghiệm ngắt bắt buộc các điểm dữ liệu phân tách phải nằm ngoài mặt phẳng phân tách không bị vi phạm biên có gọi là biên cứng
    - Biên cứng nó chỉ hoạt động tốt trên dữ liệu linearly separable (phân tách tuyến tính), và nó rất nhảy cảm với các điểm outlier
    - Biên mềm giải quyết các vấn đề của biến cứng, mục tiêu vẫn là tìm sự cân bằng những vẫn giữ được độ rộng càng lớn càng tốt, đồng thời hạn chế vi phạm biên.
    - Có thể tìm w,b không có điều kiện ràng buộc, tìm nghiệm qua GD
    - Hàm hinge loss là hàm liên tục, có đạo hàm mọi nơi trừ điểm có hoành độ bằng 1, có đạo hàm giống như RELU
    - Công thức: <a href="https://www.codecogs.com/eqnedit.php?latex={\color{Orange}&space;L(w,b)&space;=&space;max(0,1&space;-&space;y_{n}*z_{n})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\color{Orange}&space;L(w,b)&space;=&space;max(0,1&space;-&space;y_{n}*z_{n})}" title="{\color{Orange} L(w,b) = max(0,1 - y_{n}*z_{n})}" /></a>
    - Tuy nhiên nếu không có điều kiện dẫn đến tìm nghiệm nó sẽ không ổn định và có thể lớn vì vậy chúng ta thêm yếu tố L2-regularization
    - Công thức tổng quát: <a href="https://www.codecogs.com/eqnedit.php?latex={\color{Orange}&space;J(w,b)&space;=&space;L(w,b)&space;&plus;&space;\lambda&space;R(w,b)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\color{Orange}&space;J(w,b)&space;=&space;L(w,b)&space;&plus;&space;\lambda&space;R(w,b)}" title="{\color{Orange} J(w,b) = L(w,b) + \lambda R(w,b)}" /></a>
    - Nhớ rằng <a href="https://www.codecogs.com/eqnedit.php?latex={\color{Magenta}&space;1&space;-&space;y_{n}(w^{T}X_{n}&space;&plus;&space;b)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\color{Magenta}&space;1&space;-&space;y_{n}(w^{T}X_{n}&space;&plus;&space;b)}" title="{\color{Magenta} 1 - y_{n}(w^{T}X_{n} + b)}" /></a> là một hàm tuyến tính theo w,b vì vậy là 1 hàm lồi => Hàm loss hinge cũng là 1 hàm lồi và Hàm norm cũng là 1 hàm lồi
    - Tham số điều chỉnh C: C chính là tham số điều chỉnh sự hy sinh trong phân loại biên mềm, C cao thì biên nó sẽ nhỏ lại tức sẽ ko có điểm nào hy sinh bài toán trở về phân loại biên cứng còn C nhỏ biên nó sẽ rộng ra. Việc hy sinh cao hay thấp không ảnh hưởng tới hàm mục tiêu.

- Regression

 ![image](https://user-images.githubusercontent.com/72034584/145607317-2b5444c7-6c2b-43f8-9373-f5ff89e67bf9.png)
 
  - Là một thuật toán supervised learning được sử dụng để dự đoán các giá trị rời rạc, SVR nó cũng giống giống SVMs. Ý tưởng cơ bản là tìm đường tốt nhất, đường tốt nhất là một mặt phẳng có số điểm tối đa.
  - Không giống các regression model là cố gắng tối ưu chi phí lỗi giữa giá trị thực và giá trị dự đoán, còn SVR là cố gắng tối ưu đường tốt nhất trong 1 giá trị ngưỡng, giá trị ngưỡng ở đây chỉ là khoảng cách giữa siêu mặt phẳng và biên. 
  - Với những tập dữ liệu lớn, thường Linear SVR, SDG Regression được sử dụng, Linear SVR cung cấp tính toán nhanh hơn chỉ xét phương diện thay đổi kernel trong SVR
  - Ưu điểm: Thuật toán SVR ít khi sử dụng
    -  Mạnh mẽ với các giá trị outlier
    -  Dễ nhàng triển khai
    -  Mô hình quyết định có thể cập nhật dễ dàng
    -  Khả năng khái quát tuyệt vời và cho độ chính xác cao
  - Nhược điểm:
    -  Không phù hợp với những tập dataset lớn
    -  Nếu số lượng đặc trưng của điểm dữ liệu vượt quá số lượng mẫu đào tạo, SVM sẽ hoạt động kém
    -  Mô hình quyết định sẽ hoạt động tốt nếu có nhiều nhiễu
- Kernel SVM:
  - Những bài toán về dữ liệu phi tuyến, không thể phân tách tuyến tính ta có thể tìm một phép biến đổi dữ liệu từ không gian này sang không gian khác và dữ liệu trở nên phân tách tuyến tính hoặc gần với phân tách tuyến tính. Hoặc có thể nói là tìm một hàm số biến đổi x từ không gian đặc trưng ban đầu thành dữ liệu trong không gian mới bằng 1 hàm f(x). Tuy nhiên trong thực tế, việc tạo dữ liệu với không gian chiều cao hơn ban đầu hoặc có thể hữu hạn, nếu chúng ta tính toán trực tiếp thì gặp rất nhiều khó khăn và tốn bộ nhớ. Có một cách tiếp cận khác đó là tính toán dựa trên kernel mô tả quan sát giữa 2 điểm bất kỳ trong không gian mới. 
  - Chúng ta cũng có thể thấy Kernel có chức năng tương tự như activation trong nerural network
  - Chúng ta không cần tính trực tiếp f(x) mà chỉ cần xác định hàm này: ![CodeCogsEqn](https://user-images.githubusercontent.com/72034584/145822084-a4f5f888-42ca-4c2c-828e-241aa5dc6c94.gif) kỹ thuật này gọi kernel trick. Thay vì tính trực tiếp tọa độ trên không gian mới, ta đi tính tích vô hướng của 2 điểm trong khoogn gian mới.
  - Một số hàm kernel thông dụng: 
    - Linear
    - Polynomial
    - RBF hay Gaussian kernel
    - Sigmoid
  - Tham số điều chỉnh gamma và C: Gamma tăng thì đường phân tách nhỏ lại và ngược lại 
    ![bangKernel png](https://user-images.githubusercontent.com/72034584/145823724-46ad4cbb-7695-44dd-af39-000a774b885a.jpg)
    
### DECISION TREE
- Decision tree là 1 thuật toán thuộc nhóm supervised learing, thuật toán này có thể sử dụng trong 2 bài toán là classification và regression
- Việc xây dựng thuật toán dự trên tập dữ liệu huấn luyện cho trước là việc đi xác định các câu hỏi và thứ tự của chúng.
- Điểm đặc biệt đó là thuật toán có thể làm việc với biến categorical thường rời rạc không thứ tự. Decision tree cũng làm việc với dữ liệu có vector đặc trưng bao gồm cả 2 thuộc tính categorical và numberic. Dữ liệu cũng ko cần chuẩn hóa dữ liệu khi đưa vào huấn luyện
- Có 2 thuật toán phổ biển để triển khai thuật toán: CART (Thuật toán tham lam), ID3
- Thuật toán CART: 
  - Step 1: Tạo 1 cây rỗng  nhị phân
  - Step 2: Lựa chọn đặc trưng để chia các nhóm nhỏ
  - Step 3: Nếu có câu hỏi nào nữa thì đưa ra dự đoán
  - Step 4: Đệ quy lại và chia tiếp step 2
  => Từ thuật toán CART ta thấy có 2 vấn đề cần giải quyết đó: Lựa chọn đặc trưng, Dừng đệ quy
  
- Mô hình cây và mô hình tuyến tính
  -  Nếu mối quan hệ giữa biến độc lập và phụ thuộc bởi mô hình tuyến tỉnh xấp xỉ tốt thì mô hình tuyến tính được sử dụng hơn là mô hình cây
  -  Nếu dữ liệu phi tuyến tính và mối quan hệ phức tạp giữa biến độc lập và phụ thuộc, mô hình cây sẽ hoạt động tốt hơn 
  -  Nếu cần xây dựng 1 mô hình để cho mọi người dễ hiểu thì model cây luôn luôn là tốt nhất
- **Classification**
  - Dựa trên thuật toán CART thì classification cũng được xây dựng như vậy tuy nhiên, cách giải quyết vấn đề để lựa chọn 1 đặc trưng phù hợp để chia thì nó sẽ dựa chí phí phân loại lỗi (hay có thể gọi độ hỗ tạp của các lớp GINI) và dừng đệ quy khi đạt giới hạn yêu cầu (max_depth) hoặc là không thể chia thêm được nữa
  - Để triển khai tính toán chi phí classification error và chi phí lỗi dựa trên độ thuật khiết (Gini hay entropy): 
    - Cho 1 tập dataset
    - Mỗi đặc trưng h(x):
      - Chia dữ liệu theo từng đặc trưng h(x_i)
      - Tính toán chí phí lỗi được chia
    - Chọn đặc trưng với chi phí lỗi thấp nhất   
  - VD: ![image](https://user-images.githubusercontent.com/72034584/146293573-64d121e8-7424-4055-9ec6-ea5b39769591.png)
  - Nó sẽ tính từ nút gốc đi rồi tính các nút trung gian, chi phí lỗi của đặc trưng được chia đó nhỏ nhất thì chọn đặc trưng đó
  - Đầu ra của bài toán sẽ là nút của lớp tương ứng
  - Cây quyết định dễ bị overfit, bởi vì overfit xảy ra khi bạn thiết kế cây quá hoản hảo khớp với dữ liệu traning, thiết kế cây hoản hảo là bạn tăng chiều sâu cây lên làm cho mô hình học quá các chi tiết. 
  - Câu hỏi đặt ra tại sao khi tăng chiều sâu (depth) thì training error lại giảm: Chúng ta sẽ quay về vấn đề lựa chọn đặc trưng để chia.
  - Để tránh vấn đề overfit thì chúng cần đề cập vấn đề cắt tỉa cây.
  - Một đề khá hay đó là: Cây quyết định không đặt giả định về dữ liệu huấn luyện (ngược lại với các mô hình tuyến tính). Nếu ko có ràng buộc thì thuật toán sẽ thích ứng với dữ liệu quá mức. Mô hình như này thường đường gọi là mô hình phi tham số, không phải là không có tham số mà nó số lượng tham số ko tham gia quyết định trước khi huấn luyện mà để mô hình khớp dữ liệu 1 các tự do. Để tránh vấn overfit ta cần đặt ràng buộc thì việc này như chúng ta đã biết đó là tiêu chuẩn
  - Total cost = Classification error  + lambda * num-leaf-node
  
  - Dừng hồi quy cũng rất quan trọng đến vấn đề hiệu suất mô hình chúng ta sẽ nghiên cứu về vấn đề cắt tỉa cây (Pruning) để tăng hiệu suất
  - Độ phức tạp của cây thì tùy thuộc vòng vấn đề bạn số cây bạn chia
  - Phương pháp cắt tỉa cây đơn giản và nhanh chóng làm việc trên mỗi nút lá và tính toán và đánh giá hiệu suất cắt bỏ chúng bằng cách sử dụng bộ thử nghiệm giữ lại 

- **Regression**
  - Dựa trên thuật toán CART thì regression thì nó cũng hoạt động như classification tuy nhiên để chọn đặc trưng phân tách thì nó không tính độ Gini như phân loại mà nó tính tổng MSE và chọn đặc trưng nó có MSE nhỏ nhất. Và dừng hồi quy nó cũng tương tự phân loại
  - Đầu ra sẽ là 1 giá trị 
### RANDOMFOREST
