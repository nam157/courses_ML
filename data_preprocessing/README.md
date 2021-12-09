## TIỀN XỬ LÝ DỮ LIỆU
### CÁC KHÁI NIỆM VÀ TỔNG QUÁT VỀ DỮ LIỆU
- **Dữ liệu bị khuyết**:
  - MNAR: Dữ liệu khuyết không phải ngẫu nhiên nếu có 1 cơ chế hoặc 1 lý do nào đó khiến các giá trị bị khuyết đưa vào tập dữ liệu.
  - MCAR: Dữ liệu khuyết hoàn toàn là ngẫu nghiên, với xác suất thiếu là như nhau.
  - MCR: Dữ liệu khuyết ngẫu nhiên, Xác suất của một quan sát bị thiếu phụ thuộc vào thông tin sẵn có, nó độc lập với các biến khác trong tập dữ liệu.
-**Nhãn hiểm (rare label)**: là những giá trị được chọn trong một nhóm các hạng mục (nhãn). Các nhãn thường xuất hiện trong tập dữ liệu có tần suất khác nhau. Nhãn hiểm có thể thêm nhiều thông tin hoặc không thêm thông tin. 
  - Các nhãn hiểm trong hạng mục có xu hướng gây ra overfitting, đặc biệt trong các thuật toán cây
  - Nhãn hiểm có thể xuất hiện trong  trainning set mà không xuất hiện trong test set gây ra overfitting cho tập training
  - Nhãn hiểm có thể xuất hiện trong tập test mà không xuất hiện trong tập training, như vậy, model sẽ không biết đánh giá nó như thế nào
- **Giả định tuyến tính**:
Một số giả định của mô hình hồi quy tuyến tính:
  - Độ tuyến tính (Linearity): Các giá trị trung bình của biến kết quả với mỗi gia số yếu tố dự đoán nằm dọc theo một đường thắng hay có một mối quan hệ tuyến tính giữa các yếu tố dự báo và mục tiêu
  - Không có đa cộng tuyến hoàn hảo: Không tồn tại mối quan hệ tuyến tính tuyệt hảo giữa 2 biến hoặc nhiều yếu tố dự báo
  - Lỗi phân phối chuẩn: Các phần dư được phân phối chuẩn ngẫu nhiên với giá trị trung bình bằng 0
  - Phương sai không đổi: Ở mức độ của biến dự báo, phương sai của các phân tử không đổi
- **Ngoại lai (Outlier)**: là điểm dữ liệu khác biệt đáng kể so với dữ liệu còn lại." Outlier là quan sát sai lệch rất nhiêu so với quan sát khác, làm dấy lên nghi ngờ rằng nó được tạo ra bằng 1 cơ chế khác. Tùy vào những hoàn cảnh ta loại bỏ outlier hoặc không, có một số mô hình ảnh hưởng outlier khá nhiều. 
  - Nếu biến là phân phối chuẩn thì giá trị nằm ngoài giá trị trung bình cộng/trừ 3 lần độ lệch chuẩn là outlier
    - outlier = mean +/- 3 * std
  - Nếu biến phân phối lệch thì phương pháp tính IQR
    - IQR = Q3 - Q1
  - Outlier sẽ nằm ngoài upper boundary và lower boundary như sau:
    - Upper boundary = Q3 + IQR*1.5
    - Lower boundary = Q1 - IQR*1.5
### XỬ LÝ DỮ LIỆU THIẾU (Missing Data)
- **Xóa giá trị bị thiếu đi**
  - Nếu dữ liệu khuyết hoàn toàn ngẫu nhiên(MCAR)
  - Dữ liệu bị thiếu không quá 5% hoặc ít hơn 5%
    - Ưu điểm:
      - Dễ thực hiện, nhanh chóng
      - Duy trì được phân phối biến( Nếu dữ liệu MCAR thì phân phối sau khi rút gọn phải khớp với phân phối ban đầu)
    - Nhược điểm:
      - Nó có thể là phần lớn của dữ liệu phần đầu
      - Các quan sát bị loại trừ có thể cung cấp thông tin quan trọng
      - Mô hình trong sản xuất, mô hình sẽ không biết các xử lý dữ liệu bị khuyết
- **Gán giá trị trung bình - trung vị**
  - Nếu dữ liệu bị khuyết hoàn toàn ngẫu nhiên MCAR
  - Các quan sát bị khuyết có thể trong giống như phần lớn các quan sát trong biến 
  - Nếu dữ liệu là phân phối chuẩn thì mean và median sẽ xấp xỉ nhau, do đó thay giá trị nào cũng được
  - Nếu dữ liệu có phân bị lệch thì ta sẽ gán median 
  - Dữ liệu bị khuyết không quá 5%, trong thực tế, gán mean/median rất được sử dụng, ngay cả khi dữ liệu ko phải MCAR và khuyết nhiều giá trị
    - Ưu điểm:
      - Dễ thực hiện, nhanh chóng
    - Nhược điểm
      - Làm thay đổi phân phối biến ban đầu
      - Làm thay đổi phương sai ban đầu
      - Làm thay đổi ma trận hiệp phương sai so với các biến còn lại trong tập dữ liệu
- Gán giá trị bất kỳ
- Gán giá trị ở cuối phân phối
- Gán giá trị hạng mục hay xuất hiện
- Gán giá trị bị thiếu với một hạng mục mới
- Gán giá trị ngẫu nhiên
- Gán chỉ số khuyết dữ liệu
- Gán theo KNN



### MÃ HÓA DỮ LIỆU (Encoding Data)
- Mã hóa one-hot
- Mã hóa hạng mục thường xuất hiện
- 

### CHUẨN HÓA DỮ LIỆU (Scale Data)
### LỰA CHỌN ĐẶC TRƯNG (Select feature)
