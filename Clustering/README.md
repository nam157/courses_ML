## CLUSTER
- Có thể hiểu là chúng ta phân ra thành các nhóm nhỏ và mỗi nhóm nhỏ đó có nét tương đồng với nhau hoặc là khoảng cách các điểm đó tới tâm là nhỏ nhất
- Về cơ bản có thể hiểu là tập hợp các đối tượng trên cơ sở giống nhau và khác nhau

### KMEAN
- Thuật toán KMEAN có các bước triển khai:
  - Random pick k điểm làm tâm cụm
  - Gán mỗi điểm dữ liệu cho trung tâm gần nhất của nó hoặc có thể là tính toán khoảng cách
  - Cập nhật lại các điểm dữ liệu rồi (tính toán khoảng cách)
  - Lặp lại khi nào các điểm dữ liệu không thay đổi
- The distance measure is the squared Euclidean distance: ![sss](https://user-images.githubusercontent.com/72034584/147743163-da088895-ecec-4509-b152-1ff314320eaf.jpg)

- Một số thuật toán cải tiển: K-mean++,eblow

- [Tham khảo](https://tjmachinelearning.com/lectures/1718/kmeans/kmeans.pdf)


