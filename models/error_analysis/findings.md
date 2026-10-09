- **Sai số tiền lớn nhất:** price_band = **>=900 triệu**, MAE **103.25 triệu VNĐ**, RMSE 148.73; 135 tin test và 640 tin train. Giá trung vị 999.00 triệu, MAPE 10.16%. Residual trung bình 90.66 triệu: dự đoán trung bình thấp hơn giá rao.

- **Sai số tương đối lớn nhất:** carmodel_name = **Fadil**, MAPE **73.18%**, MAE 31.04 triệu trên 33 tin test; trung vị sai số phần trăm 2.40%. MAPE dùng giá từng tin làm mẫu số nên xe giá thấp có thể sai nhiều theo phần trăm.

- **MAPE bị chi phối bởi ít tin:** hai tin sai nhiều theo phần trăm trong nhóm Fadil có giá niêm yết 27.00, 23.46 triệu, đóng góp 92.97% tổng sai số phần trăm của nhóm. Đây là ảnh hưởng của phần đuôi và mẫu số giá thấp; cần đối chiếu giá của hai tin gốc trước khi kết luận lỗi chung cho cả dòng xe.

- **Ảnh hưởng của mức giá:** nhóm >=900 triệu có MAE 103.25 triệu, MAPE 10.16%; nhóm <300 triệu có MAE 42.04 triệu, MAPE 38.26%. Cần đọc hai chỉ số cùng nhau: cùng một tỷ lệ lệch giá tạo ra số tiền sai lớn hơn ở xe đắt, còn MAPE nhạy với mức giá thấp.

- **Độ phủ dòng xe:** nhóm dòng xe hiếm có 157 tin test, MAE 101.48 triệu và MAPE 35.89%; nhóm dòng xe phổ biến có MAE 42.70 triệu và MAPE 15.58%. Ít mẫu và việc nhiều dòng xe dùng chung mã nhóm hiếm là khả năng cần kiểm tra; khác biệt về giá và loại xe cũng có thể ảnh hưởng kết quả này.

- **Sai số tập trung ở đuôi:** 5% tin có sai số lớn nhất đóng góp 65.64% tổng sai số bình phương. Tỷ trọng này cho biết mức RMSE chịu ảnh hưởng của phần đuôi; bảng tin sai nhiều nhất dùng để kiểm tra lại.

- **Thông tin mô hình đang dùng:** các đặc trưng đứng đầu theo mức giảm R² dương: car_age, carmodel_name, carbrand_name. Tập đầu vào chưa có phiên bản động cơ, cấp trang bị hoặc tình trạng xe đã được kiểm chứng. Khác biệt của những yếu tố này giữa các xe cùng dòng là một giả thuyết cho các tin sai lớn, cần kiểm tra từ tin gốc.


Các nhận xét trên mô tả dữ liệu test và đưa ra khả năng cần kiểm tra, chưa xác nhận quan hệ nhân quả. Giá mục tiêu là giá niêm yết; chênh lệch không tự chứng minh mô hình sai về giá giao dịch hoặc người bán rao sai.