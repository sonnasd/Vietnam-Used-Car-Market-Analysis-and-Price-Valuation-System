- **Sai số tiền lớn nhất:** carmodel_name = **Ranger**, MAE **109.97 triệu VNĐ**, RMSE 144.20; 119 tin test và 454 tin train. Giá trung vị 610.00 triệu, MAPE 36.33%. Residual trung bình -4.08 triệu: dự đoán trung bình cao hơn giá rao.

- **Sai số tương đối lớn nhất:** carmodel_name = **Fadil**, MAPE **64.41%**, MAE 31.45 triệu trên 33 tin test; trung vị sai số phần trăm 6.24%. MAPE dùng giá từng tin làm mẫu số nên xe giá thấp có thể sai nhiều theo phần trăm.

- **MAPE bị chi phối bởi ít tin:** hai tin sai nhiều theo phần trăm trong nhóm Fadil có giá niêm yết 23.46, 27.00 triệu, đóng góp 90.23% tổng sai số phần trăm của nhóm. Đây là ảnh hưởng của phần đuôi và mẫu số giá thấp; cần đối chiếu giá của hai tin gốc trước khi kết luận lỗi chung cho cả dòng xe.

- **Ảnh hưởng của mức giá:** nhóm >=900 triệu có MAE 102.48 triệu, MAPE 10.10%; nhóm <300 triệu có MAE 41.65 triệu, MAPE 37.53%. Cần đọc hai chỉ số cùng nhau: cùng một tỷ lệ lệch giá tạo ra số tiền sai lớn hơn ở xe đắt, còn MAPE nhạy với mức giá thấp.

- **Độ phủ dòng xe:** nhóm dòng xe hiếm có 157 tin test, MAE 100.20 triệu và MAPE 34.59%; nhóm dòng xe phổ biến có MAE 42.74 triệu và MAPE 15.41%. Ít mẫu và việc nhiều dòng xe dùng chung mã nhóm hiếm là khả năng cần kiểm tra; khác biệt về giá và loại xe cũng có thể ảnh hưởng kết quả này.

- **Sai số tập trung ở đuôi:** 5% tin có sai số lớn nhất đóng góp 66.77% tổng sai số bình phương. Tỷ trọng này cho biết mức RMSE chịu ảnh hưởng của phần đuôi; bảng tin sai nhiều nhất dùng để kiểm tra lại.

- **Thông tin mô hình đang dùng:** các đặc trưng đứng đầu theo mức giảm R² dương: car_age, carmodel_name, carbrand_name. Tập đầu vào chưa có phiên bản động cơ, cấp trang bị hoặc tình trạng xe đã được kiểm chứng. Khác biệt của những yếu tố này giữa các xe cùng dòng là một giả thuyết cho các tin sai lớn, cần kiểm tra từ tin gốc.


Các nhận xét trên mô tả dữ liệu test và đưa ra khả năng cần kiểm tra, chưa xác nhận quan hệ nhân quả. Giá mục tiêu là giá niêm yết; chênh lệch không tự chứng minh mô hình sai về giá giao dịch hoặc người bán rao sai.