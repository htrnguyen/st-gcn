## Lưu ý

ST-GCN đã được chuyển sang [MMSkeleton](https://github.com/open-mmlab/mmskeleton),
và tiếp tục phát triển như một công cụ mã nguồn mở linh hoạt cho việc phân tích hành động người dựa trên khung xương.
Chúng tôi khuyến khích bạn chuyển sang sử dụng MMSkeleton mới.
Các mạng tùy chỉnh, bộ nạp dữ liệu và các checkpoint của ST-GCN cũ đều tương thích với MMSkeleton.
Nếu bạn muốn sử dụng ST-GCN phiên bản cũ, vui lòng tham khảo [OLD_README.md](./OLD_README.md).

Code base này sẽ sớm không được duy trì và tồn tại như một di sản lịch sử để bổ sung cho bài báo AAAI của chúng tôi về:

> **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition**, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

### Cập nhật mới

Trong phiên bản này, chúng tôi đã thêm lớp TA (Temporal Attention) vào kiến trúc ST-GCN. Lớp TA được thiết kế để:

1. Xử lý thông tin thời gian tốt hơn thông qua cơ chế attention
2. Tích hợp 3 thành phần chính:
   - Attention theo chiều thời gian (Temporal)
   - Attention theo chiều không gian (Spatial)
   - Attention theo kênh (Channel)
3. Sử dụng các tham số alpha, beta, gamma để cân bằng giữa các thành phần attention

Lớp TA được thêm vào sau mỗi khối ST-GCN, giúp cải thiện hiệu quả nhận dạng hành động.

Để biết thêm về các công trình gần đây, vui lòng xem MMSkeleton.
