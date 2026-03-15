# FGSM Project (MNIST) — Demo tấn công đối nghịch

Repo này minh hoạ tấn công **FGSM (Fast Gradient Sign Method)** lên mô hình LeNet phân loại chữ số **MNIST** bằng PyTorch.

## Cấu trúc thư mục

- `fgsm.ipynb`: File demo
- `models/lenet_mnist_model.pth`: Trọng số mô hình đã huấn luyện sẵn.
- `dataset/`: Thư mục MNIST tải tự động

## Cách chạy

1. Cài Python và môi trường 
2. Cài thư viện:
   - `torch`
   - `torchvision`
   - `matplotlib`
   - `numpy`
   - `jupyter`

3. Mở file và chạy: `fgsm.ipynb`.

Ghi chú: Cell `datasets.MNIST('dataset', download=True, ...)` sẽ tự tải MNIST về thư mục `dataset/` nếu máy bạn chưa có.

