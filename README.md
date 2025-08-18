# Chuỗi Bài Giảng Machine Learning với PyTorch
## Từ Những Khối Xây Dựng Nhỏ Nhất

---

## **PHẦN 1: FOUNDATIONS (Nền Tảng)**

### Bài 1: Tensor - Khối Xây Dựng Cơ Bản Nhất
- Tensor là gì? Tại sao cần tensor?
- Tạo tensor, các phép toán cơ bản
- Reshape, indexing, slicing
- **Thực hành**: Tính toán với ma trận

### Bài 2: Automatic Differentiation (Tính Đạo Hàm Tự Động)  
- Gradient là gì? Tại sao cần gradient?
- `requires_grad=True` và `.backward()`
- Chain rule trong thực tế
- **Thực hành**: Tính gradient của hàm đơn giản

### Bài 3: Linear Regression - Mô Hình Đầu Tiên
- Từ công thức toán học đến code
- `y = wx + b` - hiểu từng thành phần
- Loss function, optimizer
- **Thực hành**: Dự đoán giá nhà với 1 feature

---

## **PHẦN 2: NEURAL NETWORK BASICS**

### Bài 4: Perceptron - Neuron Đơn Giản Nhất
- 1 neuron làm việc như thế nào?
- Activation function (sigmoid, tanh, ReLU)
- Forward pass từng bước
- **Thực hành**: Binary classification với 1 neuron

### Bài 5: Multi-Layer Perceptron (MLP)
- Ghép nhiều neuron lại
- Hidden layers là gì?
- Backpropagation giải thích đơn giản
- **Thực hành**: Phân loại hình ảnh MNIST

### Bài 6: PyTorch nn.Module
- Cấu trúc của một neural network class
- `__init__()` và `forward()`
- Parameters vs buffers
- **Thực hành**: Tự xây dựng MLP class

---

## **PHẦN 3: ADVANCED ARCHITECTURES**

### Bài 7: Convolutional Neural Networks (CNN)
- Convolution operation - bộ lọc là gì?
- Pooling, stride, padding
- Từ feature extraction đến classification
- **Thực hành**: Image classification với CNN

### Bài 8: Recurrent Neural Networks (RNN) - Phiên Bản Đơn Giản
- Tại sao cần "nhớ"? 
- Hidden state step by step
- Xử lý sequence dữ liệu
- **Thực hành**: Dự đoán chuỗi số đơn giản

### Bài 9: Long Short-Term Memory (LSTM)
- Vấn đề vanishing gradient của RNN
- Cell state vs hidden state
- Gates mechanism (forget, input, output)
- **Thực hành**: Text sentiment analysis

---

## **PHẦN 4: PRACTICAL ML**

### Bài 10: Data Loading & Preprocessing
- Dataset và DataLoader
- Data augmentation
- Normalization, standardization
- **Thực hành**: Xử lý dataset thực tế

### Bài 11: Training Loop & Validation
- Training loop anatomy
- Overfitting, underfitting
- Validation strategies
- **Thực hành**: Model evaluation đúng cách

### Bài 12: Transfer Learning
- Pre-trained models
- Fine-tuning vs feature extraction  
- Domain adaptation
- **Thực hành**: Sử dụng ResNet cho custom dataset

---

## **PHẦN 5: ADVANCED TOPICS**

### Bài 13: Attention Mechanism
- Attention là gì? Tại sao quan trọng?
- Query, Key, Value concept
- Self-attention cơ bản
- **Thực hành**: Simple attention model

### Bài 14: Transformer Architecture
- Multi-head attention
- Positional encoding
- Encoder-decoder structure
- **Thực hành**: Mini transformer cho sequence tasks

### Bài 15: Generative Models Basics
- Autoencoder
- Variational Autoencoder (VAE) 
- Generative Adversarial Networks (GAN) concept
- **Thực hành**: Image generation đơn giản

---

## **CÁCH TIẾP CẬN HỌC TẬP:**

### 🎯 **Mỗi Bài Sẽ Có:**
1. **Concept**: Giải thích lý thuyết bằng ngôn ngữ đơn giản
2. **Visual**: Hình vẽ, diagram minh họa  
3. **Code**: Từ code cơ bản nhất, giải thích từng dòng
4. **Practice**: Bài tập thực hành nhỏ
5. **Real Example**: Ứng dụng thực tế

### 📚 **Học Tập Tuần Tự:**
- Mỗi bài xây dựng dựa trên bài trước
- Không bỏ qua bài nào
- Code từ đơn giản → phức tạp
- Hiểu **tại sao** trước khi học **cách làm**

### ⚡ **Nguyên Tắc "Building Blocks":**
- Mỗi concept là 1 khối nhỏ
- Kết hợp các khối để tạo hệ thống lớn  
- Hiểu từng khối trước khi kết hợp
- Code minh họa rõ ràng cho từng khối

---

## **BẮT ĐẦU TỪ ĐÂU?**

**Bài đầu tiên** sẽ là **"Tensor - Khối Xây Dựng Cơ Bản Nhất"**

Tôi sẽ giải thích:
- Tensor là gì (như array nhưng mạnh hơn)
- Tại sao ML cần tensor  
- Các phép toán cơ bản
- Code examples từng bước một

**Bạn có muốn bắt đầu với Bài 1 ngay không?** 🚀