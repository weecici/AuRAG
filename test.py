from src.utils import *

s = """
Ngày 5/11/2025, Ngân hàng Nhà nước Việt Nam công bố quyết định giảm lãi suất điều hành xuống còn 3,5% nhằm kích thích tăng trưởng kinh tế.
Theo ông Nguyễn Văn Hưng, chuyên gia kinh tế tại Đại học Kinh tế Quốc dân, việc giảm lãi suất sẽ hỗ trợ doanh nghiệp nhỏ và vừa tiếp cận vốn dễ dàng hơn, đặc biệt trong lĩnh vực sản xuất.
Tập đoàn Vingroup cũng cho biết đang lên kế hoạch mở rộng đầu tư sang mảng năng lượng tái tạo trong năm 2026.
"""
print(tokenize([s]))
