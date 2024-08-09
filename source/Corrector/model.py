import py_vncorenlp
class Model():
    def predict(self):
        return


# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
# py_vncorenlp.download_model(save_dir=r'D:\Workspace\python_code\DetectAndRecog\VnCoreNLP')

# # Load the word and sentence segmentation component
# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=r'D:\Workspace\python_code\DetectAndRecog\VnCoreNLP')

# text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

# output = rdrsegmenter.word_segment(text)

# print(output)
