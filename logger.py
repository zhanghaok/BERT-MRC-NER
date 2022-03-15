import logging
import datetime

#第一步 创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

now_date  = datetime.datetime.now()
now_date = now_date.strftime("%Y-%m-%d_%H-%M-%S")
#第2步，创建一个handler，用于写入日志文件
file_handler = logging.FileHandler("./log/"+str(now_date)+".log",mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
)
# 添加handler到logger中
logger.addHandler(file_handler)

# 第三步，创建一个handler，用于输出到控制台
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(
#     logging.Formatter(
#         fmt='%(asctime)s - %(levelname)s: %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S')  
# )
# logger.addHandler(console_handler)
# logger.critical('this is a logger critical message')

logger.info("BERT-MRC命名实体识别模型")