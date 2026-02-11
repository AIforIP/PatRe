# src/utils.py
import logging
import os

def setup_logger(exp_dir):
    logger = logging.getLogger("PatRe_Eval")
    logger.setLevel(logging.INFO)
    
    # 关键修改：增加 encoding='utf-8'，防止 GBK 报错
    fh = logging.FileHandler(
        os.path.join(exp_dir, "experiment.log"), 
        encoding='utf-8' 
    )
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 防止重复添加 handler
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger