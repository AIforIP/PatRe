import os
import shutil

def clean_empty_experiment_dirs(base_path):
    # 确保路径存在
    if not os.path.exists(base_path):
        print(f"路径不存在: {base_path}")
        return

    # 遍历基准路径下的所有子文件夹
    # 使用 topdown=False 是为了从最深层开始往回删，防止嵌套空目录
    for root, dirs, files in os.walk(base_path, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            
            # 获取当前文件夹下的所有文件和子文件夹
            content = os.listdir(dir_path)
            
            # 定义删除条件：
            # 1. 文件夹内只有一个文件且该文件名为 'experiment.log'
            # 2. 或者文件夹本身就是空的
            is_only_log = (len(content) == 1 and content[0] == 'experiment.log')
            is_empty = (len(content) == 0)

            if is_only_log or is_empty:
                try:
                    # 使用 rmtree 递归删除（虽然这里理论上只有一个文件）
                    shutil.rmtree(dir_path)
                    print(f"已删除无效文件夹: {dir_path}")
                except Exception as e:
                    print(f"删除 {dir_path} 时出错: {e}")

if __name__ == "__main__":
    target_path = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\results\nips_exp"
    clean_empty_experiment_dirs(target_path)