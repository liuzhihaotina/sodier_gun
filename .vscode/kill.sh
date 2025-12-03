# 清空释放GPU显存，-r选项使得没有进程kill不会报错(不过目前似乎用不上)
ps -ef | grep alive | grep -v grep |awk '{print $2}' | xargs -r kill -9