import os
# 同时输出到控制台和文件中
import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
	# write()函数这样写，每调用一次就写到记录文件中，不需要等待程序运行结束。
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
	    pass

sys.stdout = Logger('output.log', sys.stdout)
sys.stderr = Logger('output.log_file', sys.stderr)		# redirect std err, if necessary

n = 1800
run_root_depth = 'experiments/dttr_r50_icdar2015/'
for i in range(1020, n, 40):
    run_command = 'python tools/test.py configs/textdet/dttr/dttr_r50dcnv2_cfpn_1200e_icdar2015.py ' + run_root_depth + 'epoch_' + str(i) + '.pth ' + '--eval hmean-iou'
    os.system(run_command)


# python tools/test.py configs/textdet/dttr/dttr_r50dcnv2_cfpn_1200e_icdar2015.py experiments/tdb_r50_query36_ConvTranspose2d/epoch_2000.pth --eval hmean-iou

