import os
from SPfinder import SPfinder

# cxi_name = '/Users/lqhuang/Documents/CSRC/Data/Two-color/cxi_files/f2/cxij6916-r0078-c00.cxi'
# geom_name = '/Users/lqhuang/Documents/CSRC/Data/Two-color/geometry/geometry.h5'

# run_dir = os.path.abspath('data/run78_minsize8')
# run78 = SPfinder(cxi_name, geom_name)
# run78.plot_spots_statistics(dstpath='data/run78_minsize8')
# a, b = run78.find_spots(757)
# run78_size = run78.get_frame_size()
# run78.save_all_spots(run_dir)
# run78.save_spots([756, 757, 758], run_dir)
# run78.plot_spots_statistics(dstpath=run_dir, save=False)
# frame = 0
# # frame = 756
# run78.plot_frame_map(frame, display_peaks=True)
# run78.plot_spots_statistics_for_frame(frame)
# run78.find_split_spots(run_dir, display=False, save=True)


f1_cxi_name = '/Users/lqhuang/Documents/CSRC/Data/Two-color/cxi_files/f1/cxij6916-r0078-c00.cxi'
f2_cxi_name = '/Users/lqhuang/Documents/CSRC/Data/Two-color/cxi_files/f2/cxij6916-r0078-c00.cxi'
geom_name = '/Users/lqhuang/Documents/CSRC/Data/Two-color/geometry/geometry.h5'

f1_dir = os.path.abspath('data/run78_cheetah/f1')
f2_dir = os.path.abspath('data/run78_cheetah/f2')

run78_f2 = SPfinder(f2_cxi_name, geom_name)
# run78_f2.plot_spots_statistics(dstpath=f2_dir, display=True)
# run78_f2.save_all_spots(dstpath=f2_dir)
# run78_f2.find_split_spots(dstpath=f2_dir, display=False, save=True)

run78_f1 = SPfinder(f1_cxi_name, geom_name)
run78_f1.plot_peaks()
# run78_f1.plot_spots_statistics(dstpath=f1_dir, display=True)
# run78_f1.find_split_spots(dstpath=f1_dir, display=False, save=True)
# run78_f1.save_all_spots(dstpath=f1_dir)
